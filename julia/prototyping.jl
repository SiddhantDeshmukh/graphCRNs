# Read density-temperature from file and write out a number density file
##
using Catalyst, DifferentialEquations, ModelingToolkit, TimerOutputs
using CSV, DelimitedFiles, Tables
##
const mass_hydrogen = 1.67262171e-24  # [g]
const to = TimerOutput()
solar_abundances = Dict(
  "H" => 12,
  "H2" => -4,
  "OH" => -12,
  "C" => 8.39,  # solar
  "O" => 8.66,  # solar
  "N" => 7.83,
  "CH" => -12,
  "CO" => -12,
  "CN" => -12,
  "NH" => -12,
  "NO" => -12,
  "C2" => -12,
  "O2" => -12,
  "N2" => -12,
  "M" => 11,
  "X" => 1,
  "Y" => 2,
  "Z" => 3
)

mmm20a04_abundances = Dict(
  "H" => 12,
  "H2" => -4,
  "OH" => -12,
  "C" => 6.41,
  "O" => 7.06,
  "N" => 5.80,
  "CH" => -12,
  "CO" => -12,
  "CN" => -12,
  "NH" => -12,
  "NO" => -12,
  "C2" => -12,
  "O2" => -12,
  "N2" => -12,
  "M" => 11,
  "X" => 1,
  "Y" => 2,
  "Z" => 3
)

##
arrhenius(α, β, γ, T) = α * (T / 300.)^β * exp(-γ/T)

function calculate_number_densities(gas_density, abundances::Dict)
  abu_eps_lin = Dict([k => 10. ^(v - 12.) for (k, v) in abundances])
  abundances_linear = Dict([k => 10. ^v for (k, v) in abundances])
  total_abundances = sum(values(abundances_linear))
  abundances_ratio = Dict([k => v / total_abundances for (k, v) in abundances_linear])
  h_number_density =  gas_density / mass_hydrogen * abundances_ratio["H"]
  number_densities = Dict([k => v * h_number_density for (k, v) in abu_eps_lin])
  return number_densities
end

str_replace(term::Term; token="(t)", replacement="") = replace(string(term), token => replacement)

function create_co_network(to::TimerOutput)
  @timeit to "co network def" co_rn = @reaction_network begin
    # Radiative association
    arrhenius(1.0e-17, 0., 0., T), H + C --> CH
    arrhenius(9.9e-19, -0.38, 0, T), H + O --> OH
    arrhenius(1.58e-17, 0.34, 1297.4, T), C + O --> CO
    # 3-body association
    arrhenius(9.0e-33, -0.6, 0., T), H + H + H2 --> H2 + H2
    arrhenius(4.43e-28, -4., 0., T), H + H + H --> H2 + H
    arrhenius(1.0e-32, 0., 0., T), O + H + H --> OH + H
    arrhenius(2.14e-29, -3.08, -2114., T), C + O + H --> CO + H
    # Species exchange
    arrhenius(2.7e-11, 0.38, 0., T), H + CH --> C + H2
    arrhenius(6.99e-14, 2.8, 1950., T), H + OH --> O + H2
    arrhenius(5.75e-10, 0.5, 77755., T), H + CO --> OH + C
    arrhenius(6.64e-10, 0., 11700, T), H2 + C --> CH + H
    arrhenius(3.14e-13, 2.7, 3150., T), H2 + O --> OH + H
    arrhenius(2.25e-11, 0.5, 14800., T), C + OH --> O + CH
    arrhenius(1.81e-11, 0.5, 0., T), C + OH --> CO + H
    arrhenius(2.52e-11, 0., 2381., T), CH + O --> OH + C
    arrhenius(1.02e-10, 0., 914., T), CH + O --> CO + H
    # Collisional dissociation
    arrhenius(4.67e-7, -1., 55000.,T), H + H2 --> H + H + H
    arrhenius(6.e-9, 0., 40200., T), H + CH --> C + H + H
    arrhenius(6.e-9, 0., 50900., T), H + OH --> O + H + H
    arrhenius(1.e-8, 0., 84100., T), H2 + H2 --> H2 + H + H
    arrhenius(6.e-9, 0., 40200., T), H2 + CH --> C + H2 + H
    arrhenius(6.e-9, 0., 50900., T), H2 + OH --> O + H2 + H
    arrhenius(2.79e-3, -3.52, 128700., T), CO + H --> C + O + H
    # Collision-induced dissociation
    arrhenius(2.79e-3, -3.52, 128700., T), CO + M --> C + O + M
    # Catalysed termolecular reactions
    arrhenius(4.33e-32, -1., 0., T), H + O + M --> OH + M
    arrhenius(6.43e-33, -1., 0., T), H + H + M --> H2 + M
    arrhenius(2.14e-29, -3.08, -2114., T), C + O + M --> CO + M
  end T

  return co_rn
end

function read_density_temperature_file(infile::String)
  arr = readdlm(infile, ',', Float64, skipstart=1)
  return arr
end

function evolve_system(odesys, u0, tspan, p; prob=nothing)
  if (isnothing(prob))
    prob = ODEProblem(odesys, u0, tspan, p)
    de = modelingtoolkitize(prob)
    prob = ODEProblem(de, [], tspan, jac=true)
    sol = solve(prob, Rodas5(); saveeverystep=false)
    return prob, sol
  else
    prob = remake(prob; u0=u0, p=p, tspan=tspan)
    return solve(prob, Rodas5(); saveeverystep=false)[end]
  end
end


function run(to:: TimerOutput, odesys, iter, tspan, abundances)
  # 'iter' must be size(num_points, 2); [:, 1] == density, [:, 2] == temperature
  # TODO:
  # - make a 2D Array instead and use a spread operator to populate in loop
  number_densities = zeros(size(iter)[1], length(species(odesys)))
  l = Threads.SpinLock()
  @timeit to "main run" begin
    # First run
    n = calculate_number_densities(iter[1, 1], abundances)
    u0vals = [n[str_replace(s)] for s in species(odesys)]
    u0 = Pair.(species(odesys), u0vals)
    prob, _ = evolve_system(odesys, u0, tspan, [iter[1, 2]]; prob=nothing)
    # Loop
    @inbounds Threads.@threads for i in 1:size(iter)[1]
    # @inbounds for i in 1:size(iter)[1]
      p = [iter[i, 2]]
      Threads.lock(l)
      n = calculate_number_densities(iter[i, 1], abundances)
      u0 = [n[str_replace(s)] for s in species(odesys)]
      Threads.unlock(l)
      number_densities[i, :] = evolve_system(odesys, u0, tspan, p; prob=prob)
      # number_densities[i, :] = u0
    end
  end

  return number_densities
end

function main(abundances::Dict; precompile=true)
  PROJECT_DIR =  "/home/sdeshmukh/Documents/graphCRNs/julia"
  res_dir = "$(PROJECT_DIR)/res"
  out_dir = "$(PROJECT_DIR)/out"
  co_rn = create_co_network(to)
  if (precompile)
    infile = "$(res_dir)/rho_T_test.csv"
    outfile = "$(out_dir)/catalyst_test.csv"
  else
    infile = "$(res_dir)/rho_T.csv"
    outfile = "$(out_dir)/catalyst.csv"
  end
  arr = read_density_temperature_file(infile)
  odesys = convert(ODESystem, co_rn; combinatoric_ratelaws=false)
  tspan = (1e-8, 1e6)
  n = run(to, odesys, arr, tspan, abundances)
  header = [str_replace(s) for s in species(odesys)]
  table = Tables.table(n; header=header)
  
  CSV.write(outfile, table; delim=',')
end


##
main(mmm20a04_abundances; precompile=true)

##
reset_timer!(to)
main(mmm20a04_abundances; precompile=false)

println()
@show to
println()

# TODO:
# - add types to all funcargs for speeeeeeed
# - I/O for:
#   - abundances
#   - .dat files
#   - .ntw files