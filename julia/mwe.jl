using Catalyst, DifferentialEquations, ModelingToolkit, TimerOutputs
using CSV, DelimitedFiles, Tables, Dates, DataStructures, Profile, StatProfilerHTML

Profile.init(; n=floor(Int, 1e8), delay=1e-6)

const mass_hydrogen = 1.67262171e-24  # [g]
const to = TimerOutput()
const abundances = Dict(
  "H" => 12,
  "H2" => -4,
  "OH" => -12,
  "C" => 8.43,  # solar
  "O" => 8.69,  # solar
  "N" => 7.83,
  "CH" => -12,
  "CO" => -12,
  "CN" => -12,
  "NH" => -12,
  "NO" => -12,
  "C2" => -12,
  "O2" => -12,
  "N2" => -12,
  "M" => 11
)

arrhenius(a::Float64, b::Float64, c::Float64, T) = @. a * (T / 300.)^b * exp(-c / T)
str_replace(term::Term; token="(t)", replacement="") = replace(string(term), token => replacement)

function calculate_number_densities(gas_density, abundances::Dict)
  abu_eps_lin = Dict([k => 10. ^(v - 12.) for (k, v) in abundances])
  abundances_linear = Dict([k => 10. ^v for (k, v) in abundances])
  total_abundances = sum(values(abundances_linear))
  abundances_ratio = Dict([k => v / total_abundances for (k, v) in abundances_linear])
  h_number_density = gas_density / mass_hydrogen * abundances_ratio["H"]
  number_densities = Dict([k => v * h_number_density for (k, v) in abu_eps_lin])
  return number_densities
end

function str_to_arrhenius(rxn_str::String)
  # Read Arrhenius rate and return alpha, beta, gamma, i.e.
  # r(T) = alpha * (Tgas/300)**beta * exp(-gamma / Tgas)
  function constants_from_part(part:: String; alpha=0., beta=0., gamma=0.)
    if isdigit(strip(part)[1])
      alpha = parse(Float64, replace(part, "d" => "e"))
    elseif  '^' in part
      beta = parse(Float64, replace(replace(split(part, '^')[end], '(' => ""), ')' => ""))
    elseif startswith(strip(part), "exp")
      # Note minus sign: 'rate' has '-gamma' in it because of Arrhenius form
      gamma = -parse(Float64, split(strip(part), " ")[1][5:end])
    end
    return alpha, beta, gamma
  end

  alpha, beta, gamma = 0., 0., 0.
  # Change '**' to '^' for a unique identifier
  rate = replace(rxn_str, "**" => '^')
  if contains(rate, "*")
    parts = split(rate, "*")
    for part in parts
      part = replace(string(part), 'd' => 'e')
      alpha, beta, gamma = constants_from_part(part;
                                               alpha=alpha, beta=beta, gamma=gamma)
    end
  else
    rate = replace(string(rate), 'd' => 'e')
    alpha, beta, gamma = constants_from_part(rate;
                                            alpha=alpha, beta=beta, gamma=gamma)
  end

  return alpha, beta, gamma

end

function convert_to_symbolic(entries::Vector{String})
  # Convert a list of strings to symbolic variables required by ReactionSystem
  symbolic_vars = Num[]
  parsed_entries = String[]
  @variables t
  for entry in entries
    name = Symbol(entry)
    push!(symbolic_vars, (@variables ($name)($t))[1])
    push!(parsed_entries, entry)
  end

  return symbolic_vars
end

function create_format_dict()
  format_dict = Dict(
    "idx" => [],
    "R" => String[],
    "P" => String[],
    "rate" => [],
    "Tmin" => [],
    "Tmax" => [],
    "limit" => [],
    "ref" => [],
  )
  return format_dict
end

function determine_stoichiometry(entries::Vector{Num})
  # Determine stoichiometry from symbolic variables list to create a unique
  # list with stoichiometric coefficients
  c = counter(entries)
  vars = collect(keys(c))
  stoichiometry = [c[k] for k in vars]
  return vars, stoichiometry
end

function read_network_file(network_file::String)
  # Assumes all reactions are of Arrhenius form
  rxn_format = nothing
  reactions = Reaction[]
  format_dict = create_format_dict()

  @variables t
  open(network_file, "r") do infile
    while true
      line = strip(readline(infile))
      if isempty(line)
        break
      end
      if startswith(line, "#")
        continue
      end

      # Check for format
      if startswith(line, "@format:")
        rxn_format = split(replace(line, "@format:" => ""), ",")
      else
        # Read reaction line
        split_line = split(line, ",")
        for (i, item) in enumerate(rxn_format)
          push!(format_dict[item], string(split_line[i]))
        end
        # Create Reaction
        @parameters a b c T
        a, b, c = str_to_arrhenius(string(format_dict["rate"][1]))
        reactants, reactant_stoich = determine_stoichiometry(convert_to_symbolic(format_dict["R"]))
        products, product_stoich = determine_stoichiometry(convert_to_symbolic(format_dict["P"]))
        # Reactants and products must be unique lists and I need to pass in the
        # stoichiometry!
        reaction = Reaction(arrhenius(a, b, c, T), reactants, products, reactant_stoich, product_stoich)
        push!(reactions, reaction)
        format_dict = create_format_dict()
      end
    end
  end
  @named rn = ReactionSystem(reactions, t)
  return rn
end

function read_density_temperature_file(infile::String)
  arr = readdlm(infile, ',', Float64, skipstart=1)
  return arr
end

function evolve_system(odesys, u0, tspan, p; prob=nothing)
  if (isnothing(prob))
    @timeit to "make problem" prob = ODEProblem(odesys, u0, tspan, p)
    @timeit to "auto jac" de = modelingtoolkitize(prob)
    @timeit to "autojac problem" prob = ODEProblem(de, [], tspan, jac=true)
    @timeit to "1st solve" sol = solve(prob, Rodas5(); save_everystep=false)
    return prob, sol
  else
    @timeit to "remake problem" prob = remake(prob; u0=u0, p=p, tspan=tspan)
    @timeit to "solve" @views return solve(prob, Rodas5(); save_everystep=false)[end]
  end
end


function run(to:: TimerOutput, odesys, iter, tspan, abundances)
  @timeit to "create array" number_densities = zeros(size(iter)[1], length(species(odesys)))
  @timeit to "main run" begin
    # First run
    @timeit to "1st calc n" n = calculate_number_densities(iter[1, 1], abundances)
    u0vals = [n[str_replace(s)] for s in species(odesys)]
    u0 = Pair.(species(odesys), u0vals)
<<<<<<< HEAD
    # TESTING
    prob = ODEProblem(odesys, u0, tspan, [iter[1, 2]])
    de = modelingtoolkitize(prob)
    prob = ODEProblem(de, [], tspan, jac=true)
    solve(prob, Rodas5(); save_everystep=false)
    max_iter = 1000  # for profiling
    reset_timer!(to)
=======
    @timeit to "1st evolve system" prob, _ = evolve_system(odesys, u0, tspan, [iter[1, 2]]; prob=nothing)
    # Loop
>>>>>>> parent of cdf817f... lorenz oscillator benchmark
    @inbounds for i in 1:size(iter)[1]
      if i > max_iter
        return number_densities
      end
      p = [iter[i, 2]]
      @timeit to "calc n" n = calculate_number_densities(iter[i, 1], abundances)
      u0 = [n[str_replace(s)] for s in species(odesys)]
<<<<<<< HEAD
      # @timeit to "remake problem" prob = remake(prob; u0=u0, p=p, tspan=tspan)
      remake(prob; u0=u0, p=p, tspan=tspan)
      @timeit to "solve" number_densities[i, :] = solve(prob, Rodas5(); save_everystep=false)[end]
    # end
    # END TESTING
    # @timeit to "1st evolve system" prob, _ = evolve_system(odesys, u0, tspan, [iter[1, 2]]; prob=nothing)
    # # Loop
    # @inbounds for i in 1:size(iter)[1]
    #   p = [iter[i, 2]]
    #   @timeit to "calc n" n = calculate_number_densities(iter[i, 1], abundances)
    #   u0 = [n[str_replace(s)] for s in species(odesys)]
    #   # @views number_densities[i, :] = evolve_system(odesys, u0, tspan, p; prob=prob)
    #   @timeit to "evolve system" number_densities[i, :] = evolve_system(odesys, u0, tspan, p; prob=prob)
    #   # number_densities[i, :] = u0
    # end
=======
      # @views number_densities[i, :] = evolve_system(odesys, u0, tspan, p; prob=prob)
      @timeit to "evolve system" number_densities[i, :] = evolve_system(odesys, u0, tspan, p; prob=prob)
      # number_densities[i, :] = u0
    end
>>>>>>> parent of cdf817f... lorenz oscillator benchmark
  end

  return number_densities
end

function current_time()
  return Dates.format(now(), "HH:MM")
end

function postprocess_file(infile::String, outfile::String, odesys, tspan,
                          abundances::Dict)
    println("$(current_time()): Postprocessing from $(infile), output to $(outfile)")
    @timeit to "read rho-T file" arr = read_density_temperature_file(infile)
    @timeit to "run" n = run(to, odesys, arr, tspan, abundances)

    header = [str_replace(s) for s in species(odesys)]
    @timeit to "create table" table = Tables.table(n; header=header)
    @timeit to "write csv" CSV.write(outfile, table; delim=',')
end

function main(abundances::Dict, input_dir::String, output_dir::String,
              network_file::String)
  # @show abundances
  @timeit to "read network file" rn = read_network_file(network_file)
  @timeit to "setup odesys" odesys = convert(ODESystem, rn; combinatoric_ratelaws=false)
  tspan = (1e-8, 1e6)
  infile = "$(input_dir)/rho_T_test.csv"
  outfile = "$(output_dir)/catalyst_test.csv"
  @timeit to "postprocess file" postprocess_file(infile, outfile, odesys, tspan, abundances)
  @show to
  println()
end

PROJECT_DIR =  "/home/sdeshmukh/Documents/graphCRNs/julia"
network_dir = "/home/sdeshmukh/Documents/graphCRNs/res"
res_dir = "$(PROJECT_DIR)/res"
out_dir = "$(PROJECT_DIR)/out"
model_id = "d3t63g40mm00chem2"
network_file = "$(network_dir)/cno.ntw"

@profile main(abundances, "$(res_dir)/test", "$(out_dir)/test", network_file)

statprofilehtml()