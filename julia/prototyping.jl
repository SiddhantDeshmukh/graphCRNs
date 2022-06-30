# Read density-temperature from file and write out a number density file
##
using Catalyst, DifferentialEquations, ModelingToolkit, TimerOutputs
using CSV, DelimitedFiles, Tables, Dates, DataStructures, BenchmarkTools
using ProfileView
##
const mass_hydrogen = 1.67262171e-24  # [g]
const to = TimerOutput()
const mm00_abundances = Dict(
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
  "M" => 11,
  "X" => 1,
  "Y" => 2,
  "Z" => 3
)

const mm20a04_abundances = Dict(
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

const mm30a04_abundances = Dict(
  "H" => 12,
  "H2" => -4,
  "OH" => -12,
  "C" => 5.41,
  "O" => 6.06,
  "N" => 4.80,
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

const mm30a04c20n20o04_abundances = Dict(
  "H" => 12,
  "H2" => -4,
  "OH" => -12,
  "C" => 7.39,
  "O" => 6.06,
  "N" => 6.78,
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

const mm30a04c20n20o20_abundances = Dict(
  "H" => 12,
  "H2" => -4,
  "OH" => -12,
  "C" => 7.39,
  "O" => 7.66,
  "N" => 6.78,
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

function create_cno_network()
  cno_rn = @reaction_network begin
    # Radiative association
    arrhenius(1.0e-17, 0., 0., T), H + C --> CH
    arrhenius(9.9e-19, -0.38, 0., T), H + O --> OH
    arrhenius(1.58e-17, 0.34, 1297.4, T), C + O --> CO
    arrhenius(4.36e-18, 0.35, 161.3, T), C + C --> C2
    arrhenius(5.72e-19, 0.37, 51., T), C + N --> CN
    arrhenius(4.9e-20, 1.58, 0., T), O + O --> O2
    # 3-body association
    arrhenius(9.0e-33, -0.6, 0., T), H + H + H2 --> H2 + H2
    arrhenius(4.43e-28, -4., 0., T), H + H + H --> H2 + H
    arrhenius(1.0e-32, 0., 0., T), O + H + H --> OH + H
    arrhenius(2.14e-29, -3.08, -2114., T), C + O + H --> CO + H
    # # Species exchange
    arrhenius(2.7e-11, 0.38, 0., T), H + CH --> C + H2
    arrhenius(6.99e-14, 2.8, 1950., T), H + OH --> O + H2
    arrhenius(5.75e-10, 0.5, 77755., T), H + CO --> OH + C
    arrhenius(6.64e-10, 0., 11700., T), H2 + C --> CH + H
    arrhenius(3.14e-13, 2.7, 3150., T), H2 + O --> OH + H
    arrhenius(2.25e-11, 0.5, 14800., T), C + OH --> O + CH
    arrhenius(1.81e-11, 0.5, 0., T), C + OH --> CO + H
    arrhenius(2.52e-11, 0., 2381., T), CH + O --> OH + C
    arrhenius(1.02e-10, 0., 914., T), CH + O --> CO + H
    # arrhenius(1.73e-11, 0.5, 2400., T), H + NH --> N + H2
    arrhenius(4.67e-10, 0.5, 30450., T), H + C2 --> CH + C
    arrhenius(9.29e-10, -0.1, 35220., T), H + NO --> O + NH
    arrhenius(3.6e-10, 0., 24910., T), H + NO --> OH + N
    arrhenius(2.61e-10, 0., 8156., T), H + O2 --> OH + O
    arrhenius(1.69e-9, 0., 18095., T), H2 + N --> NH + H
    arrhenius(3.16e-10, 0., 21890., T), H2 + O2 --> OH + OH
    arrhenius(6.59e-11, 0., 0., T), C + CH --> C2 + H
    arrhenius(1.73e-11, 0.5, 4000., T), C + NH --> N + CH
    arrhenius(1.2e-10, 0., 0., T), C + NH --> CN + H
    arrhenius(4.98e-10, 0., 18116., T), C + CN --> C2 + N
    arrhenius(2.94e-11, 0.5, 58020., T), C + CO --> C2 + O
    arrhenius(8.69e-11, 0., 22600., T), C + N2 --> CN + N
    arrhenius(6e-11, -0.16, 0., T), C + NO --> CN + O
    arrhenius(9.0e-11, -0.16, 0., T), C + NO --> CO + N
    arrhenius(5.56e-11, 0.41, -26.9, T), C + O2 --> CO + O
    arrhenius(3.03e-11, 0.65, 1207., T), CH + N --> NH + C
    arrhenius(7.6e-12, 0., 0., T), CH + O2 --> CO + OH
    arrhenius(4.98e-11, 0., 0., T), N + NH --> N2 + H
    arrhenius(1.88e-11, 0.1, 10700., T), N + OH --> O + NH
    arrhenius(6.05e-11, -0.23, 14.9, T), N + OH --> NO + H
    arrhenius(5.0e-11, 0., 0., T), N + C2 --> CN + C
    arrhenius(1.0e-10, 0.4, 0., T), N + CN --> N2 + C
    arrhenius(3.38e-11, -0.17, 2.8, T), N + NO --> N2 + O
    arrhenius(2.26e-12, 0.8, 3134., T), N + O2 --> NO + O
    arrhenius(1.7e-11, 0., 0., T), NH + NH --> N2 + H2
    arrhenius(1.16e-11, 0., 0., T), NH + O --> OH + N
    arrhenius(1.7e-10, 0., 300., T), NH + O --> NO + H
    arrhenius(1.46e-11, -0.58, 37., T), NH + NO --> N2 + OH
    arrhenius(1.77e-11, 0., -178., T), O + OH --> O2 + H
    arrhenius(2.0e-10, -0.12, 0., T), O + C2 --> CO + C
    arrhenius(5.37e-11, 0., 13800., T), O + CN --> NO + C
    arrhenius(5.e-11, 0., 200., T), O + CN --> CO + N
    arrhenius(2.51e-10, 0., 38602., T), O + N2 --> NO + N
    arrhenius(1.18e-11, 0., 20413., T), O + NO --> O2 + N
    arrhenius(1.5e-11, 0., 4300., T), C2 + O2 --> CO + CO
    arrhenius(2.66e-9, 0., 21638., T), CN + CN --> N2 + C2
    arrhenius(1.6e-13, 0., 0., T), CN + NO --> N2 + CO
    arrhenius(5.12e-12, -0.49, 5.2, T), CN + O2 --> NO + CO
    arrhenius(2.51e-11, 0., 30653., T), NO + NO --> O2 + N2
    arrhenius(2.54e-14, 1.18, 312., T), NH + O2 --> NO + OH
    # Collisional dissociation
    arrhenius(4.67e-7, -1., 55000.,T), H + H2 --> H + H + H
    arrhenius(6.e-9, 0., 40200., T), H + CH --> C + H + H
    arrhenius(6.e-9, 0., 50900., T), H + OH --> O + H + H
    arrhenius(1.e-8, 0., 84100., T), H2 + H2 --> H2 + H + H
    arrhenius(6.e-9, 0., 40200., T), H2 + CH --> C + H2 + H
    arrhenius(6.e-9, 0., 50900., T), H2 + OH --> O + H2 + H
    arrhenius(2.79e-3, -3.52, 128700., T), CO + H --> C + O + H
    arrhenius(1.16e-9, 0., 0., T), NH + NH --> N2 + H + H
    arrhenius(7.4e-10, 0., 10540., T), NH + NO --> N2 + O + H
    arrhenius(6.e-9, 0., 52300., T), H + O2 --> O + O + H
    arrhenius(6.e-9, 0., 52300., T), H2 + O2 --> O + O + H2
    arrhenius(1.14e-11, 0., 0., T), CH + O2 --> CO + O + H
    # Collision-induced dissociation
    arrhenius(2.79e-3, -3.52, 128700., T), CO + M --> C + O + M
    # Catalysed termolecular reactions
    arrhenius(4.33e-32, -1., 0., T), H + O + M --> OH + M
    arrhenius(6.43e-33, -1., 0., T), H + H + M --> H2 + M
    arrhenius(2.14e-29, -3.08, -2114., T), C + O + M --> CO + M
    # Radiative dissociation
    arrhenius(5.0e-10, 0., 2.3, T), NH --> N + H
    arrhenius(2.4e-10, 0., 2.6, T), C2 --> C + C
    arrhenius(2.9e-10, 0., 3.5, T), CN --> C + N
    arrhenius(2.0e-10, 0., 3.5, T), CO --> C + O
    arrhenius(2.3e-10, 0., 3.9, T), N2 --> N + N,
    arrhenius(4.7e-10, 0., 2.1, T), NO --> O + N
    arrhenius(7.9e-10, 0., 2.1, T), O2 --> O + O
  end T
  return cno_rn
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
      # @views number_densities[i, :] = evolve_system(odesys, u0, tspan, p; prob=prob)
      number_densities[i, :] = evolve_system(odesys, u0, tspan, p; prob=prob)
      # number_densities[i, :] = u0
    end
  end

  return number_densities
end

function postprocess_file(infile::String, outfile::String, odesys, tspan,
                          abundances::Dict)
    println("$(current_time()): Postprocessing from $(infile), output to $(outfile)")
    arr = read_density_temperature_file(infile)
    n = run(to, odesys, arr, tspan, abundances)

    header = [str_replace(s) for s in species(odesys)]
    table = Tables.table(n; header=header)
    CSV.write(outfile, table; delim=',')
end

function current_time()
  return Dates.format(now(), "HH:MM")
end

function main(abundances::Dict, input_dir::String, output_dir::String,
              network_file::String; precompile=true)
  # @show abundances
  rn = read_network_file(network_file)
  odesys = convert(ODESystem, rn; combinatoric_ratelaws=false)
  tspan = (1e-8, 1e6)
  if (precompile)  # single file test case
    println("Precompiling test!")
    infile = "$(input_dir)/rho_T_test.csv"
    outfile = "$(output_dir)/catalyst_test.csv"
    postprocess_file(infile, outfile, odesys, tspan, abundances)
    @show to
    println()
  else
    # Read files from input dir
    infile_names = readdir(input_dir)
    outfile_names = [replace(f, "rho_T" => "catalyst") for f in infile_names]
    infiles = ["$(input_dir)/$(f)" for f in infile_names]
    outfiles = ["$(output_dir)/$(f)" for f in outfile_names]

    for (i, (infile, outfile)) in enumerate(zip(infiles, outfiles))
      postprocess_file(infile, outfile, odesys, tspan, abundances)
      println("$(current_time()): Postprocessed $(i)/$(length(infiles))")
      @show to
      println()
    end
  end
end

##
PROJECT_DIR =  "/home/sdeshmukh/Documents/graphCRNs/julia"
network_dir = "/home/sdeshmukh/Documents/graphCRNs/res"
res_dir = "$(PROJECT_DIR)/res"
out_dir = "$(PROJECT_DIR)/out"
model_id = "d3t63g40mm30c20n20o04chem2"
network_file = "$(network_dir)/cno.ntw"

##
reset_timer!(to)
# @btime main(mm30a04c20n20o04_abundances, "$(res_dir)/test", "$(out_dir)/test", network_file;
#     precompile=true)
main(mm30a04c20n20o04_abundances, "$(res_dir)/test",
              "$(out_dir)/test", network_file;
              precompile=true)

##
for idx in 1:10
  main(mm30a04c20n20o04_abundances, "$(res_dir)/test",
              "$(out_dir)/test", network_file;
              precompile=true)
end

##
reset_timer!(to)
main(mm30a04c20n20o04_abundances, "$(res_dir)/$(model_id)", "$(out_dir)/$(model_id)",
    network_file; precompile=false)

# TODO:
  # - add types to all funcargs for speeeeeeed
# - add saving at intervals, since these are just 1D arrays now, restarting
#   becomes very easy
# - check if output file exists, skip if completed
# - I/O for:
#   - abundances
#   - .dat files
# - command line arguments to automatically choose correct options based on
#   model ID passed in
# - proper modules and imports/exports
# - allow for different abundances and other solver options
#   - would be best to perhaps read from a config file (that could be written
#     from bash for automation)