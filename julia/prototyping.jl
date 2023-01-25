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
# const mm30a04_abundances = Dict(
#   "H" => 12,
#   "H2" => 2,
#   "OH" => -12,
#   "C" => 5.41,
#   "O" => 6.06,
#   "N" => 4.80,
#   "CH" => 2,
#   "CO" => 2,
#   "CN" => 2,
#   "NH" => 2,
#   "NO" => 2,
#   "C2" => 2,
#   "O2" => 2,
#   "N2" => 2,
#   "M" => 11,
#   "X" => 1,
#   "Y" => 2,
#   "Z" => 3
# )

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

arrhenius(a::Float64, b::Float64, c::Float64, T) = @. a * (T / 300.0)^b * exp(-c / T)
str_replace(term::Term; token="(t)", replacement="") = replace(string(term), token => replacement)

function calculate_number_densities(gas_density, abundances::Dict)
  abu_eps_lin = Dict([k => 10.0^(v - 12.0) for (k, v) in abundances])
  abundances_linear = Dict([k => 10.0^v for (k, v) in abundances])
  total_abundances = sum(values(abundances_linear))
  abundances_ratio = Dict([k => v / total_abundances for (k, v) in abundances_linear])
  h_number_density = gas_density / mass_hydrogen * abundances_ratio["H"]
  number_densities = Dict([k => v * h_number_density for (k, v) in abu_eps_lin])
  return number_densities
end

function str_to_arrhenius(rxn_str::String)
  # Read Arrhenius rate and return alpha, beta, gamma, i.e.
  # r(T) = alpha * (Tgas/300)**beta * exp(-gamma / Tgas)
  function constants_from_part(part::String; alpha=0.0, beta=0.0, gamma=0.0)
    if isdigit(strip(part)[1])
      alpha = parse(Float64, replace(part, "d" => "e"))
    elseif '^' in part
      beta = parse(Float64, replace(replace(split(part, '^')[end], '(' => ""), ')' => ""))
    elseif startswith(strip(part), "exp")
      # Note minus sign: 'rate' has '-gamma' in it because of Arrhenius form
      gamma = -parse(Float64, split(strip(part), " ")[1][5:end])
    end
    return alpha, beta, gamma
  end

  alpha, beta, gamma = 0.0, 0.0, 0.0
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

function evolve_system(odesys, u0, tspan, p; prob=nothing, steadyState=false)
  if (isnothing(prob))
    if (steadyState)
      # Define a SteadyStateProblem
      prob = SteadyStateProblem(odesys, u0, p)
      sol = solve(prob, DynamicSS(Rodas5()); dt=1e-8, alg_hints=[:stiff])
    else
      prob = ODEProblem(odesys, u0, tspan, p)
      de = modelingtoolkitize(prob)
      prob = ODEProblem(de, [], tspan, jac=true)
      sol = solve(prob, Rodas5(); save_everystep=false)
    end
    return prob, sol
  else
    if (steadyState)
      prob = remake(prob; u0=u0, p=p)
      return solve(prob, DynamicSS(Rodas5()); dt=1e-8, alg_hints=[:stiff])
    else
      prob = remake(prob; u0=u0, p=p, tspan=tspan)
      return solve(prob, Rodas5(); save_everystep=false)[end]
    end
  end
end

function run(to::TimerOutput, odesys, iter, tspan, abundances; steadyState=false)
  # 'iter' must be size(num_points, 2); [:, 1] == density, [:, 2] == temperature
  # TODO:
  # - make a 2D Array instead and use a spread operator to populate in loop
  println("Steady State: $(steadyState)")
  number_densities = zeros(size(iter)[1], length(species(odesys)))
  l = Threads.SpinLock()
  @timeit to "main run" begin
    # First run
    n = calculate_number_densities(iter[1, 1], abundances)
    u0vals = [n[str_replace(s)] for s in species(odesys)]
    u0 = Pair.(species(odesys), u0vals)
    prob, _ = evolve_system(odesys, u0, tspan, [iter[1, 2]]; prob=nothing, steadyState=steadyState)
    # Loop
    @inbounds Threads.@threads for i in 1:size(iter)[1]
      # @inbounds Threads.@threads for i in 1:2
      # @inbounds for i in 1:size(iter)[1]
      p = [iter[i, 2]]
      Threads.lock(l)
      n = calculate_number_densities(iter[i, 1], abundances)
      u0 = [n[str_replace(s)] for s in species(odesys)]
      Threads.unlock(l)
      # @views number_densities[i, :] = evolve_system(odesys, u0, tspan, p; prob=prob)
      number_densities[i, :] = evolve_system(odesys, u0, tspan, p; prob=prob, steadyState=steadyState)
      # number_densities[i, :] = u0
    end
  end

  return number_densities
end

function postprocess_file(infile::String, outfile::String, odesys, tspan,
  abundances::Dict; steadyState=false)
  println("$(current_time()): Postprocessing from $(infile), output to $(outfile)")
  arr = read_density_temperature_file(infile)
  n = run(to, odesys, arr, tspan, abundances, steadyState=steadyState)

  header = [str_replace(s) for s in species(odesys)]
  table = Tables.table(n; header=header)
  CSV.write(outfile, table; delim=',')
end

function current_time()
  return Dates.format(now(), "HH:MM")
end

function main(abundances::Dict, input_dir::String, output_dir::String,
  network_file::String; precompile=true, steadyState=false, file_suffix="")
  # @show abundances
  rn = read_network_file(network_file)
  odesys = convert(ODESystem, rn; combinatoric_ratelaws=false)
  tspan = (1e-8, 1e6)
  if (precompile)  # single file test case
    println("Precompiling test!")
    infile = "$(input_dir)/rho_T_test.csv"
    outfile = "$(output_dir)/catalyst$(file_suffix)_test.csv"
    postprocess_file(infile, outfile, odesys, tspan, abundances; steadyState=steadyState)
    @show to
    println()
  else
    # Read files from input dir
    infile_names = readdir(input_dir)
    outfile_names = [replace(f, "rho_T" => "catalyst$(file_suffix)") for f in infile_names]
    infiles = ["$(input_dir)/$(f)" for f in infile_names]
    outfiles = ["$(output_dir)/$(f)" for f in outfile_names]

    for (i, (infile, outfile)) in enumerate(zip(infiles, outfiles))
      postprocess_file(infile, outfile, odesys, tspan, abundances, steadyState=steadyState)
      println("$(current_time()): Postprocessed $(i)/$(length(infiles))")
      @show to
      println()
    end
  end
end

##
PROJECT_DIR = "/home/sdeshmukh/Documents/graphCRNs/julia"
network_dir = "/home/sdeshmukh/Documents/graphCRNs/res"
res_dir = "$(PROJECT_DIR)/res"
out_dir = "$(PROJECT_DIR)/out"

model_id = "d3t63g40mm00chem2"
# model_id = "d3t63g40mm20chem2"
# model_id = "d3t63g40mm30chem2"
# model_id = "d3t63g40mm30c20n20o04chem2"

# network_file = "$(network_dir)/solar_co_w05.ntw"
network_file = "$(network_dir)/cno.ntw"

##
reset_timer!(to)
# @btime main(mm30a04c20n20o04_abundances, "$(res_dir)/test", "$(out_dir)/test", network_file;
#     precompile=true)
@btime main(mm00_abundances, "$(res_dir)/test",
  "$(out_dir)/test", network_file;
  precompile=true, steadyState=false, file_suffix="_evo")
@btime main(mm00_abundances, "$(res_dir)/test",
  "$(out_dir)/test", network_file;
  precompile=true, steadyState=true, file_suffix="_ss")

##
# for idx in 1:10
#   main(mm04c20n20o04_abundances, "$(res_dir)/test",
#               "$(out_dir)/test", network_file;
#               precompile=true)
# end

##
# reset_timer!(to)
# main(mm00_abundances, "$(res_dir)/$(model_id)", "$(out_dir)/$(model_id)",
#     network_file; precompile=false)

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