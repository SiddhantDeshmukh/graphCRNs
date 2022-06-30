# Test CO network
##
using Catalyst, DifferentialEquations, ModelingToolkit, TimerOutputs
using SteadyStateDiffEq, Plots, DataStructures

const mass_hydrogen = 1.67262171e-24  # [g]

mm00_abundances = Dict(
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
)

mm20a04_abundances = Dict(
  "H" => 12,
  "H2" => -4,
  "OH" => -12,
  "C" => 6.41,  # solar
  "O" => 7.06,  # solar
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
)

mm30a04_abundances = Dict(
  "H" => 12,
  "H2" => -4,
  "OH" => -12,
  "C" => 5.41,  # solar
  "O" => 6.06,  # solar
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
  # Assumes all  reactions are of Arrhenius form
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

function create_co_network()
  co_rn = @reaction_network begin
    # Radiative association
    arrhenius(1.0e-17, 0., 0., T), H + C --> CH
    arrhenius(9.9e-19, -0.38, 0., T), H + O --> OH
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
    arrhenius(6.64e-10, 0., 11700., T), H2 + C --> CH + H
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
    # Species exchange
    arrhenius(2.7e-11, 0.38, 0., T), H + CH --> C + H2
    arrhenius(6.99e-14, 2.8, 1950., T), H + OH --> O + H2
    arrhenius(5.75e-10, 0.5, 77755., T), H + CO --> OH + C
    arrhenius(6.64e-10, 0., 11700., T), H2 + C --> CH + H
    arrhenius(3.14e-13, 2.7, 3150., T), H2 + O --> OH + H
    arrhenius(2.25e-11, 0.5, 14800., T), C + OH --> O + CH
    arrhenius(1.81e-11, 0.5, 0., T), C + OH --> CO + H
    arrhenius(2.52e-11, 0., 2381., T), CH + O --> OH + C
    arrhenius(1.02e-10, 0., 914., T), CH + O --> CO + H
    arrhenius(1.73e-11, 0.5, 2400., T), H + NH --> N + H2  # Diverges when enabled
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
    arrhenius(3.38e-11, -0.17, -2.8, T), N + NO --> N2 + O
    arrhenius(2.26e-12, 0.86, 3134., T), N + O2 --> NO + O
    arrhenius(1.7e-11, 0., 0., T), NH + NH --> N2 + H2
    arrhenius(1.16e-11, 0., 0., T), NH + O --> OH + N
    arrhenius(1.8e-10, 0., 300., T), NH + O --> NO + H
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
    arrhenius(5.12e-12, -0.49, -5.2, T), CN + O2 --> NO + CO
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
    arrhenius(2.3e-10, 0., 3.9, T), N2 --> N + N
    arrhenius(4.7e-10, 0., 2.1, T), NO --> O + N
    arrhenius(7.9e-10, 0., 2.1, T), O2 --> O + O
  end T
  return cno_rn
end

function run_network(rn, u0, tspan, p)
  odesys = convert(ODESystem, rn; combinatoric_ratelaws=false)
  prob = ODEProblem(odesys, u0, tspan, p)
  sol = solve(prob)
  return sol
end

function main(density::Float64, temperature::Float64, abundances::Dict)
  co_rn = create_co_network()
  # rn = create_cno_network()
  network_dir = "/home/sdeshmukh/Documents/graphCRNs/res"
  # network_dir = "/home/sdeshmukh/Documents/graphCRNs/out/networks"
  # network_file = "$(network_dir)/solar_co_w05.ntw"
  network_file = "$(network_dir)/cno.ntw"
  rn = read_network_file(network_file)
  # @show length(species(co_rn)), length(reactions(co_rn))
  # @show length(species(rn)), length(reactions(rn))
  # @show states(rn)
  # @show states(co_rn)
  # println()
  # for (rxn1, rxn2) in zip(reactions(co_rn), reactions(rn))
  #   if (rxn1 != rxn2)
  #     @show rxn1
  #     @show rxn2
  #     println()
  #   end
  # end
  n = calculate_number_densities(density, abundances)
  @parameters T
  u0 = [n[str_replace(s)] for s in species(rn)]
  sol = run_network(rn, u0, (1e-8, 1e6), [T => temperature])
  gr(size=(750, 565))
  display(plot(sol, xaxis=:log, yaxis=:log))
  print([s => log10(n) for (s, n) in zip(species(rn), sol[end])])
  println()

#   sys1 = convert(ODESystem, co_rn; combinatoric_ratelaws=false)
#   sys2 = convert(ODESystem, rn; combinatoric_ratelaws=false)
#   for (eq1, eq2) in zip(sys1.eqs, sys2.eqs)
#     if (eq1 != eq2)
#       @show eq1
#       @show eq2
#       println()
#     end
#   end
 end

##
main(10^(-8), 3500., mm00_abundances)