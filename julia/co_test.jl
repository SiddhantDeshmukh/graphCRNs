# Test CO network
##
using Catalyst, DifferentialEquations, ModelingToolkit, TimerOutputs
using SteadyStateDiffEq, Plots

const mass_hydrogen = 1.67262171e-24  # [g]

abundances = Dict(
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

##

arrhenius(a, b, c, T) = a .* (T ./ 300.).^b .* exp.(-c ./ T)

function calculate_number_densities(gas_density, abundances::Dict)
  abu_eps_lin = Dict([k => 10. ^(v - 12.) for (k, v) in abundances])
  abundances_linear = Dict([k => 10. ^v for (k, v) in abundances])
  total_abundances = sum(values(abundances_linear))
  abundances_ratio = Dict([k => v / total_abundances for (k, v) in abundances_linear])
  h_number_density = gas_density / mass_hydrogen * abundances_ratio["H"]
  number_densities = Dict([k => v * h_number_density for (k, v) in abu_eps_lin])
  return number_densities
end

function create_co_network()
  co_rn = @reaction_network begin
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


function run_network(rn, u0, tspan, p)
  odesys = convert(ODESystem, rn; combinatoric_ratelaws=false)
  prob = ODEProblem(odesys, u0, tspan, p)
  sol = solve(prob, Rodas5())
  return sol
end

function main(density::Float64, temperature::Float64)
  co_rn = create_co_network()
  n = calculate_number_densities(density, abundances)
  @parameters T
  @variables t C(t) O(t) M(t) H(t) CO(t) CH(t) OH(t) H2(t)
  u0 = [C => n["C"], O => n["O"], M => n["M"], H => n["H"], CO => n["CO"],
        CH => n["CH"], OH => n["OH"], H2 => n["H2"]]
  sol = run_network(co_rn, u0, (1e-8, 1e6), [T => temperature])
  @show species(co_rn)
  @show sol[end]
  gr(size=(750, 565))
  display(plot(sol, xaxis=:log, yaxis=:log))
end

##
main(1e-12, 3000.)