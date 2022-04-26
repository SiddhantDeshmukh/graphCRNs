# File for sysimage precompile execution
using Catalyst, DifferentialEquations, ModelingToolkit, Plots

function create_network()
  # Create dummy network
  rn = @reaction_network begin
    4e-6 * T, X --> Y,
    3e-5 * T, Y --> X
  end T

  return rn
end

function main()
  rn = create_network()
  odesys = convert(ODESystem, rn; combinatoric_ratelaws=false)
  tspan = (1e-8, 1e1)
  @variables t X(t) Y(t)
  @parameters T
  u0 = [X => 10., Y => 20.]
  p = [T => 300.]
  prob = ODEProblem(odesys, u0, tspan, p)
  de = modelingtoolkitize(prob)
  prob = ODEProblem(de, [], tspan, jac=true)
  sol = [solve(prob, Rodas5(); saveeverystep=false)]

  prob = remake(prob; u0=[100., 1000.], p=[500.], tspan=tspan)
  sol[1] = solve(prob, Rodas5(); saveeverystep=false)
  
  display(plot(sol[1]))
end

main()