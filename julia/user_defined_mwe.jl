using Catalyst, DifferentialEquations, ModelingToolkit, Plots


arrhenius(a, b, c, T) = a * (T / 300.)^b * exp(-c / T)

rn1 = @reaction_network begin
  k1, X --> Y
  k2, Y --> X
end k1 k2

@parameters k1 k2
@variables t X(t) Y(t)
pmap1 = (k1 => arrhenius(1., 2., 3., 300.), k2 => arrhenius(2., 3., 4., 300.))
u0map1 = [X => 1., Y => 1.]
odesys1 = convert(ODESystem, rn1)
prob1 = ODEProblem(odesys1, u0map1, (1e-8, 10), pmap1)
sol1 = solve(prob1)

rn2 = @reaction_network begin
  arrhenius(1., 2., 3., T), X --> Y
  arrhenius(2., 3., 4., T), Y --> X
end T

@parameters T
@variables t X(t) Y(t)
# 'pmap' must be a vector, not a scalar! The only reason we have to use a tuple
# is to include params of different types, otherwise we can just use
# pmap = [T => 300.]
# pmap = (T => 300., )
pmap2 = [T => 300.]
u0map2 = [X => 1., Y => 1.]
odesys2 = convert(ODESystem, rn2)
prob2 = ODEProblem(odesys2, u0map2, (1e-8, 10), pmap2)
sol2 = solve(prob2)

plot(sol1)
plot!(sol2)

savefig("./arrhenius_test.png")