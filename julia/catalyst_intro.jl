# Testing ring reaction network with Catalyst
using Catalyst, DifferentialEquations, ModelingToolkit, Plots

arrhenius(α, β, γ, T) = α * (T / 300.)^β * exp(-γ/T)

ring_rn = @reaction_network begin
  (k1,k2), A <--> B
  (k3,k4), B <--> C
  (k5,k6), C <--> A
end k1 k2 k3 k4 k5 k6
# ode_sys = convert(ODESystem, ring_rn)
T = 300.
p = (:k1 => arrhenius(2., 0, 1.4, T), :k2 => arrhenius(3., 0., 1.5, T),
      :k3 => arrhenius(1., 0., 2.3, T), :k4 => arrhenius(5., 0., 3.1, T),
      :k5 => arrhenius(8., 0., 1.9, T), :k6 => arrhenius(1., 0., 5.3, T))
u₀ = [:A => 1., :B => 1., :C => 1.]
tspan = (1e-8, 1.)
ode_prob = ODEProblem(ring_rn, u₀, tspan, p)
sol = solve(ode_prob, Tsit5())
plot(sol)
savefig("./ring_julia.png")
