# Testing ring reaction network with Catalyst
using Catalyst, DifferentialEquations, ModelingToolkit, TimerOutputs, BenchmarkTools

const to = TimerOutput()

arrhenius(α, β, γ, T) = α * (T / 300.)^β * exp(-γ/T)

function run()
  @timeit to "network def" ring_rn = @reaction_network begin
    (k1,k2), A <--> B
    (k3,k4), B <--> C
    (k5,k6), C <--> A
  end k1 k2 k3 k4 k5 k6
  @timeit to "convert ODE" odesys = convert(ODESystem, ring_rn; combinatoric_ratelaws=false)
  tspan = (1e-8, 1.)
  densities = range(0.01, 10.; length=100)
  temperatures = range(50., 500.; length=100)

  @timeit to "main part" begin
    @parameters k1 k2 k3 k4 k5 k6
    @variables t A(t) B(t) C(t)
    p = (k1 => arrhenius(2., 0, 1.4, temperatures[1]), k2 => arrhenius(3., 0., 1.5, temperatures[1]),
        k3 => arrhenius(1., 0., 2.3, temperatures[1]), k4 => arrhenius(5., 0., 3.1, temperatures[1]),
        k5 => arrhenius(8., 0., 1.9, temperatures[1]), k6 => arrhenius(1., 0., 5.3, temperatures[1]))
    u₀ = [A => densities[1], B => densities[1], C => densities[1]]
    @timeit to "problem definition" prob = ODEProblem(odesys, u₀, tspan, p)
    @timeit to "solving first" sol = [solve(prob, abstol=1e-30, reltol=1e-4, saveeverystep=false)]
    @inbounds for rho in densities
      @inbounds for T in temperatures
        p = (k1 => arrhenius(2., 0, 1.4, T), k2 => arrhenius(3., 0., 1.5, T),
              k3 => arrhenius(1., 0., 2.3, T), k4 => arrhenius(5., 0., 3.1, T),
              k5 => arrhenius(8., 0., 1.9, T), k6 => arrhenius(1., 0., 5.3, T))
        u₀ = [A => rho, B => rho, C => rho]
        @timeit to "problem re-definition" remake(prob; u0=u₀, p=p )
        @timeit to "solving" sol[1] = solve(prob, abstol=1e-30, reltol=1e-4, saveeverystep=false)
      end
    end
  end
end

reset_timer!(to)
@timeit to "complete run" run()

show(to)
println()