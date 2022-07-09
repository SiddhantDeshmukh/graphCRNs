# Lorenz equations
##
using DifferentialEquations, Plots, TimerOutputs, BenchmarkTools, StatProfilerHTML
const to = TimerOutput()
##

function lorenz(u, p, t)
  dx = 10.0 * (u[2] - u[1])
  dy = u[1] * (28.0 - u[3]) - u[2]
  dz = u[1] * u[2] - (8/3) * u[3]
  return [dx, dy, dz]
end

function lorenz!(du, u, p, t)
  du[1] = 10.0 * (u[2] - u[1])
  du[2] = u[1] * (28.0 - u[3]) - u[2]
  du[3] = u[1] * u[2] - (8/3) * u[3]
  nothing
end

function parameterized_lorenz!(du,u,p,t)
  du[1] = p[1]*(u[2]-u[1])
  du[2] = u[1]*(p[2]-u[3]) - u[2]
  du[3] = u[1]*u[2] - p[3]*u[3]
  nothing
end

function main()
  u0 = [1.0,0.0,0.0]
  tspan = (0.0,100.0)
  p = [10.0,28.0,8/3]
  # @timeit to "make prob" prob = ODEProblem(parameterized_lorenz!,u0,tspan,p)
  # @timeit to "1st solve" sol = [solve(prob)]
  # for i in 1:100
  #   for j in 1:200
  #     @timeit to "remake prob" remake(prob; p=[i/10., j / 3., p[3]])
  #     @timeit to "solve" sol[1] = solve(prob)
  #   end
  # end
  # @show to
  # println()
  # prob = ODEProblem(lorenz, u0, tspan, p)
  # solve(prob)
  # @benchmark solve($prob, Tsit5())
  prob = ODEProblem(lorenz!, u0, tspan, p)
  solve(prob)
  @benchmark solve($prob, Tsit5())
  # tspan = (0.0,500.0)
  # prob = ODEProblem(lorenz!, u0, tspan, p)
  # solve(prob)
  # @benchmark solve($prob, Tsit5())
end

##
@profilehtml main()