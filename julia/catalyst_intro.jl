# Testing ring reaction network with Catalyst
##
using Catalyst, DifferentialEquations, ModelingToolkit, TimerOutputs, SteadyStateDiffEq, Plots, StaticArrays
using ProgressBars

##
const mass_hydrogen = 1.67262171e-24  # [g]
const to = TimerOutput()
##

arrhenius(α, β, γ, T) = α * (T / 300.)^β * exp(-γ/T)

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
  "X" => 1,
  "Y" => 2,
  "Z" => 3
)

str_replace(term::Term; token="(t)", replacement="") = replace(string(term), token => replacement)

function create_ring_network(to::TimerOutput)
 @timeit to "ring network def" ring_rn = @reaction_network begin
    (arrhenius(2., 0., 1.4, T), arrhenius(3., 0., 1.5, T)), X <--> Y
    (arrhenius(1., 0., 2.3, T), arrhenius(5., 0., 3.1, T)), Y <--> Z
    (arrhenius(8., 0., 1.9, T), arrhenius(1., 0., 5.3, T)), Z <--> X
  end T

  return ring_rn
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

function calculate_number_densities(gas_density, abundances::Dict)
  abu_eps_lin = Dict([k => 10. ^(v - 12.) for (k, v) in abundances])
  abundances_linear = Dict([k => 10. ^v for (k, v) in abundances])
  total_abundances = sum(values(abundances_linear))
  abundances_ratio = Dict([k => v / total_abundances for (k, v) in abundances_linear])
  h_number_density = gas_density / mass_hydrogen * abundances_ratio["H"]
  number_densities = Dict([k => v * h_number_density for (k, v) in abu_eps_lin])
  return number_densities
end

function setup_ring(to::TimerOutput, num_densities::Int, num_temperatures::Int)
  ring_rn = create_ring_network(to)
  densities = range(0.01, 10.; length=num_densities)
  temperatures = range(50., 500.; length=num_temperatures)
  odesys = convert(ODESystem, ring_rn; combinatoric_ratelaws=false)

  return (densities, temperatures, odesys)
end

function setup_co(to::TimerOutput, num_densities::Int, num_temperatures::Int;
                  min_density=-12., max_density=-6.,
                  min_temperature=1000., max_temperature=15000.)
  co_rn = create_co_network(to)
  densities = 10 .^(range(min_density, max_density; length=num_densities))
  temperatures = range(min_temperature, max_temperature; length=num_temperatures)
  odesys = convert(ODESystem, co_rn; combinatoric_ratelaws=false)

  return (densities, temperatures, odesys)
end

function evolve_system(to:: TimerOutput, odesys, u0, tspan, p; prob=nothing)
  # if (isnothing(prob))
  #   @timeit to "make problem" prob = ODEProblem(odesys, u0, tspan, p)
  #   @timeit to "create jacobian" de = modelingtoolkitize(prob)
  #   @timeit to "make jac problem" prob = ODEProblem(de, [], tspan, jac=true)
  #   @timeit to "1st solve" sol = solve(prob, Rodas5(); saveeverystep=false)
  #   return prob, sol
  # else
  #   @timeit to "remake problem" prob = remake(prob; u0=u0, p=p, tspan=tspan)
  #   @timeit to "solve" return solve(prob, Rodas5(); saveeverystep=false)[end]
  # end
  # Can't use 'to' profiling when multi-threading!
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

function evolve_steady_state(to::TimerOutput, odesys, u0, p; prob=nothing)
  if (isnothing(prob))
    @timeit to "make problem" prob = SteadyStateProblem(odesys, u0, p)
    @timeit to " 1st solve" sol = solve(prob, Rodas5(); saveeverystep=false)
    return prob, sol
  else
    @timeit to "remake problem" prob = remake(prob; u0=u0, p=p)
    @timeit to "solve" return solve(prob, Rodas5(); saveeverystep=false)
  end
end

function run(to:: TimerOutput, odesys, iter, tspan;
             steady_state=false)
  # We solve in a zip(densities, temperatures) loop so these arrays must be
  # provided as: iter = collect(Base.product(densities, temperatures))
  number_densities = Array{Vector{Float64}}(undef, length(iter))
  @timeit to "main run" begin
    # First run
    n = calculate_number_densities(iter[1][1], abundances)
    u0vals = [n[str_replace(s)] for s in species(odesys)]
    u0 = Pair.(species(odesys), u0vals)
    if (steady_state)
      prob, _ = evolve_steady_state(to, odesys, u0, [iter[1][2]];
                                    prob=nothing)
    else
      prob, _ = evolve_system(to, odesys, u0, tspan, [iter[1][2]];
                              prob=nothing)
    end
    # Loop
    for (i, (rho, T)) in collect(enumerate(iter))
      p = [T]
      n = calculate_number_densities(rho, abundances)
      u0 = [n[str_replace(s)] for s in species(odesys)]
      if (steady_state)
        number_densities[i] = evolve_steady_state(to, odesys, u0, p; prob=prob)
      else
        number_densities[i] = evolve_system(to, odesys, u0, tspan, p; prob=prob)
      end
      
      if ((i % 10) == 0)
        print("Done $(i) of $(length(iter))  ($((i)/length(iter)*100)%)  \r")
        flush(stdout)
      end
    end
  end

  return number_densities
end

function run_multithreaded(to:: TimerOutput, odesys, iter, tspan;
                          steady_state=false)
  # We solve in a zip(densities, temperatures) loop so these arrays must be
  # provided as: iter = collect(<SOMETHING>(densities, temperatures))
  # In testing, <SOMETHING> was Base.product, in production, will likely be
  # zip()
  number_densities = Array{Vector{Float64}}(undef, length(iter))
  @timeit to "main run" begin
    # First run
    n = calculate_number_densities(iter[1][1], abundances)
    u0vals = [n[str_replace(s)] for s in species(odesys)]
    u0 = Pair.(species(odesys), u0vals)
    if (steady_state)
      prob, _ = evolve_steady_state(to, odesys, u0, [iter[1][2]];
                                    prob=nothing)
    else
      prob, _ = evolve_system(to, odesys, u0, tspan, [iter[1][2]];
                              prob=nothing)
    end
    # Loop
    Threads.@threads for (i, (rho, T)) in collect(enumerate(iter))
      p = [T]
      n = calculate_number_densities(rho, abundances)
      u0 = [n[str_replace(s)] for s in species(odesys)]
      if (steady_state)
        number_densities[i] = evolve_steady_state(to, odesys, u0, p; prob=prob)
      else
        number_densities[i] = evolve_system(to, odesys, u0, tspan, p; prob=prob)
      end
    end
  end

  return number_densities
end

function main(num_densities::Int, num_temperatures::Int;
              setup=setup_co, tspan=(1e-8, 1e6),
              min_density=-12., max_density=-6.,
              min_temperature=1000., max_temperature=15000.,
              parallel=false, timeit=true)
  densities, temperatures, odesys = setup(to, num_densities,
                                         num_temperatures;
                                         min_density=min_density,
                                         max_density=max_density,
                                         min_temperature=min_temperature,
                                         max_temperature=max_temperature)
  # NOTE:
  # For CO5BOLD I/O, it's likely going to be a zip instead of a product!
  iter = collect(Base.product(densities, temperatures))
  if (parallel)
    if timeit
      @timeit to "complete parallel" run_multithreaded(to, odesys, iter, tspan,
                                              steady_state=false)
    else
      run_multithreaded(to, odesys, iter, tspan, steady_state=false)
    end
    # @timeit to "complete steady-state" run_multithreaded(to, odesys, iter,
    #                                                      tspan, steady_state=true)
  else
    if timeit
      @timeit to "complete serial" run(to, odesys, iter, tspan; steady_state=false)
    else
      run(to, odesys, iter, tspan; steady_state=false)
    end
    # @timeit to "complete steady-state" run(to, odesys, iter, tspan;
                                        # steady_state=true)
  end

  println()
  show(to)
  println()
end

##
# Run once to compile without timing
@timeit to "initial compilation" begin
  main(10, 10; parallel=false, timeit=false)
  main(10, 10; parallel=true, timeit=false)
end

num_trials = 1
num_densities = 1719
num_temperatures = 1719
tspan = (1e0, 1e8)
for i in 1:num_trials
  main(num_densities, num_temperatures; tspan=tspan, parallel=false, timeit=true)
  main(num_densities, num_temperatures; tspan=tspan, parallel=true, timeit=true)
end
# main(1719, 1719; parallel=true)

#=
TODO:
  - Check if solving Steady State, Jacobian ODE (1e-8, 1e6); (1e3, 1e6) give the
    same end result and then use the fastest one. No need to solve for
    1e-8 to 1e6 if we just want the final result!
      - Also make sure it's stable (though it's already way more stable than
        GCRN)
  - Find out if reaction_network and Reaction can be overloaded/modified to
    include an index in the Reaction struct; worst-case scenario, I'll have to
    wrap the calls to @reaction_network with my own struct then overload the
    method to convert it to a Reactionodesystem
  - I/O module for:
      - .ntw files
      - .dat files
      - density-temperature csv files
      - number densities output files
=#