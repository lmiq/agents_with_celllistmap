# Ensure activation of current directory
using Pkg; Pkg.activate(@__DIR__)

include("./WithAgents.jl")
include("./NoAgents.jl")
include("./OnlyAgents.jl")

nsteps = 50

# as far as I can tell, the module `Particles` offers nothing.
# I skip it and add here the benchmarking code immediatelly.
println("With Agents.jl")
WithAgents.simulate(nsteps=1); # compile
@time WithAgents.simulate(nsteps=nsteps);

println("No Agents.jl")
NoAgents.simulate(nsteps=1); # compile
@time NoAgents.simulate(nsteps=nsteps);

println("Only Agents.jl")
OnlyAgents.simulate(nsteps=1); # compile
@time OnlyAgents.simulate(nsteps=nsteps);

println("With Agents.jl, only stepping")
println(" serial: ")
model = WithAgents.initialize_model(parallel=false)
WithAgents.simulate(model=model, nsteps=1); # compile
@time WithAgents.simulate(model=model, nsteps=nsteps);
println(" parallel: ")
model = WithAgents.initialize_model(parallel=true)
WithAgents.simulate(model=model, nsteps=1); # compile
@time WithAgents.simulate(model=model, nsteps=nsteps);

println("No Agents.jl, only stepping")
println(" serial: ")
system = NoAgents.initialize_system(parallel=false)
NoAgents.simulate(system=system, nsteps=1); # compile
@time NoAgents.simulate(system=system, nsteps=nsteps);
println(" parallel: ")
system = NoAgents.initialize_system(parallel=true)
NoAgents.simulate(system=system, nsteps=1); # compile
@time NoAgents.simulate(system=system, nsteps=nsteps);

println("Only Agents.jl, only stepping")
model = OnlyAgents.initialize_model()
OnlyAgents.simulate(model=model, nsteps=1); # compile
@time OnlyAgents.simulate(model=model, nsteps=nsteps);


