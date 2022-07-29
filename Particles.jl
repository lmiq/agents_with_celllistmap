# Ensure activation of current directory
using Pkg;
Pkg.activate(@__DIR__);

include("./WithAgents.jl")
include("./NoAgents.jl")
include("./OnlyAgents.jl")

n = 1_000
nsteps = 50

# as far as I can tell, the module `Particles` offers nothing.
# I skip it and add here the benchmarking code immediatelly.
println("With Agents.jl")
WithAgents.simulate(nsteps=1, n=n); # compile
@time WithAgents.simulate(nsteps=nsteps, n=n);

println("No Agents.jl")
NoAgents.simulate(nsteps=1, n=n); # compile
@time NoAgents.simulate(nsteps=nsteps, n=n);

println("Only Agents.jl")
OnlyAgents.simulate(nsteps=1, n=n); # compile
@time OnlyAgents.simulate(nsteps=nsteps, n=n);

println("With Agents.jl, only stepping")
println(" serial: ")
model = WithAgents.initialize_model(parallel=false)
WithAgents.simulate(model=model, nsteps=1, n=n); # compile
@time WithAgents.simulate(model=model, nsteps=nsteps, n=n);
println(" parallel: ")
model = WithAgents.initialize_model(parallel=true)
WithAgents.simulate(model=model, nsteps=1, n=n); # compile
@time WithAgents.simulate(model=model, nsteps=nsteps, n=n);

println("No Agents.jl, only stepping")
println(" serial: ")
system = NoAgents.initialize_system(parallel=false)
NoAgents.simulate(system=system, nsteps=1, n=n); # compile
@time NoAgents.simulate(system=system, nsteps=nsteps, n=n);
println(" parallel: ")
system = NoAgents.initialize_system(parallel=true)
NoAgents.simulate(system=system, nsteps=1, n=n); # compile
@time NoAgents.simulate(system=system, nsteps=nsteps, n=n);

println("Only Agents.jl, only stepping")
model = OnlyAgents.initialize_model()
OnlyAgents.simulate(model=model, nsteps=1, n=n); # compile
@time OnlyAgents.simulate(model=model, nsteps=nsteps, n=n);


