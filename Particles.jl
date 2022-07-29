# Ensure activation of current directory
using Pkg; Pkg.activate(@__DIR__)

include("./WithAgents.jl")
include("./NoAgents.jl")
include("./OnlyAgents.jl")

# as far as I can tell, the module `Particles` offers nothing.
# I skip it and add here the benchmarking code immediatelly.
println("With Agents.jl")
WithAgents.simulate(); # compile
@time WithAgents.simulate();

println("No Agents.jl")
NoAgents.simulate(); # compile
@time NoAgents.simulate();

println("Only Agents.jl")
OnlyAgents.simulate(); # compile
@time OnlyAgents.simulate();

println("With Agents.jl, only stepping")
model = WithAgents.initialize_model()
WithAgents.simulate(model); # compile
@time WithAgents.simulate(model);

println("No Agents.jl, only stepping")
system = NoAgents.initialize_system()
NoAgents.simulate(system); # compile
@time NoAgents.simulate(system);

println("Only Agents.jl, only stepping")
model = OnlyAgents.initialize_model()
OnlyAgents.simulate(model); # compile
@time OnlyAgents.simulate(model);
