# Ensure activation of current directory
using Pkg; Pkg.activate(@__DIR__)

include("./WithAgents.jl")
include("./NoAgents.jl")

# as far as I can tell, the module `Particles` offers nothing.
# I skip it and add here the benchmarking code immediatelly.
println("With agents")
WithAgents.simulate(); # compile
@time WithAgents.simulate();

println("No agents")
NoAgents.simulate(); # compile
@time NoAgents.simulate();
