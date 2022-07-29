# Ensure activation of current directory
using Pkg;
Pkg.activate(@__DIR__);

include("./WithAgents.jl")
include("./NoAgents.jl")
include("./OnlyAgents.jl")

function compare(n, nsteps)
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
    model = WithAgents.initialize_model(n=n, parallel=false)
    WithAgents.simulate(model=model, nsteps=1); # compile
    @time WithAgents.simulate(model=model, nsteps=nsteps);
    println(" parallel: ")
    model = WithAgents.initialize_model(n=n, parallel=true)
    WithAgents.simulate(model=model, nsteps=1); # compile
    @time WithAgents.simulate(model=model, nsteps=nsteps);
    
    println("No Agents.jl, only stepping")
    println(" serial: ")
    system = NoAgents.initialize_system(n=n, parallel=false)
    NoAgents.simulate(system=system, nsteps=1); # compile
    @time NoAgents.simulate(system=system, nsteps=nsteps);
    println(" parallel: ")
    system = NoAgents.initialize_system(n=n, parallel=true)
    NoAgents.simulate(system=system, nsteps=1); # compile
    @time NoAgents.simulate(system=system, nsteps=nsteps);
    
    println("Only Agents.jl, only stepping")
    model = OnlyAgents.initialize_model(n=n)
    OnlyAgents.simulate(model=model, nsteps=1); # compile
    @time OnlyAgents.simulate(model=model, nsteps=nsteps);

    nothing
end

n = 1_000
nsteps = 1000
println("-------------------------")
println(" n = $n ")
println(" nsteps = $nsteps ")
println("-------------------------")
compare(n, nsteps)

n = 10_000
nsteps = 50
println("-------------------------")
println(" n = $n ")
println(" nsteps = $nsteps ")
println("-------------------------")
compare(n, nsteps)

n = 100_000
nsteps = 1
println("-------------------------")
println(" n = $n ")
println(" nsteps = $nsteps ")
println("-------------------------")
compare(n, nsteps)