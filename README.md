# agents_with_celllistmap

Run with:
```julia
julia> include("./Particles.jl")

julia> using .Particles

julia> WithAgents.simulate() # simulate using Agents.jl

julia> WithAgents.video() # to create the video using Agents.jl

julia> NoAgents.simulate() # simulate without Agents.jl

```