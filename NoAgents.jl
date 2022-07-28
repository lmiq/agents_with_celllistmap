module NoAgents

using StaticArrays
import CellListMap

struct Particle
    r::Float64 # radius
    k::Float64 # repulsion force constant
    mass::Float64
end
Particle() = Particle(10.0, 1.0, 1.0)

Base.@kwdef mutable struct System{T,B,C}
    dt::Float64 = 0.01
    n::Int64 = 0
    particles::Vector{Particle}
    positions::Vector{SVector{2,T}}
    velocities::Vector{SVector{2,T}}
    forces::Vector{SVector{2,T}}
    cutoff::Float64
    box::B
    cell_list::C
    parallel::Bool = false
end

function initialize_system(;
    n=1000,
    sides=SVector{2,Float64}(1000.0, 1000.0),
    dt=0.01,
    parallel=false
)
    # initial positions and velocities
    positions = [sides .* rand(SVector{2,Float64}) for _ in 1:n]
    velocities = [-50 .+ 100 .* rand(SVector{2,Float64}) for _ in 1:n]
    particles = [Particle() for id in 1:n]

    # initialize array of forces
    forces = zeros(SVector{2,Float64}, n)

    # cutoff is twice the maximum radius among particles
    cutoff = maximum(2 * p.r for p in particles)

    # Define cell list structure
    box = CellListMap.Box(sides, cutoff)
    cl = CellListMap.CellList(positions, box; parallel=parallel)

    # define the model
    system = System(
        dt=dt,
        n=n,
        cutoff=cutoff,
        particles=particles,
        positions=positions,
        velocities=velocities,
        forces=forces,
        box=box,
        cell_list=cl,
        parallel=parallel
    )

    return system
end

#
# This function udpates the model.forces array for each interacting pair
# The potential is a smooth short-ranged interaction:
#
# U(r) = (ki*kj)*(r^2 - (ri+rj)^2)^2 for r ≤ (ri+rj)
# U(r) = 0.0 for r > (ri+rj)
#
# where ri and rj are the radii of the two agents, and ki and kj are the
# potential energy constants.
#
function calc_forces!(x, y, i, j, d2, forces, system)
    ri = system.particles[i].r
    rj = system.particles[j].r
    d = sqrt(d2)
    if d ≤ (ri + rj)
        ki = system.particles[i].k
        kj = system.particles[j].k
        dr = y - x
        fij = 2 * (ki * kj) * (d2 - (ri + rj)^2) * (dr / d)
        forces[i] += fij
        forces[j] -= fij
    end
    return forces
end

function step!(system::System)
    # update cell lists
    system.cell_list = CellListMap.UpdateCellList!(
        system.positions, # current positions
        system.box,
        system.cell_list;
        parallel=system.parallel
    )
    # reset forces at this step, and auxiliary threaded forces array
    fill!(system.forces, zeros(eltype(system.forces)))
    # calculate pairwise forces at this step
    CellListMap.map_pairwise!(
        (x, y, i, j, d2, forces) -> calc_forces!(x, y, i, j, d2, forces, system),
        system.forces,
        system.box,
        system.cell_list;
        parallel=system.parallel
    )
    # Update positions and velocities
    dt = system.dt
    for i in eachindex(system.particles)
        p = system.particles[i]
        x = system.positions[i]
        v = system.velocities[i]
        f = system.forces[i]
        a = f / p.mass
        system.positions[i] = x + v * dt + (a / 2) * dt^2
        system.velocities[i] = v + f * dt
    end
end

function simulate(; n=1000, nsteps=1_000, parallel=false)
    system = initialize_system(n=n, parallel=parallel)
    for _ in 1:nsteps
        step!(system)
    end
    return system
end

function simulate(system = initialize_system(n=1000, parallel=false); nsteps=1_000)
    for _ in 1:nsteps
        step!(system)
    end
    return system
end

end # module NoAgents
