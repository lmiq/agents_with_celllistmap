module NoAgents

using StaticArrays
import CellListMap

Base.@kwdef struct Particle
    r::Float64 # radius
    k::Float64 # repulsion force constant
    mass::Float64
end

mutable struct CellListMapData{B,C,A,O}
    box::B
    cell_list::C
    aux::A
    output_threaded::O
end

Base.@kwdef struct System{T,CL}
    dt::Float64 = 0.01
    n::Int64 = 10_000
    particles::Vector{Particle}
    positions::Vector{SVector{2,T}}
    velocities::Vector{SVector{2,T}}
    forces::Vector{SVector{2,T}}
    cutoff::Float64
    clmap::CL # CellListMap auxiliary data
    parallel::Bool
end

function initialize_system(;
    n=10_000,
    sides=SVector{2,Float64}(1000.0, 1000.0),
    dt=0.01,
    parallel=true
)
    # initial positions and velocities
    positions = [sides .* rand(SVector{2,Float64}) for _ in 1:n]
    velocities = [-50 .+ 100 .* rand(SVector{2,Float64}) for _ in 1:n]

    # Each particle has a different radius, repulsion constant, and mass
    particles = [
        Particle(
            r=1.0 + 9 * rand(),
            k=1.0 + rand(),
            mass=10.0 + 10 * rand()
        )
        for id in 1:n]

    # maximum radius is 10.0, so cutoff is 20.0 
    cutoff = 20.0

    # initialize array of forces
    forces = zeros(SVector{2,Float64}, n)

    # Define cell list structures needed
    box = CellListMap.Box(sides, cutoff)
    cl = CellListMap.CellList(positions, box, parallel=parallel)
    aux = CellListMap.AuxThreaded(cl)
    output_threaded = [copy(forces) for _ in 1:CellListMap.nbatches(cl)]
    clmap = CellListMapData(box, cl, aux, output_threaded)

    # define the model
    system = System(
        dt=dt,
        n=n,
        particles=particles,
        positions=positions,
        velocities=velocities,
        forces=forces,
        cutoff=cutoff,
        clmap=clmap,
        parallel=parallel,
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
    system.clmap.cell_list = CellListMap.UpdateCellList!(
        system.positions, # current positions
        system.clmap.box,
        system.clmap.cell_list,
        system.clmap.aux;
        parallel=system.parallel
    )
    # reset forces at this step, and auxiliary threaded forces array
    fill!(system.forces, zeros(eltype(system.forces)))
    for i in eachindex(system.clmap.output_threaded)
        fill!(system.clmap.output_threaded[i], zeros(eltype(system.forces)))
    end
    # calculate pairwise forces at this step
    CellListMap.map_pairwise!(
        (x, y, i, j, d2, forces) -> calc_forces!(x, y, i, j, d2, forces, system),
        system.forces,
        system.clmap.box,
        system.clmap.cell_list;
        output_threaded=system.clmap.output_threaded,
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

function simulate(; system=nothing, nsteps=1_000, n=10_000)
    isnothing(system) && (system = initialize_system(n=n))
    for _ in 1:nsteps
        step!(system)
    end
    return system
end

end # module NoAgents
