module WithAgents

using Agents
using StaticArrays
import CellListMap

@agent Particle ContinuousAgent{2} begin
    r::Float64 # radius
    k::Float64 # repulsion force constant
    mass::Float64
end
Particle(; id, pos, vel, r, k, mass) = Particle(id, pos, vel, r, k, mass)

# Structure that contains the data to use CellListMap
mutable struct CellListMapData{B,C,A,O}
    box::B
    cell_list::C
    aux::A
    output_threaded::O
end

Base.@kwdef struct Properties{T,CL}
    dt::Float64 = 0.01
    n::Int64 = 0
    positions::Vector{SVector{2,T}}
    velocities::Vector{SVector{2,T}}
    forces::Vector{SVector{2,T}}
    cutoff::Float64
    clmap::CL # CellListMap data
    parallel::Bool
end

function initialize_model(;
    n=10_000,
    sides=SVector{2,Float64}(1000.0, 1000.0),
    dt=0.01,
    parallel=true
)
    # initial positions and velocities
    positions = [sides .* rand(SVector{2,Float64}) for _ in 1:n]
    velocities = [-50 .+ 100 .* rand(SVector{2,Float64}) for _ in 1:n]

    # Space and agents
    space2d = ContinuousSpace(Tuple(sides); periodic=true)

    # Each particle has a different radius, repulsion constant, and mass
    particles = [
        Particle(
            id=id,
            r=1.0 + 10 * rand(),
            k=1.0 + rand(),
            mass=10.0 + 10 * rand(),
            pos=Tuple(positions[id]),
            vel=Tuple(velocities[id]),
        )
        for id in 1:n]


    # initialize array of forces
    forces = zeros(SVector{2,Float64}, n)

    # maximum radius is 10.0 thus cutoff is 20.0
    cutoff = 20.0

    # Define cell list structure
    box = CellListMap.Box(sides, cutoff)
    cl = CellListMap.CellList(positions, box; parallel=parallel)
    aux = CellListMap.AuxThreaded(cl)
    output_threaded = [copy(forces) for _ in 1:CellListMap.nbatches(cl)]
    clmap = CellListMapData(box, cl, aux, output_threaded)

    # define the model
    properties = Properties(
        dt=dt,
        n=n,
        cutoff=cutoff,
        positions=positions,
        velocities=velocities,
        forces=forces,
        clmap=clmap,
        parallel=parallel,
    )
    model = ABM(Particle,
        space2d,
        properties=properties
    )

    # create active rods
    for id in 1:n
        add_agent_pos!(particles[id], model)
    end

    return model
end

# Here the agent positions and velocities get updated with simple Euler integration
# steps. Seems that this could/should be easy to be substituted by some better integration
# scheme of DifferentialEquations.
function agent_step!(agent, model::ABM)
    id = agent.id
    x = model.positions[id]
    v = model.velocities[id]
    f = model.forces[id]
    dt = model.properties.dt
    a = f / model.agents[id].mass
    x_new = x + v * dt + (a / 2) * dt^2
    v_new = v + f * dt
    model.positions[id] = x_new
    model.velocities[id] = v_new
    agent.pos = Tuple(x_new)
    agent.vel = Tuple(v_new)
    return
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
function calc_forces!(x, y, i, j, d2, forces, model)
    ri = model.agents[i].r
    rj = model.agents[j].r
    d = sqrt(d2)
    if d ≤ (ri + rj)
        ki = model.agents[i].k
        kj = model.agents[j].k
        dr = y - x
        fij = 2 * (ki * kj) * (d2 - (ri + rj)^2) * (dr / d)
        forces[i] += fij
        forces[j] -= fij
    end
    return forces
end

# The function below, for the moment, is actually only computing the forces
# should it do something else?
function model_step!(model::ABM)
    # update cell lists
    model.clmap.cell_list = CellListMap.UpdateCellList!(
        model.positions, # current positions
        model.clmap.box,
        model.clmap.cell_list,
        model.clmap.aux;
        parallel=model.parallel
    )
    # reset forces at this step, and auxiliary threaded forces array
    fill!(model.forces, zeros(eltype(model.forces)))
    for i in eachindex(model.clmap.output_threaded)
        fill!(model.clmap.output_threaded[i], zeros(eltype(model.forces)))
    end
    # calculate pairwise forces at this step
    CellListMap.map_pairwise!(
        (x, y, i, j, d2, forces) -> calc_forces!(x, y, i, j, d2, forces, model),
        model.forces,
        model.clmap.box,
        model.clmap.cell_list;
        output_threaded=model.clmap.output_threaded,
        parallel=model.parallel
    )
    return
end

function simulate(; model=nothing, nsteps=1_000, n=10_000)
    isnothing(model) && (model = initialize_model(n=n))
    Agents.step!(
        model, agent_step!, model_step!, nsteps, false,
    )
end

end # module WithAgents



