using Agents
using StaticArrays
using InteractiveDynamics, GLMakie
import CellListMap

# Structure that contains the data to use cell list map (serial version)
mutable struct CellListMapData{B,C}
    box::B
    cell_list::C
end

@agent Particle ContinuousAgent{2} begin
    r::Float64 # radius
    k::Float64 # repulsion force constant 
    mass::Float64
end
Particle(id, pos, vel) = Particle(id, ntuple(i -> pos[i], 2), ntuple(i -> vel[i], 2), 10.0, 1.0, 1.0)

Base.@kwdef struct Properties{T,CL}
    dt::Float64 = 0.01
    n::Int64 = 0
    positions::Vector{SVector{2,T}}
    velocities::Vector{SVector{2,T}}
    forces::Vector{SVector{2,T}}
    cutoff::Float64
    cl_data::CL
end

function initialize_model(;
    n=1000,
    sides=SVector{2,Float64}(1000.0, 1000.0),
    dt=0.01
)
    # initial positions and velocities
    positions = [sides .* rand(SVector{2,Float64}) for _ in 1:n]
    velocities = [-50 .+ 100 .* rand(SVector{2,Float64}) for _ in 1:n]

    # Space and agents
    space2d = ContinuousSpace(ntuple(i -> sides[i], 2); periodic=true)
    particles = [Particle(id, positions[id], velocities[id]) for id in 1:n]

    # initialize array of forces
    forces = zeros(SVector{2,Float64}, n)

    # cutoff is twice the maximum radius among particles
    cutoff = maximum(2 * p.r for p in particles)

    # Define cell list structure 
    box = CellListMap.Box(sides, cutoff)
    cl = CellListMap.CellList(positions, box; parallel=false)
    cl_data = CellListMapData(box, cl)

    # define the model
    properties = Properties(
        dt=dt,
        n=n,
        cutoff=cutoff,
        positions=positions,
        velocities=velocities,
        forces=forces,
        cl_data=cl_data
    )
    model = ABM(Particle,
        space2d,
        properties=properties
    )

    # create active rods
    for id in 1:n
        add_agent!(particles[id], model)
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
    agent.pos = ntuple(i -> x_new[i], 2)
    agent.vel = ntuple(i -> v_new[i], 2)
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
        fij = 2 * (ki * kj) * (d2 - (ri + rj)^2) * (dr/d)
        forces[i] += fij
        forces[j] -= fij
    end
    return forces
end

# The function below, for the moment, is actually only computing the forces
# should it do something else?
function model_step!(model::ABM)
    # update cell lists
    model.cl_data.cell_list = CellListMap.UpdateCellList!(
        model.positions, # current positions
        model.cl_data.box,
        model.cl_data.cell_list;
        parallel=false
    )
    # reset forces at this step, and auxiliary threaded forces array
    for i in eachindex(model.forces)
        model.forces[i] = zero(eltype(model.forces))
    end
    # calculate pairwise forces at this step
    CellListMap.map_pairwise!(
        (x, y, i, j, d2, forces) -> calc_forces!(x, y, i, j, d2, forces, model),
        model.forces,
        model.cl_data.box,
        model.cl_data.cell_list;
        parallel=false
    )
end

function simulate(; nsteps=1000)
    model = initialize_model()
    run!(
        model, agent_step!, model_step!, nsteps; agents_first=false,
        showprogress=true
    )
end

function video(; nsteps=1000)
    model = initialize_model()
    abmvideo(
        "test.mp4", model, agent_step!, model_step!;
        framerate=50, spf=1, frames=nsteps, agents_first=false,
        title="Particles"
    )
end



