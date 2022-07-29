using Agents
using StaticArrays
import CellListMap

"""

The simulation of the example will be of point particles in a continuous 2D space.
The particles will repel each other when their distances are smaller than the sum
of their radii, and will not interact if the distance is greater than that.

Thus, if the maximum sum of radii between particles is much smaller than the size 
of the system, cell lists can greatly accelerate the computation of the pairwise forces.

Each particle has different radii and different repulsion force constants and masses.

By default, the `Particle` type, for the `ContinousAgent{2}` space, will have additionally
an `id` and `pos` (positon) and `vel` (velocity) fields, which are automatically added
by the `@agent` macro. 

"""
@agent Particle ContinuousAgent{2} begin
    r::Float64 # radius
    k::Float64 # repulsion force constant
    mass::Float64
end
Particle(; id, pos, vel, r, k, mass) = Particle(id, pos, vel, r, k, mass)

"""

The `CellListMap` data structure contains the necessary data for the fast 
computation of interactions within a cutoff. The structure must be mutable,
because it is expected that the `box` and `cell_list` fields, which will contain
immutable data structures, will be updated at each simulation step.

Four data structures are necessary:

`box` is the `CellListMap.Box` data structure containing the size of the system
(generally with periodicity), and the cutoff that is used for pairwise interactions.

`cell_list` will contain the cell lists obtained with the `CellListMap.CellList` 
constructor. 

The next two auxiliary structures necessary for parallel runs, but for simplicity they will always be
defined:

`aux` is a data structure that is built with the `CellListMap.AuxThreaded` constructor,
and contains auxiliary arrays to paralellize the construction of the cell lists.

`output_threaded` is a vector containing copies of the output of the mapped function,
for parallelization.

"""
# Structure that contains the data to use CellListMap
mutable struct CellListMapData{B,C,A,O}
    box::B
    cell_list::C
    aux::A
    output_threaded::O
end

"""

The model properties.

For effective use of `CellListMap`, the positions have to be stored in a vector of,
preferentially, static vectors. Thus, this duplicates the position data (which
is stored in the `Particles.pos` field), but is important for the integration. 

The `forces` between particles are stored in a `Vector{SVector{2}}`, and are updated
at each simulation step by the `CellListMap.map_pairwise!` function.

The `cutoff` is the maximum possible distance between particles with non-null interactions,
meaning here twice the maximum radius that the particles may have.

"""
Base.@kwdef struct Properties{T<:Real,CL<:CellListMapData}
    dt::Float64 = 0.01
    number_of_particles::Int64 = 0
    positions::Vector{SVector{2,T}}
    forces::Vector{SVector{2,T}}
    cutoff::Float64
    clmap::CL # CellListMap data
    parallel::Bool
end

"""

In the following example, we run by default with 10_000 particles in a 2D system
with sides 1000.0. The maximum possible radius of the particles is set to 10.0, and
each particle will have a random radius assigned.

"""
function initialize_model(;
    number_of_particles=10_000,
    sides=SVector{2,Float64}(1000.0, 1000.0),
    dt=0.01,
    max_radius=10.0,
    parallel=true,
)
    # initial positions
    positions = [sides .* rand(SVector{2,Float64}) for _ in 1:number_of_particles]

    # Space and agents
    space2d = ContinuousSpace(Tuple(sides); periodic=true)

    # initialize array of forces
    forces = zeros(SVector{2,Float64}, number_of_particles)

    # maximum radius is 10.0 thus cutoff is 20.0
    cutoff = 2*max_radius

    # Define cell list structure
    box = CellListMap.Box(sides, cutoff)
    cl = CellListMap.CellList(positions, box; parallel=parallel)
    aux = CellListMap.AuxThreaded(cl)
    output_threaded = [copy(forces) for _ in 1:CellListMap.nbatches(cl)]
    clmap = CellListMapData(box, cl, aux, output_threaded)

    # define the model
    properties = Properties(
        dt=dt,
        number_of_particles=number_of_particles,
        cutoff=cutoff,
        positions=positions,
        forces=forces,
        clmap=clmap,
        parallel=parallel,
    )
    model = ABM(Particle,
        space2d,
        properties=properties
    )

    # Create active agents
    for id in 1:number_of_particles
        add_agent_pos!(
            Particle(
                id=id,
                r=1.0 + (max_radius - 1.0) * rand(),
                k=1.0 + rand(),
                mass=10.0 + 10 * rand(),
                pos=Tuple(positions[id]),
                vel=(-50 + 100*rand(), -50 + 100*rand()),
            ),
        model)
    end

    return model
end

"""

The `agent_step!` function will update the particle positons and velocities,
given the forces, which are computed in the `model_step!` function. A simple
Euler step is used here for simplicity. 

"""
function agent_step!(agent, model::ABM)
    id = agent.id
    f = model.forces[id]
    x = SVector{2,Float64}(agent.pos)
    v = SVector{2,Float64}(agent.vel)
    dt = model.properties.dt
    a = f / agent.mass
    x_new = x + v * dt + (a / 2) * dt^2
    v_new = v + f * dt
    model.positions[id] = x_new
    agent.pos = Tuple(x_new)
    agent.vel = Tuple(v_new)
    return nothing
end

"""

This function udpates the model.forces array for each interacting pair
The potential is a smooth short-ranged interaction:

U(r) = (ki*kj)*(r^2 - (ri+rj)^2)^2 for r ≤ (ri+rj)
U(r) = 0.0 for r > (ri+rj)

where ri and rj are the radii of the two agents, and ki and kj are the
potential energy constants.

The function updates the forces for *one* pair of particles, and will be 
implicitly called by `CellListMap.map_pairwise!` for the pairs of particles
that are found within the cutoff distance. The input and output parameters
of this functions are defined by the `CellListMap` API.

"""
function calc_forces!(x, y, i, j, d2, forces, model)
    pᵢ = model.agents[i]
    pⱼ = model.agents[j]
    d = sqrt(d2)
    if d ≤ (pᵢ.r + pⱼ.r)
        dr = y - x
        fij = 2 * (pᵢ.k * pⱼ.k) * (d2 - (pᵢ.r + pⱼ.r)^2) * (dr / d)
        forces[i] += fij
        forces[j] -= fij
    end
    return forces
end

"""

This function updates the forces at each step of the simulation.

"""
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
    return nothing
end

"""

The simulation function.

"""
function simulate(; model=nothing, nsteps=1_000, number_of_particles=10_000)
    isnothing(model) && (model = initialize_model(number_of_particles=number_of_particles))
    Agents.step!(
        model, agent_step!, model_step!, nsteps, false,
    )
end