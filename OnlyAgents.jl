module OnlyAgents

using Agents

@agent Particle ContinuousAgent{2} begin
    r::Float64 # radius
    k::Float64 # repulsion force constant
    mass::Float64
end

Base.@kwdef struct Properties{T}
    dt::Float64 = 0.01
    n::Int64 = 0
    forces::Vector{NTuple{2,T}}
    cutoff::Float64
end
Particle(; id, pos, vel, r, k, mass) = Particle(id, pos, vel, r, k, mass)

function initialize_model(;
    n=10_000,
    sides=(1000.0, 1000.0),
    dt=0.01
)

    # maximum radius is 10.0, so cutoff is 20.0
    cutoff = 20.0

    # Space
    space2d = ContinuousSpace(sides; periodic=true)

    # initialize vector of forces
    forces = [(0.0, 0.0) for _ in 1:n]

    # define the model
    properties = Properties(; dt, n, forces, cutoff)
    model = ABM(Particle, space2d; properties)

    # create active agents
    for id in 1:n
        add_agent_pos!(
            Particle(
                id=id,
                r=1.0 + 9 * rand(),
                k=1.0 + rand(),
                mass=10.0 + 10 * rand(),
                pos=(sides[1] * rand(), sides[2] * rand()),
                vel=(-50 + 100 * rand(), -50 + 100 * rand()),
            ),
            model)
    end

    return model
end

# Here the agent positions and velocities get updated with simple Euler integration
# steps. Seems that this could/should be easy to be substituted by some better integration
# scheme of DifferentialEquations.
function agent_step!(agent, model::ABM)
    id = agent.id
    x = agent.pos
    v = agent.vel
    f = model.forces[id]
    dt = model.dt
    a = f ./ agent.mass
    x_new = @. x + v * dt + (a / 2) * dt^2
    v_new = @. v + f * dt
    agent.vel = v_new
    x_new = normalize_position(x_new, model)
    move_agent!(agent, x_new, model)
    return
end

# This function udpates the forces array for each interacting pair
# The potential is a smooth short-ranged interaction:
#
# U(r) = (ki*kj)*(r^2 - (ri+rj)^2)^2 for r ≤ (ri+rj)
# U(r) = 0.0 for r > (ri+rj)
#
# where ri and rj are the radii of the two agents, and ki and kj are the
# potential energy constants.
function calc_forces!(forces, model, pairs)
    for (i, j) in pairs
        ai::Particle = model[i]
        aj::Particle = model[j]
        ri = ai.r
        rj = aj.r
        x = ai.pos
        y = aj.pos
        d2 = (x[1] - y[1])^2 + (x[2] - y[2])^2
        d = sqrt(d2)
        if d ≤ (ri + rj)
            ki = ai.k
            kj = aj.k
            dr = @. y - x
            fij = @. 2 * (ki * kj) * (d2 - (ri + rj)^2) * (dr / d)
            forces[i] = forces[i] .+ fij
            forces[j] = forces[j] .- fij
        end
    end
    return
end


# The function below, for the moment, is actually only computing the forces
# should it do something else?
function model_step!(model::ABM)
    fill!(model.forces, (0.0, 0.0)) # reset forces
    p_iter = interacting_pairs(model, model.cutoff, :all)
    calc_forces!(model.forces, model, p_iter.pairs)
    return
end

function simulate(; model=nothing, nsteps=1_000)
    isnothing(model) && (model = initialize_model())
    Agents.step!(
        model, agent_step!, model_step!, nsteps, false,
    )
end

end # module



