module PendulumEnv

using StaticArrays

export step_env

# Constants for the Pendulum physics
const G = 9.81
const M = 1.0
const L = 1.0
const DT = 0.05
const MAX_SPEED = 8.0
const MAX_TORQUE = 2.0

"""
    step_env(state::SVector{2, Float64}, action::Float64)

Advances the pendulum environment by one timestep.
Written strictly without array mutation to ensure 100% compatibility with Zygote.jl.
Returns a tuple: (new_state, reward)
"""
function step_env(state::SVector{2, Float64}, action::Float64)
    th, thdot = state
    
    # Clip action torque
    u = clamp(action, -MAX_TORQUE, MAX_TORQUE)
    
    # Calculate angular acceleration (ODE)
    newthdot = thdot + (-3 * G / (2 * L) * sin(th + pi) + 3 / (M * L^2) * u) * DT
    newthdot = clamp(newthdot, -MAX_SPEED, MAX_SPEED)
    
    # Calculate new angle
    newth = th + newthdot * DT
    
    # Normalize angle between -pi and pi
    newth_norm = mod(newth + pi, 2*pi) - pi
    
    # Reward function: penalize being away from top (th=0), high speeds, and high effort
    reward = -(newth_norm^2 + 0.1 * thdot^2 + 0.001 * u^2)
    
    return SVector{2, Float64}(newth_norm, newthdot), reward
end

end # module
