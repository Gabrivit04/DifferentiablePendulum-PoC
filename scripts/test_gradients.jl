# Script to demonstrate Automatic Differentiation through the environment
include("../src/PendulumEnv.jl")
using .PendulumEnv
using Zygote
using StaticArrays

println("--- Differentiable Pendulum PoC ---")

# Initial State: [theta, theta_dot]
state = SVector(0.5, 0.0) 
action = 1.0

# 1. Forward Pass
next_state, reward = step_env(state, action)
println("Current State: ", state)
println("Action Applied: ", action)
println("Next State:    ", next_state)
println("Reward:        ", reward)
println("-----------------------------------")

# 2. Backward Pass (The Magic)
# We want to find the exact gradient of the reward with respect to the action: ∂r/∂a
# This tells the RL agent exactly how to change its action to maximize reward.

gradient_wrt_action = Zygote.gradient(a -> step_env(state, a)[2], action)[1]

println("Exact Analytic Gradient (∂r/∂a): ", gradient_wrt_action)
println("Success! Zygote successfully backpropagated through the physics ODEs.")
