# 🚀 Differentiable Pendulum PoC
**A Proof of Concept for GSoC 2026: Pure-Julia Reinforcement Learning Environments**

Traditional RL environments (like OpenAI Gym) rely on C++ physics engines wrapped in Python. This creates a "black box" where RL algorithms cannot access the analytic gradients of the transition dynamics. 

This repository demonstrates a **pure-Julia, allocation-free** implementation of the classic Pendulum physics using `StaticArrays.jl`. Because it avoids array mutations, the physics engine is 100% compatible with `Zygote.jl` for Reverse-Mode Automatic Differentiation.

## 🧠 Why this matters
By passing the environment's `step` function through Zygote, we can compute exact Jacobian-vector products. For example, we can instantly calculate the gradient of the reward with respect to the action applied ($\frac{\partial r}{\partial a}$). This allows for analytic policy optimization, bypassing sample-inefficient algorithms like PPO or REINFORCE.

## ⚙️ How to run the PoC
You can verify the automatic differentiation yourself.

1. Clone the repository and instantiate the environment:
```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
