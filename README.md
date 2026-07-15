# Reinforcement Learning Course

Doron Peleg | Afeka College of Engineering | 2026

This repository contains assignments and the final project for the Reinforcement Learning course.

## Repository Structure

```
RL/
├── assignment-1/           # Multi-Armed Bandit with Costs
├── assignment-2/           # RL Algorithm Comparison on GridWorld
├── final_project/          # The Messy-Network Trajectory Follower
│   ├── experiment_1d/      # 1D sine wave tracking
│   └── experiment_2d/      # 2D figure-eight tracking
├── gridworld.py            # Value iteration on a 5x5 grid
├── gridworld_solutions.py  # GridWorld solutions
├── dynamic_itterative.py   # Dynamic iterative policy evaluation
└── linear_itterative.py    # Linear iterative policy evaluation
```

## Assignments

### Assignment 1 — Epsilon-Greedy Bandit with Costs

A 5-arm bandit problem extended with per-arm pulling costs and non-stationary dynamics (drifting rewards and costs).

**Run:** `python assignment-1/epsilon_greedy_with_costs.py`

### Assignment 2 — RL Algorithm Comparison on Stochastic Treasure Hunt GridWorld

Compares Q-Learning, SARSA, and SARSA(lambda) on a custom stochastic GridWorld environment with treasure collection, traps, and walls.

**Run:** `python assignment-2/assignment2_024954471.py`

---

## Final Project — The Messy-Network Trajectory Follower

Investigates dropout-aware Q-learning for trajectory tracking under lossy network conditions, with Inverse RL for automatic reward recovery. Compares three agent strategies across 1D and 2D environments, showing a **15.5% improvement** for the dropout-aware approach in the complex 2D case.

Based on Fan et al. (2025), *"Inverse Reinforcement Learning for Discrete-Time Systems With Data Dropouts"*, IEEE Transactions on Cybernetics, Vol. 55, No. 4.

**See the full project report: [`final_project/README.md`](final_project/README.md)**
