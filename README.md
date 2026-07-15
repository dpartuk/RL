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

Based on Fan et al. (2025), *"Inverse Reinforcement Learning for Discrete-Time Systems With Data Dropouts"*, IEEE Transactions on Cybernetics, Vol. 55, No. 4.

### Motivation

In real networked control systems, sensor observations are transmitted over unreliable channels where packets can be randomly lost. When an RL agent learns through such a **lossy network**, it may act on stale observations without knowing it. This project investigates whether explicitly accounting for data dropouts during Q-learning improves trajectory tracking, and whether Inverse RL can recover the reward function from expert demonstrations.

### Approach

Three Q-learning agents are trained to follow a reference trajectory through a lossy channel (30% dropout probability):

| Agent | Training | Evaluation | Description |
|---|---|---|---|
| **Baseline** | No dropout | No dropout | Upper bound trained and tested on clean data |
| **Naive** | With dropout | With dropout | Standard Q-learning, unaware of packet loss |
| **Dropout-Aware** | With dropout | With dropout | Modified Q-update that weights by dropout probability |

The **Dropout-Aware** agent modifies the standard Q-learning update to incorporate the dropout probability directly. When a packet is dropped, it blends the update between the true (unobserved) state and the last known state, weighted by `(1 - dropout_prob)`.

An **Inverse RL** module then recovers the reward function from expert demonstrations without hand-coding it, using feature-based reward matching.

### Experiment 1: 1D Sine Wave Tracking

**Directory:** `final_project/experiment_1d/`

A discretized 1D environment where the agent follows a sinusoidal target signal.

| Parameter | Value |
|---|---|
| States | 41 position bins |
| Actions | 3 (left, stay, right) |
| Q-table | 123 entries |
| Target | Sine wave (amplitude 2.0, period 100) |
| Training | 500 episodes, 300 steps each |
| Dropout | 30% |

**Results:**

| Agent | Training Reward (last 20) | Eval Tracking Error |
|---|---|---|
| Baseline | -31.80 | 0.0858 |
| Naive | -45.94 | 0.1236 |
| Dropout-Aware | -43.24 | 0.1236 |

The 1D problem is too simple to differentiate Naive from Dropout-Aware at evaluation — both converge to the same policy. IRL successfully recovered the reward function (identical 3x2 reward comparison).

**Modules:**

| File | Role |
|---|---|
| `messy_network_trajectory_follower.py` | Main entry point |
| `config.py` | All hyperparameters |
| `environment.py` | `TrackingEnvironment` — 1D discretized tracking |
| `agents.py` | Baseline, Naive, and Dropout-Aware Q-learners |
| `dropout.py` | `DropoutChannel` — lossy network simulation |
| `evaluation.py` | Agent evaluation with/without dropout |
| `irl.py` | Feature extraction + Inverse RL loop |
| `state_estimator.py` | State estimator (ablation study) |
| `plotting.py` | All visualization functions |
| `../create_abstract.py` | Generates `abstract.docx` (in `final_project/`) |
| `../create_presentation.py` | Generates 25-slide `presentation.pptx` (in `final_project/`) |

**Run:**
```bash
cd final_project/experiment_1d
python messy_network_trajectory_follower.py
```

**Output plots:** `output_plots/` — training curves, tracking error, cumulative rewards, trajectories, IRL convergence, IRL reward comparison.

### Experiment 2: 2D Figure-Eight Tracking

**Directory:** `final_project/experiment_2d/`

A more complex 2D environment where the agent tracks a Lissajous figure-eight trajectory, designed to reveal the true advantage of dropout-aware learning.

| Parameter | Value |
|---|---|
| States | 441 (21x21 grid) |
| Actions | 9 (8 directions + stay) |
| Q-table | 3,969 entries |
| Target | Lissajous figure-eight (amplitude 2.5/1.8, periods 120/60) |
| Training | 800 episodes, 360 steps each |
| Dropout | 30% |

**Results:**

| Agent | Training Reward (last 20) | Eval Tracking Error |
|---|---|---|
| Baseline (no dropout) | -89.27 | 0.2373 |
| Naive | -164.55 | 0.4413 |
| Dropout-Aware | -199.37 | **0.3730** |

**Dropout-Aware beats Naive by 15.5%** in evaluation tracking error (0.3730 vs 0.4413). The increased state-action space of the 2D problem reveals the advantage that was hidden in the simpler 1D case.

**Modules:**

| File | Role |
|---|---|
| `main.py` | Main entry point |
| `config.py` | All hyperparameters |
| `environment.py` | `TrackingEnvironment2D` — 2D discretized tracking |
| `agents.py` | Baseline, Naive, and Dropout-Aware Q-learners |
| `dropout.py` | `DropoutChannel` — lossy network simulation |
| `evaluation.py` | Agent evaluation with/without dropout |
| `plotting.py` | All visualization functions |

**Run:**
```bash
cd final_project/experiment_2d
python main.py
```

**Output plots:** `output_plots/` — training curves, 2D trajectories, tracking error, evaluation comparison bar chart.

### Key Findings

1. **Complexity matters.** The 1D problem (41 states, 3 actions) is too simple — both agents converge to identical policies. The 2D problem (441 states, 9 actions) reveals a 15.5% advantage for the dropout-aware agent.
2. **Dropout-aware Q-updates help.** By incorporating `(1 - dropout_prob)` into the update rule, the agent learns a more robust policy under packet loss.
3. **IRL works.** Inverse RL successfully recovered the hand-crafted reward function from expert demonstrations in the 1D experiment, validating that reward learning is viable in this domain.

### Requirements

- Python 3.10+
- NumPy
- Matplotlib
- python-pptx (for presentation generation only)
- python-docx (for abstract generation only)
