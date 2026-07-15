# The Messy-Network Trajectory Follower

Doron Peleg | Afeka College of Engineering | RL Course Final Project | 2026

## Overview

In networked control systems, sensor data travels over unreliable channels where packets can be randomly lost. An RL agent learning through such a **lossy network** may unknowingly act on stale observations, degrading its policy. This project investigates whether explicitly accounting for data dropouts during Q-learning improves trajectory tracking performance.

Based on Fan et al. (2025), *"Inverse Reinforcement Learning for Discrete-Time Systems With Data Dropouts"*, IEEE Transactions on Cybernetics, Vol. 55, No. 4.

## Project Structure

```
final_project/
├── experiment_1d/                  # 1D sine wave tracking experiment
│   ├── messy_network_trajectory_follower.py   # Main entry point
│   ├── config.py                   # Hyperparameters
│   ├── environment.py              # 1D discretized tracking environment
│   ├── agents.py                   # Baseline, Naive, Dropout-Aware agents
│   ├── dropout.py                  # Lossy network channel simulation
│   ├── evaluation.py               # Evaluation with/without dropout
│   ├── irl.py                      # Inverse RL: feature extraction + reward recovery
│   ├── state_estimator.py          # State estimator (ablation study)
│   ├── plotting.py                 # Visualization functions
│   └── output_plots/               # Generated figures (7 plots)
├── experiment_2d/                  # 2D figure-eight tracking experiment
│   ├── main.py                     # Main entry point
│   ├── config.py                   # Hyperparameters
│   ├── environment.py              # 2D discretized tracking environment
│   ├── agents.py                   # Baseline, Naive, Dropout-Aware agents
│   ├── dropout.py                  # Lossy network channel simulation
│   ├── evaluation.py               # Evaluation with/without dropout
│   ├── plotting.py                 # Visualization functions
│   └── output_plots/               # Generated figures (4 plots)
├── create_presentation.py          # Generates presentation.pptx (python-pptx)
├── create_abstract.py              # Generates abstract.docx (python-docx)
├── presentation.pptx               # 25-slide TED-style presentation
└── messy_network_trajectory_follower_plan.txt
```

## Method

### Dropout Channel

A `DropoutChannel` simulates a lossy network with a configurable dropout probability (default 30%). When a packet is dropped, the channel returns the last successfully received state instead of the true current state. The agent has no way to distinguish a fresh observation from a stale one.

### Three Agents

| Agent | How it trains | How it evaluates | Q-update rule |
|---|---|---|---|
| **Baseline** | Clean data (no dropout) | Clean data (no dropout) | Standard Q-learning |
| **Naive** | Through dropout channel | Through dropout channel | Standard Q-learning (ignores dropout) |
| **Dropout-Aware** | Through dropout channel | Through dropout channel | Weighted by `(1 - dropout_prob)` |

The **Dropout-Aware** agent modifies the Q-learning update to incorporate the dropout probability directly. When a packet may be dropped, the update blends between the observed state and the expected stale state, weighted by the known dropout probability. This produces a more robust policy without requiring packet-level drop detection.

### Inverse RL (1D only)

An IRL module recovers the reward function from expert demonstrations using feature-based reward matching. A 5-dimensional feature vector captures proximity to the target (bullseye, linear penalty, quadratic penalty, danger zone, bias). The recovered reward matches the hand-crafted reward exactly, validating that reward learning is viable in this domain.

## Experiment 1: 1D Sine Wave Tracking

The agent follows a sinusoidal target signal in a discretized 1D position space.

| Parameter | Value |
|---|---|
| States | 41 position bins |
| Actions | 3 (left, stay, right) |
| Q-table entries | 123 |
| Target trajectory | Sine wave (amplitude 2.0, period 100) |
| Training episodes | 500 |
| Steps per episode | 300 |
| Dropout probability | 30% |

### 1D Results

| Agent | Training Reward (last 20 avg) | Eval Tracking Error |
|---|---|---|
| Baseline | -31.80 | 0.0858 |
| Naive | -45.94 | 0.1236 |
| Dropout-Aware | -43.24 | 0.1236 |

**Finding:** The 1D problem is too simple — both Naive and Dropout-Aware converge to the same evaluation policy. The small state-action space (123 Q-entries) doesn't provide enough room for the dropout-aware update to yield a different policy.

### Ablation (1D)

| Variant | Training Reward (last 20) |
|---|---|
| Baseline (no dropout) | -31.80 |
| Dropout-Aware | -43.24 |
| Naive | -45.94 |
| Confidence-weighted alpha | -45.66 |
| State estimator | -68.10 |

The state estimator and confidence-weighted variants performed worse, confirming that the simple dropout-probability weighting is the most effective approach.

### IRL Results (1D)

IRL perfectly recovered the hand-crafted reward function — the 3x2 reward comparison between original and recovered rewards is identical.

### Run

```bash
cd experiment_1d
python messy_network_trajectory_follower.py
```

Generates 7 plots in `experiment_1d/output_plots/`.

## Experiment 2: 2D Figure-Eight Tracking

A more complex environment where the agent tracks a Lissajous figure-eight trajectory on a 2D grid, designed to stress-test the agents with a larger state-action space.

| Parameter | Value |
|---|---|
| States | 441 (21 x 21 grid) |
| Actions | 9 (8 compass directions + stay) |
| Q-table entries | 3,969 |
| Target trajectory | Lissajous figure-eight (amplitude 2.5/1.8, periods 120/60) |
| Training episodes | 800 |
| Steps per episode | 360 (3 full figure-eight cycles) |
| Dropout probability | 30% |

### 2D Results

| Agent | Training Reward (last 20 avg) | Eval Tracking Error |
|---|---|---|
| Baseline (no dropout at eval) | -89.27 | 0.2373 |
| Naive | -164.55 | 0.4413 |
| Dropout-Aware | -199.37 | **0.3730** |

**Dropout-Aware beats Naive by 15.5%** in evaluation tracking error (0.3730 vs 0.4413). The 32x larger state-action space reveals the advantage that was hidden in the 1D case.

Note: The Baseline is evaluated without dropout (it trained on clean data), serving as an upper bound on performance.

### Run

```bash
cd experiment_2d
python main.py
```

Generates 4 plots in `experiment_2d/output_plots/`.

## Key Findings

1. **Complexity reveals the advantage.** The 1D problem (123 Q-entries) is too simple for dropout-aware updates to matter. The 2D problem (3,969 Q-entries) shows a clear 15.5% improvement.
2. **Simple weighting works best.** Incorporating `(1 - dropout_prob)` into the Q-update outperforms more complex approaches (state estimators, confidence-weighted learning rates).
3. **IRL recovers reward from demonstrations.** Feature-based inverse RL successfully learned the exact reward function from expert trajectories, showing that hand-crafted rewards are not strictly necessary.

## Presentation

A 25-slide TED-style presentation (dark theme) covering both experiments:

- Slides 1-17: 1D experiment + IRL (problem setup, dropout mechanism, agent comparison, ablation study, IRL concept and results)
- Slide 18: Transition — "But was our problem too easy?"
- Slides 19-22: 2D experiment (setup, trajectories, results, 1D vs 2D comparison)
- Slides 23-25: Takeaways, closing, Q&A

To regenerate:
```bash
python create_presentation.py   # produces presentation.pptx
python create_abstract.py       # produces abstract.docx
```

## Requirements

- Python 3.10+
- NumPy
- Matplotlib
- python-pptx (presentation generation)
- python-docx (abstract generation)
