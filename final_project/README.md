# The Messy-Network Trajectory Follower

**Doron Peleg** | Afeka College of Engineering, Tel Aviv | Reinforcement Learning Course | 2026

---

## Table of Contents

1. [Introduction](#introduction)
2. [Background and Motivation](#background-and-motivation)
3. [Problem Formulation](#problem-formulation)
4. [Method](#method)
   - [The Dropout Channel](#the-dropout-channel)
   - [Agent Architectures](#agent-architectures)
   - [Q-Learning Update Rules](#q-learning-update-rules)
   - [Inverse Reinforcement Learning](#inverse-reinforcement-learning)
5. [Experiment 1: 1D Sine Wave Tracking](#experiment-1-1d-sine-wave-tracking)
6. [Experiment 2: 2D Figure-Eight Tracking](#experiment-2-2d-figure-eight-tracking)
7. [Results and Analysis](#results-and-analysis)
8. [Conclusions](#conclusions)
9. [Project Structure](#project-structure)
10. [How to Run](#how-to-run)
11. [References](#references)

---

## Introduction

Standard reinforcement learning algorithms assume that the agent receives reliable state observations at every time step. In real-world networked control systems, however, sensor readings and control signals are transmitted over wireless channels where packets are randomly lost. When a packet is dropped, the agent unknowingly acts on stale information, corrupting both its decisions and its learning updates.

This project presents a simplified, educational simulation that investigates whether explicitly accounting for data dropouts during Q-learning improves trajectory tracking performance, and whether Inverse Reinforcement Learning can recover the reward function solely from expert demonstrations.

Consider a concrete scenario: a surgical robot operating remotely over Wi-Fi. The connection drops for half a second. The robot's last known position of the scalpel is 2mm from where it actually is. Does it cut? Does it freeze? Does it guess? This is the data dropout problem in networked control systems.

## Background and Motivation

This project is inspired by Fan et al. (2025), *"Inverse Reinforcement Learning for Discrete-Time Systems With Data Dropouts"*, published in IEEE Transactions on Cybernetics (Vol. 55, No. 4, pp. 1744-1757). The original paper proposes both a model-based IRL algorithm using a Smith predictor for state estimation and a data-driven, model-free inverse Q-learning algorithm that explicitly accounts for random state dropouts. Their work provides rigorous convergence proofs and demonstrates that learning both the cost function and the control policy from incomplete data outperforms methods that rely on manually specified cost functions.

Our project distills the core insights of this paper into a tabular Q-learning setting that makes the effects of dropout visible and interpretable, without the complexity of continuous-state function approximation.

## Problem Formulation

Every RL algorithm assumes the agent can observe the current state of the environment. But what happens when it cannot?

When a state observation is lost, the agent faces three possible reactions:

1. **Pretend nothing happened** — use the stale observation as if it were current (the Naive approach)
2. **Try to predict what was missed** — extrapolate from the last known state and action (state estimator approach)
3. **Admit you don't know** — skip learning when the data is unreliable (the Dropout-Aware approach)

We implement and compare all three strategies, along with additional variants, in two environments of increasing complexity.

### Environment Design

**1D Environment (Experiment 1):**
- A 1D number line from -4 to +4
- Target moves in a sine wave: `x(t) = 2.0 * sin(2*pi*t / 100)`
- 3 actions: LEFT (-0.2), STAY (0), RIGHT (+0.2)
- 41 discrete states (bins of relative distance to target)
- Bin 20 = exactly on target
- Reward = `-|distance|` (on target = 0, 4 units away = -4)
- Each episode = 300 steps (3 full sine cycles)

**2D Environment (Experiment 2):**
- A 2D grid from -4 to +4 on both axes
- Target traces a Lissajous figure-eight: `x(t) = 2.5*sin(2*pi*t/120)`, `y(t) = 1.8*sin(2*pi*t/60)`
- The 2:1 period ratio creates the figure-eight pattern
- 9 actions: 8 compass directions + stay, step size 0.3
- 441 discrete states (21x21 grid)
- Each episode = 360 steps (3 full figure-eight cycles)

## Method

### The Dropout Channel

The `DropoutChannel` class simulates a lossy network with a 30% dropout probability. At each time step, there is a 30% chance the state observation is lost. When a packet is dropped, the channel returns the **last successfully received state** instead of the true current state. The agent has no way to distinguish a fresh observation from a stale one.

This is analogous to a GPS that freezes 30% of the time — showing where you *were* instead of where you *are*.

**Example dropout sequence:**

| Step | True State | Observed State | Status |
|------|-----------|---------------|--------|
| 1 | Bin 18 | Bin 18 | OK |
| 2 | Bin 19 | Bin 19 | OK |
| 3 | Bin 20 | Bin 19 | DROPPED |
| 4 | Bin 21 | Bin 21 | OK |
| 5 | Bin 22 | Bin 22 | OK |
| 6 | Bin 21 | Bin 22 | DROPPED |

In steps 3 and 6, the agent thinks it is in one bin but is actually in another. It makes decisions and Q-table updates based on wrong information.

### Agent Architectures

Three Q-learning agents share the same tabular Q-learning algorithm. The only difference is how they handle dropout:

**1. Baseline Agent** (upper bound)
- No dropout at all — sees true state every step
- Always updates Q-table with correct state information
- Represents what is possible with perfect information

**2. Naive Agent**
- Trains through the dropout channel (30% packet loss)
- Receives stale states on dropout but treats them as current
- Always updates Q-table, even with stale observations
- **Problem:** writes updates to wrong Q-table entries, corrupting its knowledge

**3. Dropout-Aware Agent**
- Trains through the dropout channel (30% packet loss)
- Tracks whether its current state came from a fresh observation
- **Skips Q-update if the current state is stale**
- Protects Q-table from corruption — learns less often, but learns correctly

### Q-Learning Update Rules

All agents use the standard tabular Q-learning update:

```
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

With hyperparameters: `alpha = 0.1`, `gamma = 0.95`, epsilon-greedy exploration decaying from 1.0 to 0.01 (decay rate 0.995).

The critical difference is in **when** the update is applied:

- **Baseline:** Updates at every step (state is always correct)
- **Naive:** Updates at every step (even when state is stale — writes to wrong Q-table row)
- **Dropout-Aware:** Only updates when `current_is_reliable = True`. When a dropout occurred in the previous step, the current observed state may be stale, so the update is skipped entirely.

This means approximately 28% of steps are skipped (~83 out of 300 per episode). But the remaining 72% write accurate values to the correct Q-table entries. A smaller set of clean updates beats a larger set of noisy ones.

### Inverse Reinforcement Learning

Standard RL requires a hand-crafted reward function (in our case, `reward = -|distance|`). But how do we know that reward is correct? What if we could discover it automatically — just by watching an expert perform the task?

**Regular RL:** Given a reward function, find the optimal policy.
**Inverse RL:** Given expert demonstrations, find the reward function.

The analogy: you watch a top student ace every exam. You don't copy their answers — you figure out what textbook they are studying from.

#### Feature Representation

We define a 5-dimensional feature vector that captures hypotheses about what the expert cares about:

| Feature | Formula | Question |
|---------|---------|----------|
| f0: Bullseye | 1.0 if within 0.1 units of target | Does the expert care about precision? |
| f1: Linear | -\|distance\| | Is the penalty proportional to distance? |
| f2: Quadratic | -distance^2 | Is the penalty proportional to distance squared? |
| f3: Danger | -1.0 if beyond 75% of range | Does the expert avoid the edges? |
| f4: Bias | constant 1.0 | Is there a constant cost per step? |

The reward is expressed as: `R(s) = w . f(s)` — a linear combination of features weighted by learnable weights.

#### The IRL Loop (200 iterations)

1. **Train** a Q-learner for 100 episodes using `reward = w . features`
2. **Evaluate** the learned policy for 10 episodes, measuring its average feature vector
3. **Compute gradient:** `gradient = expert_features - learner_features` ("Where is the learner falling short?")
4. **Update weights:** `w += 0.01 * gradient` ("Push reward toward expert behavior")

Repeat until the learner's behavior matches the expert's — at that point, the reward function has been found.

#### Expert Demonstrations

The expert is an oracle that always picks the action minimizing distance to the target (a simple greedy lookahead over the 3 possible actions). 20 expert demonstrations are generated, and the average feature expectations across all steps serve as the IRL target.

#### Learned Reward Weights

After 200 iterations, IRL converged to:

| Feature | Learned Weight | Interpretation |
|---------|---------------|----------------|
| Quadratic | 10.000 (clipped max) | Being far is very costly — dominant factor |
| Bullseye | 0.806 | Precision near target matters |
| Linear | 0.378 | Moderate linear distance penalty |
| Danger | 0.010 | Edge avoidance barely matters |
| Bias | 0.000 | No constant per-step cost |

The quadratic penalty dominates — confirming that the expert cares most about avoiding large deviations from the target.

## Experiment 1: 1D Sine Wave Tracking

### Setup

| Parameter | Value |
|---|---|
| States | 41 position bins |
| Actions | 3 (left, stay, right) |
| Q-table entries | 123 |
| Target trajectory | Sine wave (amplitude 2.0, period 100) |
| Training episodes | 500 |
| Steps per episode | 300 (3 sine cycles) |
| Dropout probability | 30% |
| Learning rate (alpha) | 0.1 |
| Discount factor (gamma) | 0.95 |
| Epsilon decay | 1.0 -> 0.01 (rate 0.995) |

### Training Results

| Agent | Training Reward (last 20 avg) |
|---|---|
| Baseline | -31.80 |
| Dropout-Aware | -43.24 |
| Naive | -45.94 |

The Dropout-Aware agent achieves a ~6% improvement over Naive during training by skipping stale updates.

### Evaluation Results

At evaluation time (greedy policy, no learning, no exploration):

| Agent | Mean Tracking Error | Total Reward |
|---|---|---|
| Baseline | 0.0858 | -25.75 |
| Naive | 0.1236 | -37.09 |
| Dropout-Aware | 0.1236 | -37.09 |

**Surprise:** Naive and Dropout-Aware produce **identical** evaluation scores. With only 41 states and 3 actions, both Q-tables converge to the same argmax policy. The dropout-aware advantage is in learning efficiency, not the final policy. The problem is too simple to separate them.

### Ablation Study

We tested every combination of dropout-handling strategies:

| Strategy | Training Reward (last 20) | Verdict |
|---|---|---|
| Baseline (no dropout) | -31.80 | Reference |
| Skip when stale (Dropout-Aware) | -43.24 | Best under dropout |
| Naive (always update) | -45.94 | Neutral |
| Confidence-weighted alpha | -45.66 | No improvement |
| State estimator only | -68.10 | Much worse |
| State estimator + confidence | -68.02 | Much worse |

**The fancy solution made it worse.** The state estimator — which tries to predict the missed state — performed dramatically worse (-68.10 vs. -45.94 for doing nothing). The best strategy was the simplest: when you don't know, don't learn.

### IRL Validation

A 3x2 comparison (3 agent types x 2 reward sources) confirms that IRL recovered the reward perfectly:

| Agent | Hand-coded Eval Error | IRL-learned Eval Error |
|---|---|---|
| Baseline | 0.0858 | 0.0858 |
| Naive | 0.1236 | 0.1236 |
| Dropout-Aware | 0.1236 | 0.1236 |

Every cell matches. The IRL-learned reward generalizes across all agent types, preserving the same performance ordering observed under the hand-coded reward.

### Output Plots (1D)

| Plot | Description |
|---|---|
| `1_training_curves.png` | Training reward over 500 episodes for all 3 agents |
| `2_tracking_error.png` | Mean absolute tracking error over training |
| `3_cumulative_rewards.png` | Cumulative reward within episodes |
| `4_trajectories.png` | Agent trajectories vs. sinusoidal target + dropout mask |
| `5_irl_convergence.png` | IRL weight convergence over 200 iterations |
| `6_irl_comparison.png` | Hand-coded vs. IRL-learned reward comparison |
| `irl_diagram.png` | IRL loop diagram |

## Experiment 2: 2D Figure-Eight Tracking

### Motivation

The 1D experiment showed that Naive and Dropout-Aware converge to the same policy — the problem was too simple. With 41 states and 3 actions, the Q-table has only 123 entries, making it nearly impossible for stale updates to flip the best action permanently.

The 2D experiment was designed to answer: **what happens when we scale up?**

### Why 2D Changes Everything

| Factor | 1D | 2D | Impact |
|---|---|---|---|
| Q-table size | 123 entries | 3,969 entries (32x) | Many states visited rarely — corrupted updates can flip the best action |
| Actions | 3 | 9 (3x) | Choosing between 8 directions is ambiguous — noise promotes wrong ones |
| Stale error | Wrong in 1 dimension | Wrong in X and Y simultaneously | Bigger mistakes, harder to recover |

### Setup

| Parameter | Value |
|---|---|
| States | 441 (21 x 21 grid) |
| Actions | 9 (8 compass directions + stay) |
| Q-table entries | 3,969 |
| Target trajectory | Lissajous figure-eight (x: amplitude 2.5, period 120; y: amplitude 1.8, period 60) |
| Training episodes | 800 |
| Steps per episode | 360 (3 full figure-eight cycles) |
| Dropout probability | 30% |
| Learning rate (alpha) | 0.1 |
| Discount factor (gamma) | 0.95 |
| Epsilon decay | 1.0 -> 0.01 (rate 0.995) |

### Training Results

| Agent | Training Reward (last 20 avg) |
|---|---|
| Baseline | -89.27 |
| Naive | -164.55 |
| Dropout-Aware | -199.37 |

### Evaluation Results

| Agent | Mean Tracking Error | Total Reward |
|---|---|---|
| Baseline (no dropout at eval) | 0.2373 | -85.44 |
| Naive (30% dropout) | 0.4413 | -158.87 |
| Dropout-Aware (30% dropout) | **0.3730** | **-134.29** |

**Dropout-Aware beats Naive by 15.5%** in evaluation tracking error (0.3730 vs. 0.4413). Unlike the 1D case, the policies are now **different**. Corrupted Q-table entries in rarely-visited states produce wrong actions. The Dropout-Aware agent, by protecting its Q-table from stale updates, arrives at a genuinely better policy.

The Baseline is evaluated without dropout (it trained on clean data), serving as an upper bound on performance.

### Output Plots (2D)

| Plot | Description |
|---|---|
| `1_training_curves.png` | Training reward over 800 episodes for all 3 agents |
| `2_trajectories_2d.png` | 2D agent trajectories overlaid on the figure-eight target |
| `3_tracking_error.png` | Mean tracking error over training |
| `4_eval_comparison.png` | Evaluation bar chart comparing all 3 agents |

## Results and Analysis

### 1D vs 2D Side-by-Side

| Metric | 1D (Simple) | 2D (Complex) |
|---|---|---|
| Q-table entries | 123 | 3,969 |
| Naive eval error | 0.1236 | 0.4413 |
| Dropout-Aware eval error | 0.1236 | 0.3730 |
| Gap (Naive vs. Aware) | 0% | **15.5%** |
| Same policy? | Yes | No |

### Key Takeaways

**1. When in doubt, don't act.**
Acting on bad information is worse than waiting. The Dropout-Aware agent improved by simply skipping uncertain updates. ~28% of steps are skipped, but the remaining 72% write accurate values to the correct Q-table entries.

**2. Simple beats clever.**
A sophisticated state estimator made things worse (-68.10 vs. -45.94 for Naive). The best strategy was the most obvious one: if the observation is stale, skip the update.

**3. Complexity reveals the truth.**
The 1D problem (123 Q-entries) is too simple to distinguish the two approaches. The 2D problem (3,969 Q-entries) reveals a clear 15.5% advantage, validating the theoretical claim from the original paper.

**4. You don't need to define the goal.**
Inverse RL discovered the reward from expert demonstrations alone. The 3x2 comparison (3 agents x 2 reward sources) produced identical results, confirming that the expert's implicit objective was successfully recovered. The objective reveals itself through behavior.

## Conclusions

This project demonstrates that explicitly accounting for data dropouts in the learning process yields measurable improvements in trajectory tracking under lossy network conditions. The Dropout-Aware agent's strategy of skipping Q-updates during unreliable observations is simple, requires no model of the network, and produces a more robust policy — provided the problem has sufficient complexity to expose the effect.

Additionally, the inverse RL pipeline confirms that hand-crafted reward functions are not strictly necessary: feature-based reward matching can recover the reward from expert demonstrations alone, producing policies with identical performance to those trained with the original reward.

Our findings reinforce the central message of Fan et al. (2025): learning from incomplete data is both possible and beneficial when dropout is explicitly modeled.

## Project Structure

```
final_project/
├── experiment_1d/                  # 1D sine wave tracking experiment
│   ├── messy_network_trajectory_follower.py   # Main entry point
│   ├── config.py                   # Hyperparameters
│   ├── environment.py              # TrackingEnvironment (1D discretized)
│   ├── agents.py                   # BaselineAgent, NaiveAgent, DropoutAwareAgent
│   ├── dropout.py                  # DropoutChannel (lossy network simulation)
│   ├── evaluation.py               # Evaluation with/without dropout
│   ├── irl.py                      # Feature extraction + IRL gradient loop
│   ├── state_estimator.py          # State estimator (ablation study)
│   ├── plotting.py                 # All visualization functions
│   └── output_plots/               # 7 generated figures
├── experiment_2d/                  # 2D figure-eight tracking experiment
│   ├── main.py                     # Main entry point
│   ├── config.py                   # Hyperparameters
│   ├── environment.py              # TrackingEnvironment2D (2D discretized)
│   ├── agents.py                   # BaselineAgent, NaiveAgent, DropoutAwareAgent
│   ├── dropout.py                  # DropoutChannel (lossy network simulation)
│   ├── evaluation.py               # Evaluation with/without dropout
│   ├── plotting.py                 # All visualization functions
│   └── output_plots/               # 4 generated figures
├── create_presentation.py          # Generates presentation.pptx (python-pptx)
├── create_abstract.py              # Generates abstract.docx (python-docx)
├── presentation.pptx               # 25-slide presentation (dark theme)
└── messy_network_trajectory_follower_plan.txt
```

## How to Run

### Requirements

- Python 3.10+
- NumPy
- Matplotlib
- python-pptx (for presentation generation only)
- python-docx (for abstract generation only)

### Running the Experiments

```bash
# 1D sine wave experiment (includes IRL)
cd experiment_1d
python messy_network_trajectory_follower.py

# 2D figure-eight experiment
cd experiment_2d
python main.py
```

### Generating Deliverables

```bash
# Generate the 25-slide presentation
python create_presentation.py    # produces presentation.pptx

# Generate the formatted abstract
python create_abstract.py        # produces abstract.docx
```

## References

Fan, J., Shi, P., Xue, W., Lian, B., Cui, Y., & Lewis, F. L. (2025). Inverse Reinforcement Learning for Discrete-Time Systems With Data Dropouts. *IEEE Transactions on Cybernetics*, 55(4), 1744-1757.
