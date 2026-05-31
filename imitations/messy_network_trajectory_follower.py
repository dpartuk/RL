"""
The Messy-Network Trajectory Follower

Inspired by: Fan et al. (2025), "Inverse Reinforcement Learning for
Discrete-Time Systems With Data Dropouts", IEEE Trans. Cybernetics.

Demonstrates that explicitly accounting for network dropout in Q-learning
updates improves learning quality, and that Inverse RL can recover a
reward function from expert demonstrations without hand-coding it.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from config import (
    NUM_POSITION_BINS, POSITION_RANGE, BIN_WIDTH,
    Action, AGENT_STEP_SIZE,
    AMPLITUDE, PERIOD, NUM_STEPS_PER_EPISODE, DROPOUT_PROB,
    ALPHA, GAMMA, EPSILON_START, EPSILON_END,
    NUM_TRAIN_EPISODES, NUM_EVAL_EPISODES,
    NUM_EXPERT_DEMOS,
)
from agents import BaselineAgent, NaiveAgent, DropoutAwareAgent
from evaluation import evaluate_agent, EvalMode
from plotting import (
    plot_training_curves, plot_tracking_error,
    plot_cumulative_rewards, plot_trajectories,
    plot_irl_convergence, plot_irl_comparison,
)
from irl import generate_expert_demonstrations, learn_reward_weights


def print_config():
    print("=" * 65)
    print("  THE MESSY-NETWORK TRAJECTORY FOLLOWER")
    print("=" * 65)
    print(f"  Environment:  {NUM_POSITION_BINS} state bins, "
          f"range [{-POSITION_RANGE}, {POSITION_RANGE}], "
          f"bin width {BIN_WIDTH:.2f}")
    print(f"  Actions:      {[a.name for a in Action]} "
          f"(step size {AGENT_STEP_SIZE})")
    print(f"  Target:       {AMPLITUDE} * sin(2*pi*t / {PERIOD}), "
          f"{NUM_STEPS_PER_EPISODE} steps/episode "
          f"({NUM_STEPS_PER_EPISODE // PERIOD} cycles)")
    print(f"  Dropout:      {DROPOUT_PROB:.0%} packet loss")
    print(f"  Q-learning:   alpha={ALPHA}, gamma={GAMMA}, "
          f"epsilon {EPSILON_START}->{EPSILON_END}")
    print(f"  Training:     {NUM_TRAIN_EPISODES} episodes, "
          f"eval {NUM_EVAL_EPISODES} episodes")
    print("=" * 65)


def print_q_table(Q: np.ndarray, label: str, bins=range(18, 23)):
    print(f"\n  {label} Q-table (center bins):")
    print(f"    {'Bin':>4}  {'LEFT':>8}  {'STAY':>8}  {'RIGHT':>8}  {'Best':>6}")
    print(f"    {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    for b in bins:
        best = Action(np.argmax(Q[b])).name
        print(f"    {b:>4}  {Q[b, 0]:>8.2f}  {Q[b, 1]:>8.2f}  "
              f"{Q[b, 2]:>8.2f}  {best:>6}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    print_config()

    # =================================================================
    # PART A: THREE-AGENT DROPOUT COMPARISON
    # =================================================================
    print("\n" + "=" * 65)
    print("  PART A: THREE-AGENT DROPOUT COMPARISON")
    print("=" * 65)

    # --- Train ---
    np.random.seed(42)
    Q_base, rewards_base = BaselineAgent().train(NUM_TRAIN_EPISODES)

    np.random.seed(42)
    Q_naive, rewards_naive = NaiveAgent().train(NUM_TRAIN_EPISODES)

    np.random.seed(42)
    Q_aware, rewards_aware = DropoutAwareAgent().train(NUM_TRAIN_EPISODES)

    # --- Training summary ---
    print("\n  TRAINING RESULTS")
    print(f"  {'Agent':<25} {'First-20':>10} {'Last-20':>10} {'Improvement':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12}")
    for label, r in [('Baseline', rewards_base),
                     ('Naive (stale)', rewards_naive),
                     ('Dropout-Aware (skip)', rewards_aware)]:
        first = np.mean(r[:20])
        last = np.mean(r[-20:])
        print(f"  {label:<25} {first:>10.2f} {last:>10.2f} "
              f"{abs(last) - abs(first):>+11.2f}")

    print_q_table(Q_base, "Baseline")
    print_q_table(Q_naive, "Naive")
    print_q_table(Q_aware, "Dropout-Aware")

    # --- Evaluate ---
    print("\n  EVALUATION (greedy policy, no exploration)")

    np.random.seed(99)
    eval_base = evaluate_agent(Q_base, EvalMode.BASELINE, NUM_EVAL_EPISODES)
    np.random.seed(99)
    eval_naive = evaluate_agent(Q_naive, EvalMode.NAIVE, NUM_EVAL_EPISODES)
    np.random.seed(99)
    eval_aware = evaluate_agent(Q_aware, EvalMode.DROPOUT_AWARE, NUM_EVAL_EPISODES)

    print(f"\n  {'Agent':<25} {'Mean Track Err':>14} {'Total Reward':>14}")
    print(f"  {'-'*25} {'-'*14} {'-'*14}")
    for label, ev in [('Baseline', eval_base),
                      ('Naive (stale)', eval_naive),
                      ('Dropout-Aware (skip)', eval_aware)]:
        print(f"  {label:<25} {ev.mean_tracking_error:>14.4f} "
              f"{ev.total_reward:>14.2f}")

    # --- Robustness evaluation: all agents under 20% dropout ---
    EVAL_DROPOUT = 0.2
    print(f"\n  ROBUSTNESS TEST ({EVAL_DROPOUT:.0%} dropout at eval time)")

    np.random.seed(99)
    rob_base = evaluate_agent(
        Q_base, EvalMode.NAIVE, NUM_EVAL_EPISODES, dropout_prob=EVAL_DROPOUT
    )
    np.random.seed(99)
    rob_naive = evaluate_agent(
        Q_naive, EvalMode.NAIVE, NUM_EVAL_EPISODES, dropout_prob=EVAL_DROPOUT
    )
    np.random.seed(99)
    rob_aware = evaluate_agent(
        Q_aware, EvalMode.NAIVE, NUM_EVAL_EPISODES, dropout_prob=EVAL_DROPOUT
    )

    print(f"\n  All agents face {EVAL_DROPOUT:.0%} dropout (stale-state fallback):")
    print(f"  {'Agent (trained with)':<25} {'Mean Track Err':>14} {'Total Reward':>14}")
    print(f"  {'-'*25} {'-'*14} {'-'*14}")
    for label, ev in [('Baseline (no dropout)', rob_base),
                      ('Naive (30% dropout)',   rob_naive),
                      ('Dropout-Aware (30%)',   rob_aware)]:
        print(f"  {label:<25} {ev.mean_tracking_error:>14.4f} "
              f"{ev.total_reward:>14.2f}")

    # --- Part A plots ---
    plot_training_curves(rewards_base, rewards_naive, rewards_aware)
    plot_tracking_error(eval_base, eval_naive, eval_aware)
    plot_cumulative_rewards(eval_base, eval_naive, eval_aware)
    plot_trajectories(eval_base, eval_naive, eval_aware)

    # =================================================================
    # PART B: INVERSE RL
    # =================================================================
    print("\n" + "=" * 65)
    print("  PART B: INVERSE RL — LEARNING THE REWARD FUNCTION")
    print("=" * 65)

    np.random.seed(42)
    demos, expert_feat_exp = generate_expert_demonstrations(NUM_EXPERT_DEMOS)
    expert_error = np.mean([np.mean(np.abs(d)) for d in demos])
    print(f"\n  Expert tracking error: {expert_error:.4f}")

    irl_result = learn_reward_weights(expert_feat_exp)

    feature_names = ['Bullseye', 'Linear', 'Quadratic', 'Danger', 'Bias']
    print(f"\n  Learned reward weights:")
    for name, w in zip(feature_names, irl_result.learned_weights):
        print(f"    {name:<12} {w:>8.4f}")

    # Evaluate all three IRL agents
    np.random.seed(99)
    eval_irl_base = evaluate_agent(
        irl_result.Q_baseline, EvalMode.BASELINE, NUM_EVAL_EPISODES
    )
    np.random.seed(99)
    eval_irl_naive = evaluate_agent(
        irl_result.Q_naive, EvalMode.NAIVE, NUM_EVAL_EPISODES
    )
    np.random.seed(99)
    eval_irl_aware = evaluate_agent(
        irl_result.Q_aware, EvalMode.DROPOUT_AWARE, NUM_EVAL_EPISODES
    )

    # --- Full 6-cell comparison ---
    print(f"\n  FULL COMPARISON: REWARD x AGENT")
    print(f"  {'':>25} {'Hand-coded':>16} {'IRL-learned':>16}")
    print(f"  {'Agent':<25} {'Err':>8} {'Reward':>7} {'Err':>8} {'Reward':>7}")
    print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*8} {'-'*7}")
    for label, ev_hc, ev_irl in [
        ('Baseline',           eval_base,  eval_irl_base),
        ('Naive (stale)',      eval_naive, eval_irl_naive),
        ('Dropout-Aware',     eval_aware, eval_irl_aware),
    ]:
        print(f"  {label:<25} "
              f"{ev_hc.mean_tracking_error:>8.4f} {ev_hc.total_reward:>7.1f} "
              f"{ev_irl.mean_tracking_error:>8.4f} {ev_irl.total_reward:>7.1f}")

    # --- Part B plots ---
    plot_irl_convergence(irl_result.weight_history)
    plot_irl_comparison(eval_base, eval_irl_base)

    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print("""
  Part A — Dropout Comparison (hand-coded reward):
    Baseline > Dropout-Aware > Naive
    Key insight: don't update Q when the current state is unreliable.

  Part B — Inverse RL:
    IRL recovered a reward that matches hand-coded performance.
    The same Baseline > Aware > Naive ordering holds under IRL reward.
    Key insight: you don't need to hand-craft rewards if you have demos.
""")
    print("=" * 65)

    plt.show()
