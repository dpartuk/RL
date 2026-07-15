"""2D Figure-Eight Trajectory Follower with Data Dropouts.

Extends the 1D sine-tracking experiment to a more complex 2D environment
where the difference between Naive and Dropout-Aware agents becomes visible
at evaluation time.
"""

import logging
import sys

import numpy as np

from config import NUM_TRAIN_EPISODES, NUM_EVAL_EPISODES, NUM_STATES, NUM_ACTIONS
from agents import BaselineAgent, NaiveAgent, DropoutAwareAgent
from evaluation import evaluate_agent, EvalMode
from plotting import (
    plot_training_curves,
    plot_2d_trajectories,
    plot_tracking_error,
    plot_eval_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 65)
    logger.info("  2D FIGURE-EIGHT TRAJECTORY FOLLOWER WITH DATA DROPOUTS")
    logger.info("=" * 65)
    logger.info(f"  State space: {NUM_STATES} states ({int(NUM_STATES**0.5)}x"
                f"{int(NUM_STATES**0.5)} bins)")
    logger.info(f"  Action space: {NUM_ACTIONS} actions (8 directions + stay)")
    logger.info(f"  Q-table size: {NUM_STATES * NUM_ACTIONS:,} entries")
    logger.info(f"  Training: {NUM_TRAIN_EPISODES} episodes")
    logger.info("")

    # ---- Train all three agents ----
    np.random.seed(42)
    baseline = BaselineAgent()
    result_base = baseline.train()

    np.random.seed(42)
    naive = NaiveAgent()
    result_naive = naive.train()

    np.random.seed(42)
    aware = DropoutAwareAgent()
    result_aware = aware.train()

    # ---- Training summary ----
    logger.info("\n" + "=" * 65)
    logger.info("  TRAINING SUMMARY")
    logger.info("=" * 65)
    for name, res in [("Baseline", result_base), ("Naive", result_naive),
                      ("Dropout-Aware", result_aware)]:
        first20 = np.mean(res.episode_rewards[:20])
        last20 = np.mean(res.episode_rewards[-20:])
        logger.info(f"  {name:<15} first-20={first20:.2f}  last-20={last20:.2f}")

    # ---- Plot training curves ----
    plot_training_curves(
        result_base.episode_rewards,
        result_naive.episode_rewards,
        result_aware.episode_rewards,
    )
    logger.info("\n  Plot 1: Training curves saved.")

    # ---- Evaluate all three agents ----
    logger.info("\n" + "=" * 65)
    logger.info("  EVALUATION (greedy policy)")
    logger.info("=" * 65)

    # Baseline: no dropout (upper bound — perfect info)
    np.random.seed(99)
    eval_base = evaluate_agent(result_base.Q, EvalMode.BASELINE,
                               NUM_EVAL_EPISODES, dropout_prob=0.0)
    # Naive & Dropout-Aware: 30% dropout (realistic conditions)
    np.random.seed(99)
    eval_naive = evaluate_agent(result_naive.Q, EvalMode.NAIVE, NUM_EVAL_EPISODES)
    np.random.seed(99)
    eval_aware = evaluate_agent(result_aware.Q, EvalMode.NAIVE, NUM_EVAL_EPISODES)

    logger.info(f"\n  {'Agent':<25} {'Mean Track Err':>14} {'Total Reward':>14}")
    logger.info(f"  {'-'*25} {'-'*14} {'-'*14}")
    for name, ev in [("Baseline (no dropout)", eval_base),
                     ("Naive (30% dropout)", eval_naive),
                     ("Dropout-Aware (30% drop)", eval_aware)]:
        logger.info(f"  {name:<25} {ev.mean_tracking_error:>14.4f} "
                    f"{ev.total_reward:>14.2f}")

    # ---- Check if there's a gap between naive and aware ----
    gap = eval_naive.mean_tracking_error - eval_aware.mean_tracking_error
    pct = (gap / eval_naive.mean_tracking_error) * 100 if eval_naive.mean_tracking_error > 0 else 0
    logger.info(f"\n  Gap (Naive - Aware): {gap:.4f} ({pct:.1f}%)")
    if abs(gap) > 0.001:
        logger.info("  >> Dropout-Aware shows DIFFERENT eval performance than Naive!")
    else:
        logger.info("  >> Both converged to same eval performance.")

    # ---- Plot trajectories ----
    plot_2d_trajectories(eval_base, eval_naive, eval_aware)
    logger.info("  Plot 2: 2D trajectories saved.")

    # ---- Plot tracking error ----
    plot_tracking_error(eval_base, eval_naive, eval_aware)
    logger.info("  Plot 3: Tracking error saved.")

    # ---- Plot bar comparison ----
    plot_eval_comparison({
        "Baseline": eval_base,
        "Naive": eval_naive,
        "Dropout-Aware": eval_aware,
    })
    logger.info("  Plot 4: Eval comparison saved.")

    logger.info("\n" + "=" * 65)
    logger.info("  DONE — all plots saved to output_plots/")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
