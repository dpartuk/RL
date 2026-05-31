import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from config import Action, NUM_ACTIONS, NUM_STEPS_PER_EPISODE
from environment import TrackingEnvironment
from dropout import DropoutChannel

logger = logging.getLogger(__name__)


class EvalMode(Enum):
    BASELINE = auto()
    NAIVE = auto()
    DROPOUT_AWARE = auto()


@dataclass
class EvalResult:
    tracking_errors: np.ndarray
    cumulative_rewards: np.ndarray
    agent_trajectory: np.ndarray
    target_trajectory: np.ndarray
    dropout_mask: np.ndarray
    total_reward: float
    mean_tracking_error: float


def evaluate_agent(Q: np.ndarray, mode: EvalMode,
                   num_episodes: int,
                   dropout_prob: float | None = None) -> EvalResult:
    """Evaluate a trained Q-table using greedy policy over multiple episodes.

    Args:
        Q: trained Q-table
        mode: BASELINE (no dropout), NAIVE, or DROPOUT_AWARE
        num_episodes: number of episodes to average over
        dropout_prob: override dropout probability (None uses config default)
    """
    all_errors = np.zeros((num_episodes, NUM_STEPS_PER_EPISODE))
    all_cumulative = np.zeros((num_episodes, NUM_STEPS_PER_EPISODE))

    sample_agent_traj = None
    sample_target_traj = None
    sample_dropout_mask = None

    use_dropout = mode in (EvalMode.NAIVE, EvalMode.DROPOUT_AWARE)

    for ep in range(num_episodes):
        env = TrackingEnvironment()
        if use_dropout:
            channel = DropoutChannel(dropout_prob) if dropout_prob is not None else DropoutChannel()
        else:
            channel = None

        true_state = env.reset()
        if channel:
            channel.reset(true_state)
        observed_state = true_state
        current_is_reliable = True

        agent_positions = [env.agent_pos]
        target_positions = [env.target_position(0)]
        dropout_flags = []
        cumulative = 0.0

        for t in range(NUM_STEPS_PER_EPISODE):
            action = Action(np.argmax(Q[observed_state]))

            true_next, reward, rel_dist, target_pos = env.step(action)

            if channel:
                observed_next, was_dropped = channel.transmit(true_next)
            else:
                observed_next = true_next
                was_dropped = False

            if mode == EvalMode.DROPOUT_AWARE:
                observed_state = observed_next if current_is_reliable or not was_dropped else observed_next
                current_is_reliable = not was_dropped
            observed_state = observed_next

            cumulative += reward
            all_errors[ep, t] = abs(rel_dist)
            all_cumulative[ep, t] = cumulative
            agent_positions.append(env.agent_pos)
            target_positions.append(target_pos)
            dropout_flags.append(was_dropped)

        if ep == 0:
            sample_agent_traj = np.array(agent_positions)
            sample_target_traj = np.array(target_positions)
            sample_dropout_mask = np.array(dropout_flags)

    avg_errors = np.mean(all_errors, axis=0)
    avg_cumulative = np.mean(all_cumulative, axis=0)

    result = EvalResult(
        tracking_errors=avg_errors,
        cumulative_rewards=avg_cumulative,
        agent_trajectory=sample_agent_traj,
        target_trajectory=sample_target_traj,
        dropout_mask=sample_dropout_mask,
        total_reward=float(avg_cumulative[:, -1].mean()) if avg_cumulative.ndim > 1 else float(avg_cumulative[-1]),
        mean_tracking_error=float(np.mean(avg_errors)),
    )

    drop_info = ""
    if use_dropout:
        rate = dropout_prob if dropout_prob is not None else "default"
        drop_info = f" dropout={rate}"
    logger.info(
        f"[Eval:{mode.name}] {num_episodes} episodes:{drop_info} "
        f"mean_error={result.mean_tracking_error:.4f} "
        f"total_reward={result.total_reward:.2f}"
    )

    return result
