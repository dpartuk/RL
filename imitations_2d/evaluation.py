import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from config import (
    Action, NUM_STEPS_PER_EPISODE, DROPOUT_PROB,
)
from environment import TrackingEnvironment2D
from dropout import DropoutChannel

logger = logging.getLogger(__name__)


class EvalMode(Enum):
    BASELINE = "baseline"
    NAIVE = "naive"


@dataclass
class EvalResult:
    mean_tracking_error: float
    total_reward: float
    per_step_errors: list[float]
    per_step_rewards: list[float]
    trajectories_x: list[float]
    trajectories_y: list[float]
    target_x: list[float]
    target_y: list[float]
    drop_mask: list[bool]


def evaluate_agent(Q: np.ndarray, mode: EvalMode,
                   num_episodes: int, dropout_prob: float = DROPOUT_PROB
                   ) -> EvalResult:
    all_errors = []
    all_rewards = []
    total_reward = 0.0
    last_traj_x, last_traj_y = [], []
    last_target_x, last_target_y = [], []
    last_drops: list[bool] = []

    for ep in range(num_episodes):
        env = TrackingEnvironment2D()
        true_state = env.reset()

        if mode == EvalMode.BASELINE:
            channel = DropoutChannel(dropout_prob=dropout_prob)
        else:
            channel = DropoutChannel(dropout_prob=dropout_prob)
        channel.reset(true_state)
        observed_state = true_state

        ep_errors, ep_rewards = [], []
        ep_traj_x, ep_traj_y = [env.agent_x], [env.agent_y]
        tx, ty = env.target_position(0)
        ep_target_x, ep_target_y = [tx], [ty]
        ep_drops: list[bool] = []

        for t in range(NUM_STEPS_PER_EPISODE):
            action = Action(np.argmax(Q[observed_state]))
            true_next, reward, rel_x, rel_y = env.step(action)

            if mode == EvalMode.BASELINE:
                observed_next, was_dropped = channel.transmit(true_next)
            else:
                observed_next, was_dropped = channel.transmit(true_next)

            observed_state = observed_next

            distance = np.sqrt(rel_x ** 2 + rel_y ** 2)
            ep_errors.append(distance)
            ep_rewards.append(reward)
            total_reward += reward

            ep_traj_x.append(env.agent_x)
            ep_traj_y.append(env.agent_y)
            tx, ty = env.target_position(env.t)
            ep_target_x.append(tx)
            ep_target_y.append(ty)
            ep_drops.append(was_dropped)

        all_errors.extend(ep_errors)
        all_rewards.extend(ep_rewards)
        last_traj_x, last_traj_y = ep_traj_x, ep_traj_y
        last_target_x, last_target_y = ep_target_x, ep_target_y
        last_drops = ep_drops

    mean_error = np.mean(all_errors)
    return EvalResult(
        mean_tracking_error=mean_error,
        total_reward=total_reward / num_episodes,
        per_step_errors=all_errors,
        per_step_rewards=all_rewards,
        trajectories_x=last_traj_x,
        trajectories_y=last_traj_y,
        target_x=last_target_x,
        target_y=last_target_y,
        drop_mask=last_drops,
    )
