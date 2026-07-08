import logging
from dataclasses import dataclass

import numpy as np

from config import (
    Action, NUM_STATES, NUM_ACTIONS,
    ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY,
    NUM_STEPS_PER_EPISODE, NUM_TRAIN_EPISODES,
)
from environment import TrackingEnvironment2D
from dropout import DropoutChannel

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    Q: np.ndarray
    episode_rewards: list[float]
    episode_skips: list[int]


class BaseQLearner:
    """Base tabular Q-learner for 2D tracking."""

    def __init__(self):
        self.Q = np.zeros((NUM_STATES, NUM_ACTIONS))
        self.epsilon = EPSILON_START

    def select_action(self, state: int) -> Action:
        if np.random.rand() < self.epsilon:
            return Action(np.random.randint(NUM_ACTIONS))
        return Action(np.argmax(self.Q[state]))

    def update_q(self, state: int, action: Action,
                 reward: float, next_state: int) -> None:
        best_next = np.max(self.Q[next_state])
        td_error = reward + GAMMA * best_next - self.Q[state, action]
        self.Q[state, action] += ALPHA * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


class BaselineAgent(BaseQLearner):
    """Trains with perfect observations (no dropout)."""

    def train(self) -> TrainResult:
        logger.info("[Baseline] Training with perfect observations...")
        episode_rewards = []

        for ep in range(NUM_TRAIN_EPISODES):
            env = TrackingEnvironment2D()
            state = env.reset()
            total_reward = 0.0

            for t in range(NUM_STEPS_PER_EPISODE):
                action = self.select_action(state)
                next_state, reward, _, _ = env.step(action)
                self.update_q(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            self.decay_epsilon()
            episode_rewards.append(total_reward)

            if (ep + 1) % 100 == 0:
                avg = np.mean(episode_rewards[-20:])
                logger.info(f"  Episode {ep+1}/{NUM_TRAIN_EPISODES} "
                            f"avg(last20)={avg:.2f}")

        return TrainResult(self.Q.copy(), episode_rewards, [0] * NUM_TRAIN_EPISODES)


class NaiveAgent(BaseQLearner):
    """Trains under dropout but always updates Q-table."""

    def train(self) -> TrainResult:
        logger.info("[Naive] Training under dropout (always update)...")
        episode_rewards = []

        for ep in range(NUM_TRAIN_EPISODES):
            env = TrackingEnvironment2D()
            channel = DropoutChannel()
            true_state = env.reset()
            channel.reset(true_state)
            observed_state = true_state
            total_reward = 0.0

            for t in range(NUM_STEPS_PER_EPISODE):
                action = self.select_action(observed_state)
                true_next, reward, _, _ = env.step(action)
                observed_next, _ = channel.transmit(true_next)

                self.update_q(observed_state, action, reward, observed_next)
                observed_state = observed_next
                total_reward += reward

            self.decay_epsilon()
            episode_rewards.append(total_reward)

            if (ep + 1) % 100 == 0:
                avg = np.mean(episode_rewards[-20:])
                logger.info(f"  Episode {ep+1}/{NUM_TRAIN_EPISODES} "
                            f"avg(last20)={avg:.2f}")

        return TrainResult(self.Q.copy(), episode_rewards, [0] * NUM_TRAIN_EPISODES)


class DropoutAwareAgent(BaseQLearner):
    """Trains under dropout, skips Q-update when current state is stale."""

    def train(self) -> TrainResult:
        logger.info("[Dropout-Aware] Training (skip stale updates)...")
        episode_rewards = []
        episode_skips = []

        for ep in range(NUM_TRAIN_EPISODES):
            env = TrackingEnvironment2D()
            channel = DropoutChannel()
            true_state = env.reset()
            channel.reset(true_state)
            observed_state = true_state
            current_is_reliable = True
            total_reward = 0.0
            skips = 0

            for t in range(NUM_STEPS_PER_EPISODE):
                action = self.select_action(observed_state)
                true_next, reward, _, _ = env.step(action)
                observed_next, was_dropped = channel.transmit(true_next)

                if current_is_reliable:
                    self.update_q(observed_state, action, reward, observed_next)
                else:
                    skips += 1

                current_is_reliable = not was_dropped
                observed_state = observed_next
                total_reward += reward

            self.decay_epsilon()
            episode_rewards.append(total_reward)
            episode_skips.append(skips)

            if (ep + 1) % 100 == 0:
                avg = np.mean(episode_rewards[-20:])
                avg_skip = np.mean(episode_skips[-20:])
                logger.info(f"  Episode {ep+1}/{NUM_TRAIN_EPISODES} "
                            f"avg(last20)={avg:.2f} skips={avg_skip:.0f}")

        return TrainResult(self.Q.copy(), episode_rewards, episode_skips)
