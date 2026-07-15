import logging
import numpy as np
from config import (
    Action, NUM_ACTIONS, NUM_POSITION_BINS,
    ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY,
    NUM_STEPS_PER_EPISODE,
)
from environment import TrackingEnvironment
from dropout import DropoutChannel

logger = logging.getLogger(__name__)

STEP_LOG_INTERVAL = 50


class BaseQLearner:
    """Base class for tabular Q-learning agents."""

    def __init__(self, name: str = "BaseQLearner"):
        self.name = name
        self.Q = np.zeros((NUM_POSITION_BINS, NUM_ACTIONS))
        self.epsilon = EPSILON_START
        self.training_rewards = []
        logger.info(f"[{self.name}] Initialized Q-table "
                     f"({NUM_POSITION_BINS}x{NUM_ACTIONS}), "
                     f"epsilon={self.epsilon}")

    def select_action(self, state: int) -> Action:
        if np.random.rand() < self.epsilon:
            return Action(np.random.randint(NUM_ACTIONS))
        return Action(np.argmax(self.Q[state]))

    def update_q(self, state: int, action: Action, reward: float,
                 next_state: int):
        best_next = np.max(self.Q[next_state])
        td_error = reward + GAMMA * best_next - self.Q[state, action]
        self.Q[state, action] += ALPHA * td_error

    def decay_epsilon(self):
        old_epsilon = self.epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        if int(old_epsilon * 100) != int(self.epsilon * 100):
            logger.info(f"[{self.name}] Epsilon: {old_epsilon:.4f} -> "
                         f"{self.epsilon:.4f}")

    def train(self, num_episodes: int):
        raise NotImplementedError


class BaselineAgent(BaseQLearner):
    """Q-learning with perfect observations — no dropout."""

    def __init__(self):
        super().__init__(name="Baseline")

    def train(self, num_episodes: int):
        env = TrackingEnvironment()
        logger.info(f"[{self.name}] Starting training for "
                     f"{num_episodes} episodes")

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            cumulative_reward = 0.0

            for t in range(NUM_STEPS_PER_EPISODE):
                action = self.select_action(state)
                next_state, reward, rel_dist, target_pos = env.step(action)
                self.update_q(state, action, reward, next_state)
                cumulative_reward += reward

                if (t + 1) % STEP_LOG_INTERVAL == 0:
                    logger.debug(
                        f"[{self.name}] Ep {episode+1} Step {t+1:>3}: "
                        f"state={state:>2}->{next_state:>2} "
                        f"action={action.name:<5} "
                        f"reward={reward:>7.3f} "
                        f"rel_dist={rel_dist:>7.3f} "
                        f"cum_reward={cumulative_reward:>8.2f}"
                    )

                state = next_state
                episode_reward += reward

            self.decay_epsilon()
            self.training_rewards.append(episode_reward)

            logger.info(
                f"[{self.name}] Episode {episode+1:>3}/{num_episodes} "
                f"reward={episode_reward:>8.2f} "
                f"epsilon={self.epsilon:.4f} "
                f"avg_reward={np.mean(self.training_rewards[-20:]):>8.2f}"
            )

        logger.info(f"[{self.name}] Training complete. "
                     f"Final avg reward (last 20): "
                     f"{np.mean(self.training_rewards[-20:]):.2f}")

        return self.Q, self.training_rewards


class NaiveAgent(BaseQLearner):
    """Q-learning with dropout — uses stale state on packet loss, no special handling."""

    def __init__(self):
        super().__init__(name="Naive")

    def train(self, num_episodes: int):
        env = TrackingEnvironment()
        channel = DropoutChannel()
        logger.info(f"[{self.name}] Starting training for "
                     f"{num_episodes} episodes "
                     f"(dropout_prob={channel.dropout_prob})")

        for episode in range(num_episodes):
            true_state = env.reset()
            channel.reset(true_state)
            observed_state = true_state
            episode_reward = 0.0
            cumulative_reward = 0.0
            episode_drops = 0

            for t in range(NUM_STEPS_PER_EPISODE):
                action = self.select_action(observed_state)
                true_next_state, reward, rel_dist, target_pos = env.step(action)
                observed_next, was_dropped = channel.transmit(true_next_state)
                if was_dropped:
                    episode_drops += 1

                self.update_q(observed_state, action, reward, observed_next)
                cumulative_reward += reward

                if (t + 1) % STEP_LOG_INTERVAL == 0:
                    logger.debug(
                        f"[{self.name}] Ep {episode+1} Step {t+1:>3}: "
                        f"true={true_next_state:>2} "
                        f"observed={observed_next:>2} "
                        f"{'DROPPED' if was_dropped else 'OK     '} "
                        f"action={action.name:<5} "
                        f"reward={reward:>7.3f} "
                        f"cum_reward={cumulative_reward:>8.2f}"
                    )

                observed_state = observed_next
                episode_reward += reward

            self.decay_epsilon()
            self.training_rewards.append(episode_reward)

            logger.info(
                f"[{self.name}] Episode {episode+1:>3}/{num_episodes} "
                f"reward={episode_reward:>8.2f} "
                f"epsilon={self.epsilon:.4f} "
                f"drops={episode_drops}/{NUM_STEPS_PER_EPISODE} "
                f"avg_reward={np.mean(self.training_rewards[-20:]):>8.2f}"
            )

        logger.info(f"[{self.name}] Training complete. "
                     f"Final avg reward (last 20): "
                     f"{np.mean(self.training_rewards[-20:]):.2f}, "
                     f"overall drop rate: {channel.drop_rate:.1%}")

        return self.Q, self.training_rewards


class DropoutAwareAgent(BaseQLearner):
    """Q-learning that skips updates when the current state is unreliable.

    When a packet is dropped, the observed state is stale — it points to
    a Q-table entry that may not represent where the agent truly is.
    Updating that entry corrupts the Q-table. This agent tracks whether
    its current state came from a fresh observation, and only performs
    Q-updates when it did.
    """

    def __init__(self):
        super().__init__(name="DropoutAware")

    def train(self, num_episodes: int):
        env = TrackingEnvironment()
        channel = DropoutChannel()
        logger.info(f"[{self.name}] Starting training for "
                     f"{num_episodes} episodes "
                     f"(dropout_prob={channel.dropout_prob})")

        for episode in range(num_episodes):
            true_state = env.reset()
            channel.reset(true_state)
            observed_state = true_state
            current_is_reliable = True
            episode_reward = 0.0
            cumulative_reward = 0.0
            episode_drops = 0
            episode_skips = 0

            for t in range(NUM_STEPS_PER_EPISODE):
                action = self.select_action(observed_state)
                true_next_state, reward, rel_dist, target_pos = env.step(action)
                observed_next, was_dropped = channel.transmit(true_next_state)
                if was_dropped:
                    episode_drops += 1

                if current_is_reliable:
                    self.update_q(observed_state, action, reward, observed_next)
                else:
                    episode_skips += 1

                cumulative_reward += reward

                if (t + 1) % STEP_LOG_INTERVAL == 0:
                    logger.debug(
                        f"[{self.name}] Ep {episode+1} Step {t+1:>3}: "
                        f"true={true_next_state:>2} "
                        f"obs={observed_next:>2} "
                        f"{'DROPPED' if was_dropped else 'OK     '} "
                        f"{'SKIP' if not current_is_reliable else 'UPD '} "
                        f"action={action.name:<5} "
                        f"reward={reward:>7.3f} "
                        f"cum_reward={cumulative_reward:>8.2f}"
                    )

                current_is_reliable = not was_dropped
                observed_state = observed_next
                episode_reward += reward

            self.decay_epsilon()
            self.training_rewards.append(episode_reward)

            logger.info(
                f"[{self.name}] Episode {episode+1:>3}/{num_episodes} "
                f"reward={episode_reward:>8.2f} "
                f"epsilon={self.epsilon:.4f} "
                f"drops={episode_drops} skips={episode_skips}/{NUM_STEPS_PER_EPISODE} "
                f"avg_reward={np.mean(self.training_rewards[-20:]):>8.2f}"
            )

        logger.info(f"[{self.name}] Training complete. "
                     f"Final avg reward (last 20): "
                     f"{np.mean(self.training_rewards[-20:]):.2f}, "
                     f"overall drop rate: {channel.drop_rate:.1%}")

        return self.Q, self.training_rewards
