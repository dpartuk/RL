import logging
from dataclasses import dataclass

import numpy as np

from config import (
    Action, ACTION_EFFECTS, NUM_ACTIONS, NUM_POSITION_BINS,
    AGENT_STEP_SIZE, BIN_WIDTH, POSITION_RANGE,
    ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY,
    NUM_STEPS_PER_EPISODE, NUM_FEATURE_DIMS,
    IRL_LEARNING_RATE, IRL_ITERATIONS, NUM_EXPERT_DEMOS,
)
from environment import TrackingEnvironment
from dropout import DropoutChannel

logger = logging.getLogger(__name__)


def compute_features(relative_distance: float) -> np.ndarray:
    """5-dimensional feature vector for a given relative distance.

    f0: bullseye — 1.0 if within half a bin width of target
    f1: linear penalty — -|rel_dist|
    f2: quadratic penalty — -rel_dist^2
    f3: danger zone — -1.0 if far from target (>75% of range)
    f4: bias — constant 1.0
    """
    abs_dist = abs(relative_distance)
    features = np.zeros(NUM_FEATURE_DIMS)
    features[0] = 1.0 if abs_dist < 0.5 * BIN_WIDTH else 0.0
    features[1] = -abs_dist
    features[2] = -(relative_distance ** 2)
    features[3] = -1.0 if abs_dist > POSITION_RANGE * 0.75 else 0.0
    features[4] = 1.0
    return features


def generate_expert_demonstrations(num_demos: int) -> tuple[list, np.ndarray]:
    """Generate demonstrations from an optimal expert that always minimizes distance.

    Returns:
        demos: list of episodes, each a list of relative distances
        expert_feature_exp: average feature vector across all demo steps
    """
    logger.info(f"[IRL] Generating {num_demos} expert demonstrations")
    demos = []
    total_features = np.zeros(NUM_FEATURE_DIMS)
    total_steps = 0

    for d in range(num_demos):
        env = TrackingEnvironment()
        env.reset()
        episode_dists = []

        for t in range(NUM_STEPS_PER_EPISODE):
            best_action = None
            best_abs_dist = float('inf')
            for action in Action:
                next_pos = env.agent_pos + ACTION_EFFECTS[action] * AGENT_STEP_SIZE
                next_target = env.target_position(env.t + 1)
                abs_dist = abs(next_pos - next_target)
                if abs_dist < best_abs_dist:
                    best_abs_dist = abs_dist
                    best_action = action

            _, _, rel_dist, _ = env.step(best_action)
            episode_dists.append(rel_dist)
            total_features += compute_features(rel_dist)
            total_steps += 1

        demos.append(episode_dists)

    expert_feature_exp = total_features / total_steps

    logger.info(f"[IRL] Expert feature expectations: "
                f"[{', '.join(f'{v:.4f}' for v in expert_feature_exp)}]")

    return demos, expert_feature_exp


def compute_policy_feature_expectations(Q: np.ndarray,
                                        num_episodes: int) -> np.ndarray:
    """Run a greedy Q-policy and compute average feature expectations."""
    total_features = np.zeros(NUM_FEATURE_DIMS)
    total_steps = 0

    for ep in range(num_episodes):
        env = TrackingEnvironment()
        state = env.reset()

        for t in range(NUM_STEPS_PER_EPISODE):
            action = Action(np.argmax(Q[state]))
            state, _, rel_dist, _ = env.step(action)
            total_features += compute_features(rel_dist)
            total_steps += 1

    return total_features / total_steps


def _train_with_reward_weights(weights: np.ndarray,
                               num_episodes: int) -> np.ndarray:
    """Train a Q-learner using reward = dot(weights, features). Returns Q-table."""
    Q = np.zeros((NUM_POSITION_BINS, NUM_ACTIONS))
    epsilon = EPSILON_START

    for ep in range(num_episodes):
        env = TrackingEnvironment()
        state = env.reset()

        for t in range(NUM_STEPS_PER_EPISODE):
            if np.random.rand() < epsilon:
                action = Action(np.random.randint(NUM_ACTIONS))
            else:
                action = Action(np.argmax(Q[state]))

            next_state, _, rel_dist, _ = env.step(action)
            reward = float(np.dot(weights, compute_features(rel_dist)))

            best_next = np.max(Q[next_state])
            td_error = reward + GAMMA * best_next - Q[state, action]
            Q[state, action] += ALPHA * td_error
            state = next_state

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    return Q


def _train_naive_with_reward_weights(weights: np.ndarray,
                                     num_episodes: int) -> np.ndarray:
    """Train a naive agent (dropout, stale fallback) with learned reward."""
    Q = np.zeros((NUM_POSITION_BINS, NUM_ACTIONS))
    epsilon = EPSILON_START

    for ep in range(num_episodes):
        env = TrackingEnvironment()
        channel = DropoutChannel()
        true_state = env.reset()
        channel.reset(true_state)
        observed_state = true_state

        for t in range(NUM_STEPS_PER_EPISODE):
            if np.random.rand() < epsilon:
                action = Action(np.random.randint(NUM_ACTIONS))
            else:
                action = Action(np.argmax(Q[observed_state]))

            true_next, _, rel_dist, _ = env.step(action)
            reward = float(np.dot(weights, compute_features(rel_dist)))
            observed_next, _ = channel.transmit(true_next)

            best_next = np.max(Q[observed_next])
            td_error = reward + GAMMA * best_next - Q[observed_state, action]
            Q[observed_state, action] += ALPHA * td_error
            observed_state = observed_next

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    return Q


def _train_aware_with_reward_weights(weights: np.ndarray,
                                     num_episodes: int) -> np.ndarray:
    """Train a dropout-aware agent (skip stale updates) with learned reward."""
    Q = np.zeros((NUM_POSITION_BINS, NUM_ACTIONS))
    epsilon = EPSILON_START

    for ep in range(num_episodes):
        env = TrackingEnvironment()
        channel = DropoutChannel()
        true_state = env.reset()
        channel.reset(true_state)
        observed_state = true_state
        current_is_reliable = True

        for t in range(NUM_STEPS_PER_EPISODE):
            if np.random.rand() < epsilon:
                action = Action(np.random.randint(NUM_ACTIONS))
            else:
                action = Action(np.argmax(Q[observed_state]))

            true_next, _, rel_dist, _ = env.step(action)
            reward = float(np.dot(weights, compute_features(rel_dist)))
            observed_next, was_dropped = channel.transmit(true_next)

            if current_is_reliable:
                best_next = np.max(Q[observed_next])
                td_error = reward + GAMMA * best_next - Q[observed_state, action]
                Q[observed_state, action] += ALPHA * td_error

            current_is_reliable = not was_dropped
            observed_state = observed_next

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    return Q


IRL_TRAIN_EPISODES = 100
IRL_FEATURE_EVAL_EPISODES = 10
WEIGHT_CLIP = 10.0


@dataclass
class IRLResult:
    learned_weights: np.ndarray
    weight_history: np.ndarray
    Q_baseline: np.ndarray
    Q_naive: np.ndarray
    Q_aware: np.ndarray


def learn_reward_weights(expert_feature_exp: np.ndarray) -> IRLResult:
    """Learn reward weights via feature-matching IRL.

    At each iteration:
        1. Train a Q-learner using reward = dot(w, features)
        2. Compute the learner's feature expectations
        3. Push w toward making expert look more optimal:
           w += lr * (expert_features - learner_features)
    """
    weights = np.zeros(NUM_FEATURE_DIMS)
    weight_history = [weights.copy()]

    logger.info(f"[IRL] Starting reward weight learning "
                f"({IRL_ITERATIONS} iterations)")

    for i in range(IRL_ITERATIONS):
        Q = _train_with_reward_weights(weights, IRL_TRAIN_EPISODES)
        learner_feat_exp = compute_policy_feature_expectations(
            Q, IRL_FEATURE_EVAL_EPISODES
        )

        gradient = expert_feature_exp - learner_feat_exp
        weights = weights + IRL_LEARNING_RATE * gradient
        weights = np.clip(weights, -WEIGHT_CLIP, WEIGHT_CLIP)
        weight_history.append(weights.copy())

        if (i + 1) % 20 == 0:
            grad_norm = np.linalg.norm(gradient)
            logger.info(
                f"[IRL] Iteration {i+1:>3}/{IRL_ITERATIONS} "
                f"grad_norm={grad_norm:.6f} "
                f"weights=[{', '.join(f'{w:.3f}' for w in weights)}]"
            )

    logger.info(f"[IRL] Learning complete. "
                f"Final weights: [{', '.join(f'{w:.4f}' for w in weights)}]")

    logger.info("[IRL] Training final agents with learned reward...")
    Q_baseline = _train_with_reward_weights(weights, NUM_STEPS_PER_EPISODE)
    Q_naive = _train_naive_with_reward_weights(weights, NUM_STEPS_PER_EPISODE)
    Q_aware = _train_aware_with_reward_weights(weights, NUM_STEPS_PER_EPISODE)

    return IRLResult(
        learned_weights=weights,
        weight_history=np.array(weight_history),
        Q_baseline=Q_baseline,
        Q_naive=Q_naive,
        Q_aware=Q_aware,
    )
