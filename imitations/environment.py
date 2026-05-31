import numpy as np
from config import (
    NUM_POSITION_BINS, POSITION_RANGE, BIN_WIDTH,
    Action, ACTION_EFFECTS, AGENT_STEP_SIZE, AMPLITUDE, PERIOD,
    NUM_STEPS_PER_EPISODE,
)


class TrackingEnvironment:
    """1D tracking environment where an agent follows a sinusoidal target.

    State is the relative distance (agent_pos - target_pos) discretized
    into bins. The agent chooses from {move_left, stay, move_right} at
    each timestep.
    """

    def __init__(self):
        self.agent_pos = 0.0
        self.t = 0
        self.state_bin = self._discretize(self.agent_pos - self.target_position(0))

    def reset(self) -> int:
        self.agent_pos = 0.0
        self.t = 0
        rel_dist = self.agent_pos - self.target_position(0)
        self.state_bin = self._discretize(rel_dist)
        return self.state_bin

    def step(self, action: Action) -> tuple:
        """Advance one timestep.

        Returns:
            next_state_bin: discretized relative distance after the step
            reward: -|relative_distance|
            rel_dist: continuous relative distance (for logging)
            target_pos: target position at t+1 (for logging)
        """
        self.agent_pos += ACTION_EFFECTS[action] * AGENT_STEP_SIZE
        self.t += 1
        target_pos = self.target_position(self.t)
        rel_dist = self.agent_pos - target_pos
        self.state_bin = self._discretize(rel_dist)
        reward = self._compute_reward(rel_dist)
        return self.state_bin, reward, rel_dist, target_pos

    @staticmethod
    def target_position(t: int) -> float:
        return AMPLITUDE * np.sin(2 * np.pi * t / PERIOD)

    @staticmethod
    def _discretize(relative_distance: float) -> int:
        clipped = np.clip(relative_distance, -POSITION_RANGE, POSITION_RANGE)
        bin_index = (clipped + POSITION_RANGE) / BIN_WIDTH
        return int(np.round(bin_index))

    @staticmethod
    def _compute_reward(relative_distance: float) -> float:
        return -abs(relative_distance)

    @property
    def num_states(self) -> int:
        return NUM_POSITION_BINS

    @property
    def num_actions(self) -> int:
        return len(Action)

    @property
    def max_steps(self) -> int:
        return NUM_STEPS_PER_EPISODE
