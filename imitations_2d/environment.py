import numpy as np

from config import (
    NUM_BINS_PER_AXIS, POSITION_RANGE, BIN_WIDTH,
    ACTION_EFFECTS, AGENT_STEP_SIZE,
    AMPLITUDE_X, AMPLITUDE_Y, PERIOD_X, PERIOD_Y,
    NUM_STEPS_PER_EPISODE, Action,
)


class TrackingEnvironment2D:
    """2D trajectory tracking: agent follows a Lissajous figure-eight target."""

    def __init__(self):
        self.agent_x = 0.0
        self.agent_y = 0.0
        self.t = 0

    def target_position(self, t: int) -> tuple[float, float]:
        target_x = AMPLITUDE_X * np.sin(2 * np.pi * t / PERIOD_X)
        target_y = AMPLITUDE_Y * np.sin(2 * np.pi * t / PERIOD_Y)
        return target_x, target_y

    def _discretize(self, rel_x: float, rel_y: float) -> int:
        bin_x = int(round((rel_x + POSITION_RANGE) / BIN_WIDTH))
        bin_x = max(0, min(NUM_BINS_PER_AXIS - 1, bin_x))
        bin_y = int(round((rel_y + POSITION_RANGE) / BIN_WIDTH))
        bin_y = max(0, min(NUM_BINS_PER_AXIS - 1, bin_y))
        return bin_y * NUM_BINS_PER_AXIS + bin_x

    def reset(self) -> int:
        self.agent_x = 0.0
        self.agent_y = 0.0
        self.t = 0
        target_x, target_y = self.target_position(0)
        rel_x = self.agent_x - target_x
        rel_y = self.agent_y - target_y
        return self._discretize(rel_x, rel_y)

    def step(self, action: Action) -> tuple[int, float, float, float]:
        effect = ACTION_EFFECTS[action]
        self.agent_x += effect[0] * AGENT_STEP_SIZE
        self.agent_y += effect[1] * AGENT_STEP_SIZE

        self.agent_x = np.clip(self.agent_x, -POSITION_RANGE, POSITION_RANGE)
        self.agent_y = np.clip(self.agent_y, -POSITION_RANGE, POSITION_RANGE)

        self.t += 1
        target_x, target_y = self.target_position(self.t)
        rel_x = self.agent_x - target_x
        rel_y = self.agent_y - target_y

        distance = np.sqrt(rel_x ** 2 + rel_y ** 2)
        reward = -distance

        state = self._discretize(rel_x, rel_y)
        return state, reward, rel_x, rel_y
