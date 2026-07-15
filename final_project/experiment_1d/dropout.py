import logging
import numpy as np
from config import DROPOUT_PROB

logger = logging.getLogger(__name__)


class DropoutChannel:
    """Simulates a lossy network channel that randomly drops state observations.

    When a packet is dropped, the channel returns the last successfully
    received state instead of the true current state.
    """

    def __init__(self, dropout_prob: float = DROPOUT_PROB):
        self.dropout_prob = dropout_prob
        self.last_known_state = None
        self.total_steps = 0
        self.total_drops = 0

    def reset(self, initial_state: int):
        self.last_known_state = initial_state
        self.total_steps = 0
        self.total_drops = 0

    def transmit(self, true_state: int) -> tuple[int, bool]:
        """Send a state observation through the lossy channel.

        Returns:
            observed_state: the state the agent actually sees
            was_dropped: whether this observation was lost
        """
        self.total_steps += 1
        dropped = np.random.rand() < self.dropout_prob

        if dropped:
            self.total_drops += 1
            return self.last_known_state, True

        self.last_known_state = true_state
        return true_state, False

    @property
    def drop_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_drops / self.total_steps
