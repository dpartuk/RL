import numpy as np

from config import DROPOUT_PROB


class DropoutChannel:
    """Simulates packet dropout: with probability DROPOUT_PROB, returns
    the last successfully received state instead of the current one."""

    def __init__(self, dropout_prob: float = DROPOUT_PROB):
        self.dropout_prob = dropout_prob
        self.last_known_state: int = 0
        self.drop_history: list[bool] = []

    def reset(self, initial_state: int) -> None:
        self.last_known_state = initial_state
        self.drop_history = []

    def transmit(self, true_state: int) -> tuple[int, bool]:
        if np.random.rand() < self.dropout_prob:
            self.drop_history.append(True)
            return self.last_known_state, True
        else:
            self.last_known_state = true_state
            self.drop_history.append(False)
            return true_state, False
