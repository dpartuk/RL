import logging
import numpy as np
from config import NUM_POSITION_BINS, ACTION_EFFECTS, CONFIDENCE_FRESH, CONFIDENCE_DECAY

logger = logging.getLogger(__name__)


class StateEstimator:
    """Action-tracking state estimator for handling observation dropouts.

    When an observation is dropped, the estimator corrects the stale state
    using the actions the agent has taken since the last successful
    observation. Each action shifts the relative-distance bin by a known
    amount, so the agent's own movement is accounted for exactly — only
    the unknown target movement remains as error.

    Confidence decays exponentially with consecutive misses.
    """

    def __init__(self):
        self.last_known_bin = 0
        self.action_shift_since_obs = 0
        self.steps_since_obs = 0
        self.confidence = CONFIDENCE_FRESH

    def reset(self, initial_state: int):
        self.last_known_bin = initial_state
        self.action_shift_since_obs = 0
        self.steps_since_obs = 0
        self.confidence = CONFIDENCE_FRESH

    def update(self, observed_bin: int, was_dropped: bool,
               action_index: int) -> tuple[int, float]:
        """Process an observation and return best state estimate.

        Args:
            observed_bin: state from the channel (true if received, stale if dropped)
            was_dropped: whether this observation was lost
            action_index: the action the agent just took (needed to track movement)

        Returns:
            estimated_bin: best guess of current state bin
            confidence: trust level (1.0 = fresh observation, decays on drops)
        """
        action_effect = ACTION_EFFECTS[action_index]

        if not was_dropped:
            self.last_known_bin = observed_bin
            self.action_shift_since_obs = 0
            self.steps_since_obs = 0
            self.confidence = CONFIDENCE_FRESH
            return observed_bin, self.confidence

        self.steps_since_obs += 1
        self.action_shift_since_obs += action_effect
        predicted = self.last_known_bin + self.action_shift_since_obs
        estimated_bin = int(np.clip(np.round(predicted), 0, NUM_POSITION_BINS - 1))
        self.confidence = CONFIDENCE_FRESH * (CONFIDENCE_DECAY ** self.steps_since_obs)

        logger.debug(
            f"[Estimator] Drop #{self.steps_since_obs}: "
            f"last_obs={self.last_known_bin} "
            f"action_shift={self.action_shift_since_obs} "
            f"estimated={estimated_bin} "
            f"confidence={self.confidence:.3f}"
        )

        return estimated_bin, self.confidence
