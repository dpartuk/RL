import numpy as np
from enum import IntEnum


class Action(IntEnum):
    LEFT = 0
    STAY = 1
    RIGHT = 2


# --- Environment ---
NUM_POSITION_BINS = 41
POSITION_RANGE = 4.0
BIN_WIDTH = 2 * POSITION_RANGE / (NUM_POSITION_BINS - 1)

ACTION_EFFECTS = np.array([-1, 0, 1])
NUM_ACTIONS = len(Action)
AGENT_STEP_SIZE = 0.2

# --- Target Trajectory ---
AMPLITUDE = 2.0
PERIOD = 100
NUM_STEPS_PER_EPISODE = 300

# --- Dropout ---
DROPOUT_PROB = 0.3

# --- Q-Learning ---
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
NUM_TRAIN_EPISODES = 500
NUM_EVAL_EPISODES = 50

# --- State Estimator (dropout-aware agent) ---
CONFIDENCE_FRESH = 1.0
CONFIDENCE_DECAY = 0.6

# --- Inverse RL ---
NUM_FEATURE_DIMS = 5
IRL_LEARNING_RATE = 0.01
IRL_ITERATIONS = 200
NUM_EXPERT_DEMOS = 20
