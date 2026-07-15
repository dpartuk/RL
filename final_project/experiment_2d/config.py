import numpy as np
from enum import IntEnum


class Action(IntEnum):
    UP_LEFT = 0
    UP = 1
    UP_RIGHT = 2
    LEFT = 3
    STAY = 4
    RIGHT = 5
    DOWN_LEFT = 6
    DOWN = 7
    DOWN_RIGHT = 8


ACTION_EFFECTS = np.array([
    [-1, +1],   # UP_LEFT
    [ 0, +1],   # UP
    [+1, +1],   # UP_RIGHT
    [-1,  0],   # LEFT
    [ 0,  0],   # STAY
    [+1,  0],   # RIGHT
    [-1, -1],   # DOWN_LEFT
    [ 0, -1],   # DOWN
    [+1, -1],   # DOWN_RIGHT
])

NUM_ACTIONS = len(Action)

# --- Environment ---
NUM_BINS_PER_AXIS = 21
NUM_STATES = NUM_BINS_PER_AXIS ** 2  # 441
POSITION_RANGE = 4.0
BIN_WIDTH = 2 * POSITION_RANGE / (NUM_BINS_PER_AXIS - 1)  # 0.4
AGENT_STEP_SIZE = 0.3

# --- Target Trajectory (Lissajous figure-eight) ---
AMPLITUDE_X = 2.5
AMPLITUDE_Y = 1.8
PERIOD_X = 120
PERIOD_Y = 60  # 2:1 ratio creates figure-eight
NUM_STEPS_PER_EPISODE = 360  # 3 full figure-eight cycles

# --- Dropout ---
DROPOUT_PROB = 0.3

# --- Q-Learning ---
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
NUM_TRAIN_EPISODES = 800
NUM_EVAL_EPISODES = 50

# --- Inverse RL ---
NUM_FEATURE_DIMS = 6
IRL_LEARNING_RATE = 0.01
IRL_ITERATIONS = 200
NUM_EXPERT_DEMOS = 20
