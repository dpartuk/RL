import numpy as np

# Discount factor
gamma = 0.7

# Initial state values
V = np.zeros(4)
V_new = np.zeros(4)

# Convergence threshold
epsilon = 1e-6

# Maximum number of iterations
max_iterations = 1000

# Iterative update
for iteration in range(max_iterations):
    V_new = (1 / 4) * (5 + gamma * V) + (1 / 4) * \
               (gamma * V) + (1 / 2) * (gamma * V)
    V_new = (1 / 2) * (5 + gamma * V) + (1 / 4) * \
               (gamma * V) + (1 / 4) * (gamma * V)
    V_new = (1 / 4) * (gamma * V) + (1 / 4) * \
               (gamma * V) + (1 / 2) * (gamma * V)
    V_new = (1 / 4) * (5 + gamma * V) + (1 / 4) * \
               (gamma * V) + (1 / 2) * (gamma * V)

    # Check for convergence
    if np.max(np.abs(V_new - V)) < epsilon:
        break

    # Update V for the next iteration
    V = V_new.copy()
