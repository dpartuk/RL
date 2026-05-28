import numpy as np

# =====================================================================
# 1. Iterative Solution
# =====================================================================

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
    V_new[0] = (1 / 4) * (5 + gamma * V[1]) + (1 / 4) * (gamma * V[2]) + (1 / 2) * (gamma * V[0])
    V_new[1] = (1 / 2) * (5 + gamma * V[1]) + (1 / 4) * (gamma * V[0]) + (1 / 4) * (gamma * V[3])
    V_new[2] = (1 / 4) * (gamma * V[0]) + (1 / 4) * (gamma * V[3]) + (1 / 2) * (gamma * V[2])
    V_new[3] = (1 / 4) * (5 + gamma * V[1]) + (1 / 4) * (gamma * V[2]) + (1 / 2) * (gamma * V[3])

    # Print progress every 50 steps (and on the first iteration)
    print(f"Iteration {iteration:3d}: V = [{V_new[0]:.4f}, {V_new[1]:.4f}, {V_new[2]:.4f}, {V_new[3]:.4f}]")
    
    # Check for convergence
    if np.max(np.abs(V_new - V)) < epsilon:
        print(f"Converged at iteration {iteration}")
        break
        
    # Update V for the next iteration
    V = V_new.copy()

print("--- Iterative Solution Results ---")
print("V_pi(A) = {:.4f}".format(V[0]))
print("V_pi(B) = {:.4f}".format(V[1]))
print("V_pi(C) = {:.4f}".format(V[2]))
print("V_pi(D) = {:.4f}\n".format(V[3]))


# =====================================================================
# 2. Linear System Solution
# =====================================================================

# Define the coefficient matrix A
A = np.array([
    [ 0.65,  -0.175, -0.175,  0     ],
    [-0.175,   0.65,   0,    -0.175 ],
    [-0.175,   0,      0.65, -0.175 ],
    [ 0,      -0.175, -0.175,  0.65  ]
])

# Define the right-hand side vector b
b = np.array([1.25, 2.5, 0, 1.25])

# Solve the system of equations
V_linear = np.linalg.solve(A, b)

print("--- Linear System Solution Results ---")
print("V_pi(A) = {:.4f}".format(V_linear[0]))
print("V_pi(B) = {:.4f}".format(V_linear[1]))
print("V_pi(C) = {:.4f}".format(V_linear[2]))
print("V_pi(D) = {:.4f}".format(V_linear[3]))
