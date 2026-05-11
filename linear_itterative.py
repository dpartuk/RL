import numpy as np

# Define the coefficient matrix A
A = np.array([
    [ 0.65,  -0.175, -0.175,  0    ],
    [-0.175,   0.65,   0,     -0.175 ],
    [-0.175,   0,      0.65,  -0.175 ],
    [ 0.175,  -0.175, -0.175,  0.65  ]
])

# Define the right-hand side vector b
b = np.array([1.25, 2.25, 0, 1.25])

# Solve the system of equations
V = np.linalg.solve(A, b)

# Print the solution
print("V_pi(A) = {:.4f}".format(V))
print("V_pi(B) = {:.4f}".format(V))
print("V_pi(C) = {:.4f}".format(V))
print("V_pi(D) = {:.4f}".format(V))
