import numpy as np

# 1 Configuration
# Grid dimensions
GRID_SIZE = 5
ACTIONS = ['N', 'S', 'E', 'W']
ACTION_PROBABILITY = 0.25
DISCOUNT_FACTOR = 0.9
THRESHOLD = 1e-6

# Define the grid layout
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Action offsets
action_offsets = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1)
}

# 2 Reward Function
def get_reward(state, action):
    if state == (0, 3):    # State B
        return 5
    elif state == (0, 1):  # State A
        return 10
    
    row, col = state
    dr, dc = action_offsets[action]
    new_row, new_col = row + dr, col + dc
    
    if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
        return 0
    else:
        return -1 # Off-grid action

# 3 Transition Function
def get_next_state(state, action):
    
    if state == (0, 3):    # State B
        return (2, 3)
    
    elif state == (0, 1):  # State A
        return (4, 1)
    
    row, col = state
    
    dr, dc = action_offsets[action]
    
    new_row, new_col = row + dr, col + dc
    
    if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
        return (new_row, new_col)
    
    else:
        return state # Stay in the same state when out of bounds


# 2. Value Iteration Code
def value_iteration():
    V = np.zeros((GRID_SIZE, GRID_SIZE))   # Initialize state values to 0
    
    while True:
        V_new = np.copy(V)                 # Copy current values
        delta = 0                          # Track max change
        
        for row in range(GRID_SIZE):       # Iterate over all states
            for col in range(GRID_SIZE):
                state = (row, col)         # Current state (row, col)
                total_value = 0            # Accumulate expected return
                
                for action in ACTIONS:     # For each action
                    next_state = get_next_state(state, action) # Next state s'
                    reward = get_reward(state, action)         # Immediate reward R(s,a)
                    
                    # Bellman update
                    total_value += ACTION_PROBABILITY * (reward + 
                                   DISCOUNT_FACTOR * V[next_state])
                
                V_new[state] = total_value # Save new value
                delta = max(delta, abs(V_new[state] - V[state])) # Update max change
        
        V = V_new                          # Prepare for next iteration
        if delta < THRESHOLD:              # Check convergence
            break                          # Converged
            
    return V                               # Return final value function


# 3. Run & Output
V = value_iteration()  # Run value iteration

print("Final State Values:")
print(V)

# Print the grid of values
for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        print(f"{V[row, col]:.2f}", end="\t")
    print()
