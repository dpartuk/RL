import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Implement the Environment ---
def get_reward_and_cost(arm):
    # Parameters for each arm
    reward_means = [0.8, 0.6, 0.9, 0.4, 0.7]
    reward_stds = [0.1, 0.1, 0.1, 0.1, 0.1]
    costs = [0.2, 0.1, 0.3, 0.05, 0.15]
    
    # Sample reward from the normal distribution
    reward = np.random.normal(reward_means[arm], reward_stds[arm])
    # Return reward and cost
    return reward, costs[arm]

# --- Step 2: Implement the Epsilon-Greedy Algorithm ---
def epsilon_greedy_with_costs(num_arms, num_steps, epsilon):
    # Initialize Q-values and counts for each arm
    q_values = np.zeros(num_arms)
    arm_counts = np.zeros(num_arms)
    cumulative_net_reward = 0
    history = []

    for step in range(num_steps):
        # Exploration vs Exploitation selection logic
        if np.random.rand() < epsilon:
            arm = np.random.randint(num_arms)  # Explore random arm
        else:
            arm = np.argmax(q_values)  # Exploit best known arm

        # Interact with the environment
        reward, cost = get_reward_and_cost(arm)
        net_reward = reward - cost  # Calculate net reward (R_i - C_i)
        
        # Incremental sample-average update rule
        arm_counts[arm] += 1
        q_values[arm] += (net_reward - q_values[arm]) / arm_counts[arm]
        
        # Track cumulative performance
        cumulative_net_reward += net_reward
        history.append(cumulative_net_reward)
        
    return history, q_values, arm_counts

# --- Step 3: Run the Simulation Execution Section ---
if __name__ == "__main__":
    # Define Parameters
    num_arms = 5 [cite: 40]
    num_steps = 1000 [cite: 41]
    epsilon = 0.1 [cite: 42]
    
    print(Running epsilon-greedy simulation with epsilon = {epsilon} over {num_steps} steps...)
    
    # Run the algorithm
    cumulative_net_rewards, final_q_values, final_counts = epsilon_greedy_with_costs(
        num_arms, num_steps, epsilon
    ) [cite: 44]
    
    # Print basic summary metrics to the console
    print(\n--- Simulation Summary ---)
    print(fFinal Cumulative Net Reward: {cumulative_net_rewards[-1]:.2f})
    for i in range(num_arms):
        print(fArm {i} -> Pulled: {int(final_counts[i]):3d} times | Estimated Q-Value (Net): {final_q_values[i]:.4f})
        
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_net_rewards, color='royalblue', linewidth=2) [cite: 46]
    plt.xlabel("Steps") [cite: 47]
    plt.ylabel("Cumulative Net Reward") [cite: 48]
    plt.title("$\epsilon$-Greedy Algorithm with Costs") [cite: 49, 51]
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show() [cite: 50]
