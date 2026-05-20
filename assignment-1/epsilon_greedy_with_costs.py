import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Implement the Environment ---
REWARD_MEANS = [0.8, 0.6, 0.9, 0.4, 0.7]
REWARD_STDS = [0.1, 0.1, 0.1, 0.1, 0.1]
INITIAL_COSTS = [0.2, 0.1, 0.3, 0.05, 0.15]

def get_reward_and_cost(arm, costs=None):
    if costs is None:
        costs = INITIAL_COSTS
    reward = np.random.normal(REWARD_MEANS[arm], REWARD_STDS[arm])
    return reward, costs[arm]

def drift_costs(costs, drift_std=0.05):
    new_costs = []
    for c in costs:
        new_c = c + np.random.normal(0, drift_std)
        new_costs.append(np.clip(new_c, 0.01, 0.95))
    return new_costs

# --- Step 2a: Original Epsilon-Greedy (sample-average) ---
def epsilon_greedy_with_costs(num_arms, num_steps, epsilon):
    q_values = np.zeros(num_arms)
    arm_counts = np.zeros(num_arms)
    cumulative_net_reward = 0
    history = []

    for step in range(num_steps):
        if np.random.rand() < epsilon:
            arm = np.random.randint(num_arms)
        else:
            arm = np.argmax(q_values)

        reward, cost = get_reward_and_cost(arm)
        net_reward = reward - cost

        arm_counts[arm] += 1
        q_values[arm] += (net_reward - q_values[arm]) / arm_counts[arm]

        cumulative_net_reward += net_reward
        history.append(cumulative_net_reward)

    return history, q_values, arm_counts

# --- Step 2b: Non-Stationary Epsilon-Greedy (constant step-size) ---
def epsilon_greedy_nonstationary(num_arms, num_steps, epsilon, alpha=0.1, drift_interval=100):
    q_values = np.zeros(num_arms)
    arm_counts = np.zeros(num_arms)
    cumulative_net_reward = 0
    history = []
    costs = list(INITIAL_COSTS)
    cost_history = [list(costs)]

    for step in range(num_steps):
        if step > 0 and step % drift_interval == 0:
            costs = drift_costs(costs)
            cost_history.append(list(costs))

        if np.random.rand() < epsilon:
            arm = np.random.randint(num_arms)
        else:
            arm = np.argmax(q_values)

        reward, cost = get_reward_and_cost(arm, costs)
        net_reward = reward - cost

        arm_counts[arm] += 1
        # Constant step-size alpha gives more weight to recent observations
        q_values[arm] += alpha * (net_reward - q_values[arm])

        cumulative_net_reward += net_reward
        history.append(cumulative_net_reward)

    return history, q_values, arm_counts, cost_history

# --- Step 3: Run the Simulation Execution Section ---
if __name__ == "__main__":
    num_arms = 5
    num_steps = 1000
    epsilon = 0.1
    alpha = 0.1
    drift_interval = 100

    print(f"Non-stationary environment: costs drift every {drift_interval} steps")
    print(f"Epsilon = {epsilon}, Alpha (constant step-size) = {alpha}\n")

    # Use the same seed for fair comparison
    np.random.seed(42)
    hist_adaptive, q_adaptive, counts_adaptive, cost_hist = epsilon_greedy_nonstationary(
        num_arms, num_steps, epsilon, alpha, drift_interval
    )

    np.random.seed(42)
    hist_sample_avg, q_sample_avg, counts_sample_avg = epsilon_greedy_with_costs(
        num_arms, num_steps, epsilon
    )

    print("--- Constant Step-Size (alpha=0.1) — adapts to non-stationarity ---")
    print(f"Final Cumulative Net Reward: {hist_adaptive[-1]:.2f}")
    for i in range(num_arms):
        print(f"  Arm {i} -> Pulled: {int(counts_adaptive[i]):3d} times | Q-Value: {q_adaptive[i]:.4f}")

    print("\n--- Sample Average — does NOT adapt to non-stationarity ---")
    print(f"Final Cumulative Net Reward: {hist_sample_avg[-1]:.2f}")
    for i in range(num_arms):
        print(f"  Arm {i} -> Pulled: {int(counts_sample_avg[i]):3d} times | Q-Value: {q_sample_avg[i]:.4f}")

    print(f"\nCost drift history (every {drift_interval} steps):")
    for idx, c in enumerate(cost_hist):
        step = idx * drift_interval
        print(f"  Step {step:4d}: {[f'{x:.3f}' for x in c]}")

    plt.figure(figsize=(10, 6))
    plt.plot(hist_adaptive, color='royalblue', linewidth=2, label=f'Constant step-size (α={alpha})')
    plt.plot(hist_sample_avg, color='tomato', linewidth=2, linestyle='--', label='Sample average')
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Net Reward")
    plt.title(r"$\epsilon$-Greedy: Non-Stationary Costs (drift every 100 steps)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
