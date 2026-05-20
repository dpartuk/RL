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

def drift_rewards(means, drift_std=0.1):
    new_means = []
    for m in means:
        new_m = m + np.random.normal(0, drift_std)
        new_means.append(np.clip(new_m, 0.0, 1.5))
    return new_means

def get_reward_and_cost_dynamic(arm, reward_means, costs):
    reward = np.random.normal(reward_means[arm], REWARD_STDS[arm])
    return reward, costs[arm]

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

# --- Step 2c: Fully Dynamic Epsilon-Greedy (rewards + costs drift) ---
def epsilon_greedy_dynamic(num_arms, num_steps, epsilon, alpha=0.1, drift_interval=100):
    q_values = np.zeros(num_arms)
    arm_counts = np.zeros(num_arms)
    cumulative_net_reward = 0
    history = []
    costs = list(INITIAL_COSTS)
    reward_means = list(REWARD_MEANS)
    env_history = [{"step": 0, "means": list(reward_means), "costs": list(costs)}]

    for step in range(num_steps):
        if step > 0 and step % drift_interval == 0:
            reward_means = drift_rewards(reward_means)
            costs = drift_costs(costs)
            env_history.append({"step": step, "means": list(reward_means), "costs": list(costs)})

        if np.random.rand() < epsilon:
            arm = np.random.randint(num_arms)
        else:
            arm = np.argmax(q_values)

        reward, cost = get_reward_and_cost_dynamic(arm, reward_means, costs)
        net_reward = reward - cost

        arm_counts[arm] += 1
        q_values[arm] += alpha * (net_reward - q_values[arm])

        cumulative_net_reward += net_reward
        history.append(cumulative_net_reward)

    return history, q_values, arm_counts, env_history

def epsilon_greedy_dynamic_sample_avg(num_arms, num_steps, epsilon, drift_interval=100):
    q_values = np.zeros(num_arms)
    arm_counts = np.zeros(num_arms)
    cumulative_net_reward = 0
    history = []
    costs = list(INITIAL_COSTS)
    reward_means = list(REWARD_MEANS)

    for step in range(num_steps):
        if step > 0 and step % drift_interval == 0:
            reward_means = drift_rewards(reward_means)
            costs = drift_costs(costs)

        if np.random.rand() < epsilon:
            arm = np.random.randint(num_arms)
        else:
            arm = np.argmax(q_values)

        reward, cost = get_reward_and_cost_dynamic(arm, reward_means, costs)
        net_reward = reward - cost

        arm_counts[arm] += 1
        q_values[arm] += (net_reward - q_values[arm]) / arm_counts[arm]

        cumulative_net_reward += net_reward
        history.append(cumulative_net_reward)

    return history, q_values, arm_counts

# --- Step 2d: UCB Algorithm with Costs ---
def ucb_with_costs(num_arms, num_steps, c=2.0):
    q_values = np.zeros(num_arms)
    arm_counts = np.zeros(num_arms)
    cumulative_net_reward = 0
    history = []

    for step in range(num_steps):
        if step < num_arms:
            arm = step
        else:
            ucb_values = q_values + c * np.sqrt(np.log(step) / arm_counts)
            arm = np.argmax(ucb_values)

        reward, cost = get_reward_and_cost(arm)
        net_reward = reward - cost

        arm_counts[arm] += 1
        q_values[arm] += (net_reward - q_values[arm]) / arm_counts[arm]

        cumulative_net_reward += net_reward
        history.append(cumulative_net_reward)

    return history, q_values, arm_counts

# --- Step 3: Run the Simulation Execution Section ---
if __name__ == "__main__":
    num_arms = 5
    num_steps = 1000
    epsilon = 0.1
    alpha = 0.1
    drift_interval = 100

    print(f"Dynamic environment: rewards AND costs drift every {drift_interval} steps")
    print(f"Epsilon = {epsilon}, Alpha = {alpha}\n")

    np.random.seed(42)
    hist_adaptive, q_adaptive, counts_adaptive, env_hist = epsilon_greedy_dynamic(
        num_arms, num_steps, epsilon, alpha, drift_interval
    )

    np.random.seed(42)
    hist_sa, q_sa, counts_sa = epsilon_greedy_dynamic_sample_avg(
        num_arms, num_steps, epsilon, drift_interval
    )

    print(f"--- Constant Step-Size (α={alpha}) — adapts to dynamic rewards ---")
    print(f"Final Cumulative Net Reward: {hist_adaptive[-1]:.2f}")
    for i in range(num_arms):
        print(f"  Arm {i} -> Pulled: {int(counts_adaptive[i]):3d} times | Q-Value: {q_adaptive[i]:.4f}")

    print(f"\n--- Sample Average — does NOT adapt ---")
    print(f"Final Cumulative Net Reward: {hist_sa[-1]:.2f}")
    for i in range(num_arms):
        print(f"  Arm {i} -> Pulled: {int(counts_sa[i]):3d} times | Q-Value: {q_sa[i]:.4f}")

    print(f"\nDifference: {hist_adaptive[-1] - hist_sa[-1]:.2f} (adaptive - sample avg)")

    print(f"\nEnvironment drift history (every {drift_interval} steps):")
    print(f"  {'Step':>4s}  {'Net Rewards (mean - cost) per arm':s}")
    for e in env_hist:
        nets = [e['means'][i] - e['costs'][i] for i in range(num_arms)]
        best = np.argmax(nets)
        net_str = "  ".join(f"{n:.3f}" for n in nets)
        print(f"  {e['step']:4d}  [{net_str}]  best=Arm {best}")

    plt.figure(figsize=(10, 6))
    plt.plot(hist_adaptive, color='royalblue', linewidth=2, label=f'Constant step-size (α={alpha})')
    plt.plot(hist_sa, color='tomato', linewidth=2, linestyle='--', label='Sample average')
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Net Reward")
    plt.title(r"$\epsilon$-Greedy: Dynamic Rewards & Costs (drift every 100 steps)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
