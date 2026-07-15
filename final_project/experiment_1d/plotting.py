import numpy as np
import matplotlib.pyplot as plt

from evaluation import EvalResult
from config import NUM_FEATURE_DIMS

COLORS = {
    'baseline': 'forestgreen',
    'naive': 'tomato',
    'aware': 'royalblue',
}

LABELS = {
    'baseline': 'Baseline (no dropout)',
    'naive': 'Naive (30% dropout)',
    'aware': 'Dropout-Aware (30% dropout)',
}

SMOOTH_WINDOW = 20


def plot_training_curves(rewards_baseline: list, rewards_naive: list,
                         rewards_aware: list):
    """Episode reward over training for all three agents."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for key, data in [('baseline', rewards_baseline),
                      ('naive', rewards_naive),
                      ('aware', rewards_aware)]:
        smoothed = np.convolve(data, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW,
                               mode='valid')
        ax.plot(data, alpha=0.15, color=COLORS[key])
        ax.plot(range(SMOOTH_WINDOW - 1, len(data)), smoothed,
                color=COLORS[key], linewidth=2, label=LABELS[key])

    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_tracking_error(eval_baseline: EvalResult, eval_naive: EvalResult,
                        eval_aware: EvalResult):
    """Average tracking error |agent - target| over timesteps within an episode."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for key, result in [('baseline', eval_baseline),
                        ('naive', eval_naive),
                        ('aware', eval_aware)]:
        ax.plot(result.tracking_errors, color=COLORS[key], linewidth=1.5,
                label=f"{LABELS[key]} (mean={result.mean_tracking_error:.3f})")

    ax.set_xlabel('Timestep')
    ax.set_ylabel('|agent_pos - target_pos|')
    ax.set_title('Tracking Error Over Time (averaged over eval episodes)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cumulative_rewards(eval_baseline: EvalResult, eval_naive: EvalResult,
                            eval_aware: EvalResult):
    """Cumulative reward over timesteps within an episode."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for key, result in [('baseline', eval_baseline),
                        ('naive', eval_naive),
                        ('aware', eval_aware)]:
        ax.plot(result.cumulative_rewards, color=COLORS[key], linewidth=1.5,
                label=f"{LABELS[key]} (total={result.total_reward:.1f})")

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward Over Time (averaged over eval episodes)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_trajectories(eval_baseline: EvalResult, eval_naive: EvalResult,
                      eval_aware: EvalResult):
    """Position traces: target + all three agents, with dropout markers."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 3, 1]})

    timesteps = np.arange(len(eval_baseline.target_trajectory))
    target = eval_baseline.target_trajectory

    agent_data = [
        ('baseline', eval_baseline),
        ('naive', eval_naive),
        ('aware', eval_aware),
    ]

    for i, (key, result) in enumerate(agent_data):
        ax = axes[i]
        ax.plot(timesteps, target, color='gray', linewidth=1,
                linestyle='--', label='Target')
        ax.plot(timesteps, result.agent_trajectory, color=COLORS[key],
                linewidth=1.5, label=LABELS[key])

        if result.dropout_mask is not None and result.dropout_mask.any():
            drop_times = np.where(result.dropout_mask)[0] + 1
            ax.scatter(drop_times,
                       result.agent_trajectory[drop_times],
                       color='red', marker='x', s=15, alpha=0.5,
                       zorder=5, label='Packet drop')

        ax.set_ylabel('Position')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

    # Bottom panel: dropout mask as binary strip
    ax_drop = axes[3]
    if eval_naive.dropout_mask is not None:
        drop_times = np.arange(1, len(eval_naive.dropout_mask) + 1)
        ax_drop.fill_between(drop_times, 0, eval_naive.dropout_mask.astype(float),
                             color='red', alpha=0.4, step='mid')
    ax_drop.set_ylabel('Dropped')
    ax_drop.set_xlabel('Timestep')
    ax_drop.set_yticks([0, 1])
    ax_drop.set_yticklabels(['OK', 'Drop'])
    ax_drop.grid(alpha=0.3)

    fig.suptitle('Agent Trajectories vs Target (single eval episode)', fontsize=13)
    fig.tight_layout()
    return fig


FEATURE_NAMES = ['Bullseye', 'Linear', 'Quadratic', 'Danger', 'Bias']
FEATURE_COLORS = ['gold', 'steelblue', 'coral', 'mediumpurple', 'gray']


def plot_irl_convergence(weight_history: np.ndarray):
    """IRL weight convergence over iterations."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(NUM_FEATURE_DIMS):
        ax.plot(weight_history[:, i], linewidth=1.5,
                color=FEATURE_COLORS[i], label=FEATURE_NAMES[i])

    ax.set_xlabel('IRL Iteration')
    ax.set_ylabel('Weight Value')
    ax.set_title('IRL Reward Weight Convergence')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_irl_comparison(eval_handcoded: EvalResult, eval_irl: EvalResult):
    """Compare tracking error: hand-coded reward vs IRL-learned reward."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(eval_handcoded.tracking_errors, color=COLORS['baseline'],
            linewidth=1.5,
            label=f"Hand-coded reward (mean={eval_handcoded.mean_tracking_error:.3f})")
    ax.plot(eval_irl.tracking_errors, color='darkorange', linewidth=1.5,
            linestyle='--',
            label=f"IRL-learned reward (mean={eval_irl.mean_tracking_error:.3f})")

    ax.set_xlabel('Timestep')
    ax.set_ylabel('|agent_pos - target_pos|')
    ax.set_title('Tracking Error: Hand-Coded vs IRL-Learned Reward')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
