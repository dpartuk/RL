import os
import numpy as np
import matplotlib.pyplot as plt

from evaluation import EvalResult

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_training_curves(rewards_base, rewards_naive, rewards_aware, window=20):
    fig, ax = plt.subplots(figsize=(12, 5))

    for rewards, label, color, alpha in [
        (rewards_base, "Baseline (no dropout)", "green", 0.15),
        (rewards_naive, "Naive (30% dropout)", "red", 0.15),
        (rewards_aware, "Dropout-Aware (30% dropout)", "blue", 0.15),
    ]:
        ax.plot(rewards, color=color, alpha=alpha, linewidth=0.5)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed,
                color=color, label=label, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Curves (2D Figure-Eight Tracking)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_training_curves.png"), dpi=150)
    plt.close()


def plot_2d_trajectories(eval_base, eval_naive, eval_aware):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, ev, title, color in [
        (axes[0], eval_base, "Baseline", "green"),
        (axes[1], eval_naive, "Naive (30% dropout)", "red"),
        (axes[2], eval_aware, "Dropout-Aware (30% dropout)", "blue"),
    ]:
        ax.plot(ev.target_x, ev.target_y, 'g--', alpha=0.5,
                linewidth=2, label="Target path")
        ax.plot(ev.trajectories_x, ev.trajectories_y,
                color=color, linewidth=1.5, alpha=0.8, label="Agent path")

        drops_x = [ev.trajectories_x[i + 1] for i, d in enumerate(ev.drop_mask) if d]
        drops_y = [ev.trajectories_y[i + 1] for i, d in enumerate(ev.drop_mask) if d]
        if drops_x:
            ax.scatter(drops_x, drops_y, color="red", marker="x",
                       s=15, alpha=0.4, label="Packet drop")

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Agent Trajectories vs Target (2D Figure-Eight)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_trajectories_2d.png"), dpi=150)
    plt.close()


def plot_tracking_error(eval_base, eval_naive, eval_aware, steps=360):
    fig, ax = plt.subplots(figsize=(14, 5))

    for ev, label, color in [
        (eval_base, f"Baseline (mean={ev.mean_tracking_error:.3f})"
         if False else None, "green"),
        (eval_naive, None, "red"),
        (eval_aware, None, "blue"),
    ]:
        lbl = f"{['Baseline', 'Naive', 'Dropout-Aware'][['green','red','blue'].index(color)]} (mean={ev.mean_tracking_error:.3f})"
        errors = ev.per_step_errors[:steps]
        ax.plot(errors, color=color, alpha=0.7, linewidth=1, label=lbl)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Tracking Error (Euclidean distance)")
    ax.set_title("Per-Step Tracking Error (Single Eval Episode)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_tracking_error.png"), dpi=150)
    plt.close()


def plot_eval_comparison(results: dict[str, EvalResult]):
    agents = list(results.keys())
    errors = [results[a].mean_tracking_error for a in agents]
    rewards = [results[a].total_reward for a in agents]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["green", "red", "blue"]

    ax1.bar(agents, errors, color=colors, alpha=0.8)
    ax1.set_ylabel("Mean Tracking Error")
    ax1.set_title("Evaluation: Mean Tracking Error")
    for i, v in enumerate(errors):
        ax1.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(agents, rewards, color=colors, alpha=0.8)
    ax2.set_ylabel("Total Reward")
    ax2.set_title("Evaluation: Total Reward")
    for i, v in enumerate(rewards):
        ax2.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("2D Figure-Eight: Evaluation Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_eval_comparison.png"), dpi=150)
    plt.close()
