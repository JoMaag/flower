"""Plot training results matching the original FedPG-BR paper style.

Replicates the plotting from:
https://github.com/flint-xf-fan/Byzantine-Federated-RL/blob/master/codes/agent.py

Uses RBF interpolation and 90% confidence intervals, just like the paper.
"""

import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
from scipy import stats as st


# Paper Y-axis limits per environment
ENV_LIMITS = {
    "CartPole-v1": (0, 500),
    "LunarLander-v2": (-300, 300),
    "LunarLander-v3": (-300, 300),
    "HalfCheetah-v2": (-500, 5000),
    "HalfCheetah-v5": (-500, 5000),
}


def parse_output_file(output_file):
    """Parse Flower output file to extract round-wise rewards."""
    rounds = []
    rewards = []

    with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Pattern 1: fit progress lines (server rewards during training)
    fit_pattern = r"fit progress: \((\d+), -?([\d.]+), \{'server_avg_reward': np\.float64\(([\d.]+)\)"
    for match in re.finditer(fit_pattern, content):
        round_num = int(match.group(1))
        reward = float(match.group(3))
        rounds.append(round_num)
        rewards.append(reward)

    # Pattern 2: Evaluation rewards
    eval_pattern = r"Round (\d+): Avg Reward = ([\d.]+)"
    eval_rounds = []
    eval_rewards = []
    for match in re.finditer(eval_pattern, content):
        round_num = int(match.group(1))
        reward = float(match.group(2))
        eval_rounds.append(round_num)
        eval_rewards.append(reward)

    return rounds, rewards, eval_rounds, eval_rewards


def detect_env(output_file):
    """Detect environment name from output file."""
    with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    match = re.search(r'env=(\S+)', content)
    if match:
        return match.group(1).rstrip(',')
    return "CartPole-v1"


def plot_paper_style(output_files, labels=None, save_path='learning_curve.png',
                     title=None, env_name=None):
    """Plot learning curves exactly like the original paper.

    Uses RBF interpolation and 90% confidence intervals.
    Matches plot_graph() from the original agent.py.
    """
    plt.ioff()
    fig = plt.figure(figsize=(8, 4))

    if env_name is None:
        env_name = detect_env(output_files[0])

    if title is None:
        title = f"FedPG-BR {env_name}"

    # Collect all runs
    all_rewards = []
    all_steps = []
    max_round = 0

    for output_file in output_files:
        rounds, rewards, eval_rounds, eval_rewards = parse_output_file(output_file)
        if rounds:
            all_steps.append(rounds)
            all_rewards.append(rewards)
            max_round = max(max_round, max(rounds))

    if not all_rewards:
        print(f"No data found in any output file")
        return

    # RBF interpolation across runs (paper style)
    x_interp = np.arange(max_round + 1)
    y_interp = []

    for steps, rewards in zip(all_steps, all_rewards):
        x = np.array(steps)
        y = np.array(rewards)
        try:
            rbf = Rbf(x, y, function='linear')
            y_interp.append(rbf(x_interp))
        except Exception:
            # Fallback: simple interpolation
            y_interp.append(np.interp(x_interp, x, y))

    y_interp = np.array(y_interp)
    mean = np.mean(y_interp, axis=0)

    # Plot mean line
    plt.plot(x_interp, mean, linewidth=1.5)

    # 90% confidence interval (paper style)
    if len(y_interp) > 1:
        l, h = st.norm.interval(0.90, loc=np.mean(y_interp, axis=0),
                                scale=st.sem(y_interp, axis=0))
        plt.fill_between(x_interp, l, h, alpha=0.5)
    else:
        # Single run: show raw variance via rolling std
        window = max(1, len(mean) // 20)
        rolling_std = np.array([np.std(mean[max(0, i-window):i+1])
                                for i in range(len(mean))])
        plt.fill_between(x_interp, mean - rolling_std, mean + rolling_std, alpha=0.3)

    # Y-axis limits (paper style)
    min_r, max_r = ENV_LIMITS.get(env_name, (None, None))
    if min_r is not None:
        plt.ylim([min_r, max_r])

    plt.xlabel("Number of Trajectories")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved to: {save_path}")
    plt.close()


def print_statistics(output_file, label='Experiment'):
    """Print statistics from the run."""
    rounds, rewards, eval_rounds, eval_rewards = parse_output_file(output_file)

    if not rounds:
        print(f"No data found in {output_file}")
        return

    print(f"\n{'='*60}")
    print(f"Statistics for: {label}")
    print(f"{'='*60}")
    print(f"Total Rounds: {max(rounds) if rounds else 0}")
    print(f"Initial Reward: {rewards[0]:.2f}" if rewards else "N/A")
    print(f"Final Reward: {rewards[-1]:.2f}" if rewards else "N/A")
    print(f"Max Reward: {max(rewards):.2f}" if rewards else "N/A")
    if len(rewards) >= 50:
        print(f"Mean Reward (last 50 rounds): {np.mean(rewards[-50:]):.2f}")
        print(f"Std Reward (last 50 rounds): {np.std(rewards[-50:]):.2f}")

    if eval_rewards:
        print(f"\nEvaluation Statistics:")
        print(f"  Evaluations performed: {len(eval_rewards)}")
        print(f"  Final eval reward: {eval_rewards[-1]:.2f}")
        print(f"  Max eval reward: {max(eval_rewards):.2f}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot FedPG-BR training results (paper style)')
    parser.add_argument('output_files', nargs='+', help='Output file(s) to plot')
    parser.add_argument('--labels', nargs='+', help='Labels for each file')
    parser.add_argument('--output', default='learning_curve.png', help='Output plot file')
    parser.add_argument('--title', default=None, help='Plot title')
    parser.add_argument('--env', default=None, help='Environment name (auto-detected if not set)')
    parser.add_argument('--stats', action='store_true', help='Print statistics')

    args = parser.parse_args()

    if not args.labels:
        args.labels = [f"Run {i+1}" for i in range(len(args.output_files))]

    # Print statistics
    if args.stats:
        for output_file, label in zip(args.output_files, args.labels):
            print_statistics(output_file, label)

    # Create plot (paper style)
    plot_paper_style(args.output_files, args.labels, args.output,
                     title=args.title, env_name=args.env)

    print(f"\n[OK] Done! Plot saved to: {args.output}")
