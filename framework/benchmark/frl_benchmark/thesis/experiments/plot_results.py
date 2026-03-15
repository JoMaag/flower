"""Plot comparison results from TensorBoard runs/.

Reads ``eval/server_avg_reward`` from TensorBoard event files, applies
exponential smoothing, and generates publication-quality comparison figures.

Usage
-----
# Compare all methods (reads runs/ directory)
python experiments/plot_results.py

# Only show specific runs (substring match on run directory name)
python experiments/plot_results.py --filter "gomdp,fedpg-br,independent,centralized"

# Save figures to disk
python experiments/plot_results.py --save experiments/results/figures/

# Smooth coefficient (0 = raw, 0.9 = heavy smoothing)
python experiments/plot_results.py --smooth 0.85
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

ROOT = Path(__file__).parent.parent
RUNS_DIR = ROOT / "runs"

# ── Colours & labels ─────────────────────────────────────────────────────────

# Map method prefix → display name + colour (matplotlib names)
METHOD_STYLE: Dict[str, Tuple[str, str]] = {
    "independent":  ("Independent",    "#6366f1"),   # indigo
    "gomdp":        ("FedPG",          "#f59e0b"),   # amber
    "svrpg":        ("SVRPG",          "#10b981"),   # emerald
    "fedpg-br":     ("FedPG-BR (AFedPG)", "#ef4444"),  # red
    "centralized":  ("Centralized",    "#64748b"),   # slate
}


def _smooth(values: List[float], alpha: float) -> np.ndarray:
    """Exponential moving average smoothing."""
    if alpha <= 0:
        return np.array(values)
    out = np.empty(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out


def read_tb_scalar(event_dir: Path, tag: str) -> Tuple[List[int], List[float]]:
    """Read a scalar tag from a TensorBoard event directory."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError("tensorboard must be installed: pip install tensorboard")

    ea = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    steps  = [e.step  for e in events]
    values = [e.value for e in events]
    return steps, values


def find_runs(runs_dir: Path, filters: Optional[List[str]] = None) -> List[Path]:
    """Return sorted list of run directories matching filters."""
    if not runs_dir.exists():
        return []
    dirs = [d for d in sorted(runs_dir.iterdir()) if d.is_dir()]
    if filters:
        dirs = [d for d in dirs if any(f in d.name for f in filters)]
    return dirs


def _method_from_name(name: str) -> str:
    """Infer method key from run directory name (format: method__env__ts)."""
    return name.split("__")[0] if "__" in name else name


def plot_comparison(
    runs_dir: Path,
    filters: Optional[List[str]],
    smooth: float,
    tag: str = "eval/server_avg_reward",
    title: str = "Method Comparison — CartPole-v1",
    save_path: Optional[Path] = None,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    run_dirs = find_runs(runs_dir, filters)
    if not run_dirs:
        print(f"No runs found in {runs_dir}" + (f" matching {filters}" if filters else ""))
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0

    for run_dir in run_dirs:
        steps, values = read_tb_scalar(run_dir, tag)
        if not values:
            continue

        method = _method_from_name(run_dir.name)
        style = METHOD_STYLE.get(method, (run_dir.name, "#334155"))
        label_str, color = style

        # Include Byzantine info in label if present
        m = re.search(r"byz(\d+)of(\d+)", run_dir.name)
        if m:
            label_str += f" (byz {m.group(1)}/{m.group(2)})"

        raw   = np.array(values)
        sm    = _smooth(values, smooth)

        ax.plot(steps, raw, color=color, alpha=0.2, linewidth=0.8)
        ax.plot(steps, sm,  color=color, linewidth=2, label=label_str)
        plotted += 1

    if plotted == 0:
        print(f"No scalar data found for tag '{tag}'.")
        return

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Average Return", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_ablation_bar(
    runs_dir: Path,
    axis: str,             # "byz" | "workers"
    env: str = "CartPole-v1",
    smooth: float = 0.85,
    tag: str = "eval/server_avg_reward",
    save_path: Optional[Path] = None,
) -> None:
    """Bar chart of final reward for a single ablation axis."""
    import matplotlib.pyplot as plt

    # Pattern to identify runs belonging to this axis
    if axis == "byz":
        pattern = "fedpg-br"
        x_label = "Byzantine Ratio (B/K)"
        title = f"Ablation: Byzantine Ratio — FedPG-BR — {env}"
        def x_val(name):
            m = re.search(r"byz-?(\d+)of(\d+)", name) or re.search(r"(\d+)byz", name)
            return int(m.group(1)) / int(m.group(2)) if m and m.lastindex == 2 else None
    elif axis == "workers":
        pattern = "fedpg-br"
        x_label = "Number of Workers (K)"
        title = f"Ablation: Workers — FedPG-BR — {env}"
        def x_val(name):
            m = re.search(r"k(\d+)", name, re.IGNORECASE) or re.search(r"workers?(\d+)", name)
            return int(m.group(1)) if m else None
    else:
        raise ValueError(f"Unknown axis: {axis}")

    run_dirs = find_runs(runs_dir, [pattern, env.lower().replace("-", "")])
    if not run_dirs:
        print(f"No runs found for axis={axis}")
        return

    xs, ys, labels = [], [], []
    for rd in run_dirs:
        xv = x_val(rd.name)
        if xv is None:
            continue
        _, values = read_tb_scalar(rd, tag)
        if not values:
            continue
        sm = _smooth(values, smooth)
        final = float(np.mean(sm[-10:]))   # mean of last 10 eval points
        xs.append(xv)
        ys.append(final)
        labels.append(str(xv))

    if not xs:
        print("No data found.")
        return

    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(xs)), ys, color="#ef4444", alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Final Avg Return (last 10 evals)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, y + 1, f"{y:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot FRL Benchmark Results")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--filter", default="", help="Comma-separated substrings to filter run dirs")
    parser.add_argument("--smooth", type=float, default=0.85, help="EMA smoothing (0=raw, 0.95=heavy)")
    parser.add_argument("--tag", default="eval/server_avg_reward", help="TensorBoard scalar tag")
    parser.add_argument("--title", default="FRL Method Comparison")
    parser.add_argument("--save", default="", help="Directory to save figures (empty = display)")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--plot", choices=["comparison", "byz-ablation", "workers-ablation", "all"],
                        default="comparison")
    args = parser.parse_args()

    runs_dir  = Path(args.runs_dir)
    filters   = [f.strip() for f in args.filter.split(",") if f.strip()] or None
    save_dir  = Path(args.save) if args.save else None

    if args.plot in ("comparison", "all"):
        plot_comparison(
            runs_dir, filters, args.smooth, args.tag, args.title,
            save_path=(save_dir / "comparison.png") if save_dir else None,
        )

    if args.plot in ("byz-ablation", "all"):
        plot_ablation_bar(
            runs_dir, axis="byz", env=args.env, smooth=args.smooth,
            save_path=(save_dir / "ablation_byz.png") if save_dir else None,
        )

    if args.plot in ("workers-ablation", "all"):
        plot_ablation_bar(
            runs_dir, axis="workers", env=args.env, smooth=args.smooth,
            save_path=(save_dir / "ablation_workers.png") if save_dir else None,
        )


if __name__ == "__main__":
    main()
