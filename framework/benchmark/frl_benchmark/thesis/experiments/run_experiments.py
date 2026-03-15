"""Empirical experiment runner for FRL Benchmark.

Runs the four-method comparison (Independent, FedPG, AFedPG, Centralized)
and the Byzantine robustness suite.  Each run is launched via
``flower-simulation`` and results are logged to TensorBoard (runs/).

Usage
-----
# Full comparison (312 rounds per method, ~10 h total)
python experiments/run_experiments.py

# Quick smoke-test (50 rounds per method, ~1 h total)
python experiments/run_experiments.py --quick

# Only Byzantine robustness suite
python experiments/run_experiments.py --suite byzantine

# Only the ablation study (use experiments/ablation.py instead)
python experiments/run_experiments.py --suite ablation
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).parent.parent          # repo root
RUNS_DIR = ROOT / "runs"
RESULTS_FILE = ROOT / "experiments" / "results" / "experiment_log.json"


# ── Experiment definitions ────────────────────────────────────────────────────

# Each entry: name → env-var overrides passed to run_training.py
COMPARISON_SUITE: List[Dict] = [
    {
        "label": "Independent",
        "FRL_METHOD": "independent",
        "FRL_WORKERS": "10",
        "FRL_BYZANTINE": "0",
    },
    {
        "label": "FedPG",
        "FRL_METHOD": "gomdp",
        "FRL_WORKERS": "10",
        "FRL_BYZANTINE": "0",
    },
    {
        "label": "SVRPG",
        "FRL_METHOD": "svrpg",
        "FRL_WORKERS": "10",
        "FRL_BYZANTINE": "0",
    },
    {
        "label": "FedPG-BR (AFedPG)",
        "FRL_METHOD": "fedpg-br",
        "FRL_WORKERS": "10",
        "FRL_BYZANTINE": "0",
    },
    {
        "label": "Centralized",
        "FRL_METHOD": "centralized",
        "FRL_WORKERS": "1",
        "FRL_BYZANTINE": "0",
        "FRL_BATCH_SIZE": "160",   # 10 × 16 = same total trajectories as federated
    },
]

BYZANTINE_SUITE: List[Dict] = [
    {
        "label": "FedPG (30% Byzantine, sign-flip)",
        "FRL_METHOD": "gomdp",
        "FRL_WORKERS": "10",
        "FRL_BYZANTINE": "3",
        "FRL_ATTACK": "sign-flip",
    },
    {
        "label": "SVRPG (30% Byzantine, sign-flip)",
        "FRL_METHOD": "svrpg",
        "FRL_WORKERS": "10",
        "FRL_BYZANTINE": "3",
        "FRL_ATTACK": "sign-flip",
    },
    {
        "label": "FedPG-BR (30% Byzantine, sign-flip)",
        "FRL_METHOD": "fedpg-br",
        "FRL_WORKERS": "10",
        "FRL_BYZANTINE": "3",
        "FRL_ATTACK": "sign-flip",
    },
    # Random-noise attacks
    {
        "label": "FedPG (30% Byzantine, random-noise)",
        "FRL_METHOD": "gomdp",
        "FRL_WORKERS": "10",
        "FRL_BYZANTINE": "3",
        "FRL_ATTACK": "random-noise",
    },
    {
        "label": "FedPG-BR (30% Byzantine, random-noise)",
        "FRL_METHOD": "fedpg-br",
        "FRL_WORKERS": "10",
        "FRL_BYZANTINE": "3",
        "FRL_ATTACK": "random-noise",
    },
]

ABLATION_SUITE: List[Dict] = [
    # Vary Byzantine ratio (FedPG-BR, sign-flip attack)
    {"label": "FedPG-BR byz=0/10", "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "10", "FRL_BYZANTINE": "0"},
    {"label": "FedPG-BR byz=1/10", "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "10", "FRL_BYZANTINE": "1", "FRL_ATTACK": "sign-flip"},
    {"label": "FedPG-BR byz=2/10", "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "10", "FRL_BYZANTINE": "2", "FRL_ATTACK": "sign-flip"},
    {"label": "FedPG-BR byz=3/10", "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "10", "FRL_BYZANTINE": "3", "FRL_ATTACK": "sign-flip"},
    # Vary number of workers (FedPG-BR, no Byzantine)
    {"label": "FedPG-BR K=5",  "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "5",  "FRL_BYZANTINE": "0"},
    {"label": "FedPG-BR K=10", "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "10", "FRL_BYZANTINE": "0"},
    {"label": "FedPG-BR K=20", "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "20", "FRL_BYZANTINE": "0"},
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_one(experiment: Dict, env_name: str, rounds: int, round_timeout: int = 600) -> Dict:
    """Launch a single flower-simulation run; return metadata."""
    flower_sim = shutil.which("flower-simulation")
    if not flower_sim:
        sys.exit("ERROR: flower-simulation not found. Install flwr[simulation]==1.20.0")

    label = experiment["label"]
    method = experiment.get("FRL_METHOD", "fedpg-br")
    workers = experiment.get("FRL_WORKERS", "10")
    byzantine = experiment.get("FRL_BYZANTINE", "0")
    attack = experiment.get("FRL_ATTACK", "random-noise")
    batch_size = experiment.get("FRL_BATCH_SIZE", "0")

    # Build run-config
    cfg_parts = [
        f'env="{env_name}"',
        f'method="{method}"',
        f"num-server-rounds={rounds}",
        f"num-workers={workers}",
        f"num-byzantine={byzantine}",
        f'attack-type="{attack}"',
        f"round-timeout={round_timeout}",
    ]
    if int(batch_size) > 0:
        cfg_parts += [f"batch-size={batch_size}", f"batch-size-min={batch_size}", f"batch-size-max={batch_size}"]

    cmd = [
        flower_sim,
        "--app", str(ROOT),
        "--num-supernodes", workers,
        "--run-config", " ".join(cfg_parts),
    ]

    # Fresh HOME dir to avoid SQLite conflicts between sequential runs
    tmp_home = tempfile.mkdtemp(prefix="flwr-exp-")
    env = os.environ.copy()
    env["HOME"] = tmp_home
    # Tell server_app not to start a dashboard server
    env.setdefault("DASHBOARD_URL", "")

    print(f"\n{'='*60}")
    print(f"  STARTING: {label}")
    print(f"  env={env_name}  method={method}  K={workers}  byz={byzantine}  rounds={rounds}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    shutil.rmtree(tmp_home, ignore_errors=True)

    meta = {
        "label": label,
        "env": env_name,
        "method": method,
        "workers": int(workers),
        "byzantine": int(byzantine),
        "attack": attack,
        "rounds": rounds,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"  → {status}  ({elapsed/60:.1f} min)", flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description="FRL Benchmark Experiment Runner")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium environment name")
    parser.add_argument("--rounds", type=int, default=0,
                        help="Override number of rounds (0 = use --quick or full default)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 50 rounds per experiment (smoke test)")
    parser.add_argument("--suite", choices=["comparison", "byzantine", "ablation", "all"],
                        default="comparison", help="Which experiment suite to run")
    parser.add_argument("--round-timeout", type=int, default=600,
                        help="Per-round timeout in seconds (default 600)")
    args = parser.parse_args()

    # Determine round count
    FULL_ROUNDS = {"CartPole-v1": 312, "LunarLander-v3": 323, "HalfCheetah-v5": 208}
    default_rounds = FULL_ROUNDS.get(args.env, 150)
    rounds = args.rounds if args.rounds > 0 else (50 if args.quick else default_rounds)

    # Select suites
    suites: Dict[str, List[Dict]] = {
        "comparison": COMPARISON_SUITE,
        "byzantine": BYZANTINE_SUITE,
        "ablation": ABLATION_SUITE,
        "all": COMPARISON_SUITE + BYZANTINE_SUITE + ABLATION_SUITE,
    }
    experiments = suites[args.suite]

    print(f"\nFRL Benchmark Experiment Runner")
    print(f"  Suite:  {args.suite}  ({len(experiments)} experiments)")
    print(f"  Env:    {args.env}")
    print(f"  Rounds: {rounds}{'  [QUICK MODE]' if args.quick else ''}")
    print(f"  Output: {RUNS_DIR}/\n")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_meta = []

    for exp in experiments:
        meta = run_one(exp, env_name=args.env, rounds=rounds, round_timeout=args.round_timeout)
        all_meta.append(meta)
        # Save incrementally so partial results survive a crash
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All {len(experiments)} experiments complete.")
    print(f"Results log: {RESULTS_FILE}")
    print(f"TensorBoard: tensorboard --logdir {RUNS_DIR}")
    ok = sum(1 for m in all_meta if m["returncode"] == 0)
    print(f"Succeeded: {ok}/{len(experiments)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
