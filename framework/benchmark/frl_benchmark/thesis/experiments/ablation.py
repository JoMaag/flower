"""Ablation study runner for FRL Benchmark.

Three ablation axes:
  1. Byzantine ratio   — vary B/K with FedPG-BR (sign-flip attack)
  2. Number of workers — vary K with FedPG-BR (no Byzantine)
  3. Variance reduction — GOMDP vs SVRPG vs FedPG-BR (no Byzantine)

Results are logged to TensorBoard (runs/) and a JSON summary is written to
experiments/results/ablation_log.json.

Usage
-----
python experiments/ablation.py [--quick] [--axis byz|workers|variance|all]
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

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "experiments" / "results" / "ablation_log.json"


# ── Ablation axes ─────────────────────────────────────────────────────────────

BYZ_RATIO_AXIS: List[Dict] = [
    {"label": "byz=0/10 (0%)",  "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "10", "FRL_BYZANTINE": "0"},
    {"label": "byz=1/10 (10%)", "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "10", "FRL_BYZANTINE": "1", "FRL_ATTACK": "sign-flip"},
    {"label": "byz=2/10 (20%)", "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "10", "FRL_BYZANTINE": "2", "FRL_ATTACK": "sign-flip"},
    {"label": "byz=3/10 (30%)", "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "10", "FRL_BYZANTINE": "3", "FRL_ATTACK": "sign-flip"},
    {"label": "byz=4/10 (40%)", "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "10", "FRL_BYZANTINE": "4", "FRL_ATTACK": "sign-flip"},
]

WORKERS_AXIS: List[Dict] = [
    {"label": "K=5",  "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "5",  "FRL_BYZANTINE": "0"},
    {"label": "K=10", "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "10", "FRL_BYZANTINE": "0"},
    {"label": "K=15", "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "15", "FRL_BYZANTINE": "0"},
    {"label": "K=20", "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "20", "FRL_BYZANTINE": "0"},
]

VARIANCE_AXIS: List[Dict] = [
    {"label": "GOMDP (no VR, no filter)",       "FRL_METHOD": "gomdp",    "FRL_WORKERS": "10", "FRL_BYZANTINE": "0"},
    {"label": "SVRPG (VR only, no filter)",     "FRL_METHOD": "svrpg",    "FRL_WORKERS": "10", "FRL_BYZANTINE": "0"},
    {"label": "FedPG-BR (filter + VR)",         "FRL_METHOD": "fedpg-br", "FRL_WORKERS": "10", "FRL_BYZANTINE": "0"},
]


# ── Runner (same as run_experiments.py) ───────────────────────────────────────

def run_one(experiment: Dict, env_name: str, rounds: int, round_timeout: int = 600) -> Dict:
    flower_sim = shutil.which("flower-simulation")
    if not flower_sim:
        sys.exit("ERROR: flower-simulation not found.")

    label    = experiment["label"]
    method   = experiment.get("FRL_METHOD", "fedpg-br")
    workers  = experiment.get("FRL_WORKERS", "10")
    byzantine = experiment.get("FRL_BYZANTINE", "0")
    attack   = experiment.get("FRL_ATTACK", "random-noise")

    cfg_parts = [
        f'env="{env_name}"',
        f'method="{method}"',
        f"num-server-rounds={rounds}",
        f"num-workers={workers}",
        f"num-byzantine={byzantine}",
        f'attack-type="{attack}"',
        f"round-timeout={round_timeout}",
    ]

    cmd = [
        flower_sim,
        "--app", str(ROOT),
        "--num-supernodes", workers,
        "--run-config", " ".join(cfg_parts),
    ]

    tmp_home = tempfile.mkdtemp(prefix="flwr-abl-")
    env = os.environ.copy()
    env["HOME"] = tmp_home
    env.setdefault("DASHBOARD_URL", "")

    print(f"\n{'='*60}")
    print(f"  ABLATION: {label}")
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
    parser = argparse.ArgumentParser(description="FRL Benchmark Ablation Study")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="50 rounds (smoke test)")
    parser.add_argument("--axis", choices=["byz", "workers", "variance", "all"], default="all")
    parser.add_argument("--round-timeout", type=int, default=600)
    args = parser.parse_args()

    FULL_ROUNDS = {"CartPole-v1": 312, "LunarLander-v3": 323}
    default_rounds = FULL_ROUNDS.get(args.env, 150)
    rounds = args.rounds if args.rounds > 0 else (50 if args.quick else default_rounds)

    axes: Dict[str, List[Dict]] = {
        "byz": BYZ_RATIO_AXIS,
        "workers": WORKERS_AXIS,
        "variance": VARIANCE_AXIS,
        "all": BYZ_RATIO_AXIS + WORKERS_AXIS + VARIANCE_AXIS,
    }
    experiments = axes[args.axis]

    print(f"\nFRL Ablation Study")
    print(f"  Axis:   {args.axis}  ({len(experiments)} runs)")
    print(f"  Env:    {args.env}")
    print(f"  Rounds: {rounds}{'  [QUICK]' if args.quick else ''}\n")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_meta = []

    for exp in experiments:
        meta = run_one(exp, env_name=args.env, rounds=rounds, round_timeout=args.round_timeout)
        all_meta.append(meta)
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_meta, f, indent=2)

    print(f"\nAblation complete. TensorBoard: tensorboard --logdir {ROOT / 'runs'}")


if __name__ == "__main__":
    main()
