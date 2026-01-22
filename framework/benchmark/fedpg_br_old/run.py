#!/usr/bin/env python3
"""
FedPG-BR: Fault-Tolerant Federated Reinforcement Learning

Main entry point for training with Flower simulation.

Usage:
    python -m fedpg_br.run --env CartPole-v1 --num_workers 10 --num_byzantine 3
"""

import warnings
# Suppress all deprecation and dependency warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*DEPRECATED.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")

import argparse
import logging
import json
import os
import time
from pathlib import Path

# Suppress requests warning before importing flwr
os.environ["PYTHONWARNINGS"] = "ignore"

import flwr as fl
import numpy as np
import torch

from fedpg_br.config import get_config, get_env_info, ATTACK_TYPES
from fedpg_br.flower import FedPGStrategy, create_client_fn
from fedpg_br.utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FedPG-BR: Fault-Tolerant Federated RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Environment
    parser.add_argument(
        "--env", 
        type=str, 
        default="CartPole-v1",
        choices=["CartPole-v1", "LunarLander-v2", "HalfCheetah-v2"],
        help="Gymnasium environment",
    )
    
    # Federation
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers (K)")
    parser.add_argument("--num_byzantine", type=int, default=0, help="Number of Byzantine agents")
    parser.add_argument(
        "--attack", 
        type=str, 
        default="random-noise",
        choices=ATTACK_TYPES,
        help="Byzantine attack type",
    )
    
    # Algorithm
    parser.add_argument("--fedpg_br", action="store_true", help="Use adaptive batch size (FedPG-BR)")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of rounds")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    
    # Experiment
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--no_save", action="store_true", help="Don't save results")
    
    # Logging
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    return parser.parse_args()


def run_experiment(
    config,
    env_info: dict,
    num_workers: int,
    num_byzantine: int,
    attack_type: str,
    use_adaptive_batch: bool,
    num_rounds: int,
    seed: int,
) -> dict:
    """
    Run a single experiment.
    
    Returns:
        dict with results
    """
    set_seed(seed)
    
    # Calculate Byzantine ratio
    byzantine_ratio = num_byzantine / num_workers if num_workers > 0 else 0.0
    
    logger.info(f"Starting experiment: {config.env_name}")
    logger.info(f"  Workers: {num_workers}, Byzantine: {num_byzantine} ({byzantine_ratio:.1%})")
    logger.info(f"  Attack: {attack_type}")
    logger.info(f"  Rounds: {num_rounds}, Seed: {seed}")
    
    # Create client factory
    client_fn = create_client_fn(
        env_name=config.env_name,
        hidden_units=config.hidden_units,
        gamma=config.gamma,
        activation=config.activation,
        output_activation=config.output_activation,
        num_byzantine=num_byzantine,
        attack_type=attack_type,
        max_episode_len=config.max_episode_len,
        device="cpu",
    )
    
    # Create strategy
    batch_range = config.batch_size_range if use_adaptive_batch else None
    
    strategy = FedPGStrategy(
        env_name=config.env_name,
        state_dim=env_info["state_dim"],
        action_dim=env_info["action_dim"],
        hidden_units=config.hidden_units,
        activation=config.activation,
        output_activation=config.output_activation,
        batch_size=config.batch_size,
        batch_size_range=batch_range,
        mini_batch_size=config.mini_batch_size,
        lr=config.lr,
        gamma=config.gamma,
        sigma=config.sigma,
        delta=config.delta,
        num_agents=num_workers,
        byzantine_ratio=byzantine_ratio,
        evaluate_every=10,
    )
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_workers,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"num_cpus": min(num_workers + 2, os.cpu_count() or 4)},
    )
    
    return {
        "seed": seed,
        "num_rounds": num_rounds,
        "losses": history.losses_distributed,
        "metrics": history.metrics_distributed,
    }


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load config
    overrides = {}
    if args.lr is not None:
        overrides["lr"] = args.lr
    
    config = get_config(args.env, **overrides)
    env_info = get_env_info(args.env)
    
    # Calculate number of rounds
    if args.rounds is not None:
        num_rounds = args.rounds
    else:
        avg_traj_per_round = (
            config.batch_size * args.num_workers + 
            config.mini_batch_size * config.svrpg_iterations
        ) / (1 + args.num_workers)
        num_rounds = int(config.max_trajectories / avg_traj_per_round)
    
    logger.info("="*60)
    logger.info("FedPG-BR Configuration")
    logger.info("="*60)
    logger.info(f"Environment: {args.env}")
    logger.info(f"Algorithm: {'FedPG-BR' if args.fedpg_br else 'SVRPG/GPOMDP'}")
    logger.info(f"Batch size: {config.batch_size} (range: {config.batch_size_range})")
    logger.info(f"Learning rate: {config.lr}")
    logger.info(f"Sigma: {config.sigma}, Delta: {config.delta}")
    logger.info("="*60)
    
    # Validate
    if args.num_byzantine > 0:
        alpha = args.num_byzantine / args.num_workers
        if alpha >= 0.5:
            raise ValueError(f"Byzantine ratio must be < 0.5, got {alpha:.2%}")
    
    # Output directory
    if not args.no_save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / args.env / f"w{args.num_workers}_b{args.num_byzantine}_{args.attack}" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config (convert numpy types to native Python)
        def to_serializable(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        with open(output_dir / "config.json", "w") as f:
            json.dump({
                "args": {k: to_serializable(v) for k, v in vars(args).items()},
                "config": {k: to_serializable(v) for k, v in config.__dict__.items()},
                "env_info": {k: to_serializable(v) for k, v in env_info.items()},
            }, f, indent=2)
    
    # Run experiments
    all_results = []
    seeds = list(range(args.seed, args.seed + args.num_runs))
    
    for i, seed in enumerate(seeds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {i+1}/{args.num_runs} (seed={seed})")
        logger.info(f"{'='*60}")
        
        result = run_experiment(
            config=config,
            env_info=env_info,
            num_workers=args.num_workers,
            num_byzantine=args.num_byzantine,
            attack_type=args.attack,
            use_adaptive_batch=args.fedpg_br,
            num_rounds=num_rounds,
            seed=seed,
        )
        all_results.append(result)
        
        if not args.no_save:
            with open(output_dir / f"result_seed{seed}.json", "w") as f:
                # Convert to JSON-serializable format
                json.dump({
                    "seed": result["seed"],
                    "num_rounds": result["num_rounds"],
                    "final_loss": result["losses"][-1][1] if result["losses"] else None,
                }, f, indent=2)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*60)
    logger.info(f"Completed {len(all_results)}/{args.num_runs} runs")
    
    if not args.no_save:
        logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
