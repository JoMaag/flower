#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportGeneralTypeIssues=false
# type: ignore
"""
Flower-based FedPG-BR Runner
Equivalent to run.py from original implementation

Main entry point for training and evaluation with Flower simulation
"""

import os
import json
import torch
import pprint
import numpy as np
import warnings
from pathlib import Path
from typing import Optional

# Flower imports
import flwr as fl

# Local imports
from flower_fedpg_client import create_client_fn
from framework.benchmark.flower_fedpg_strategy import FedPGStrategy
from original_configs_v2 import get_config, get_env_info
from original_policy import create_policy

# Optional: tensorboard for logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")


def setup_directories(opts):
    """Setup save and log directories"""
    if not opts.no_saving:
        os.makedirs(opts.save_dir, exist_ok=True)
        print(f"Saving checkpoints to: {opts.save_dir}")
    
    if not opts.no_tb and TENSORBOARD_AVAILABLE:
        os.makedirs(opts.log_dir, exist_ok=True)
        print(f"TensorBoard logs to: {opts.log_dir}")


def setup_tensorboard(opts, run_id: int):
    """Setup tensorboard writer for a run"""
    if opts.no_tb or not TENSORBOARD_AVAILABLE:
        return None
    
    log_path = os.path.join(opts.log_dir, f"run_{run_id}")
    return SummaryWriter(log_path)


def save_args(opts):
    """Save arguments to JSON for reproducibility"""
    if not opts.no_saving:
        args_path = os.path.join(opts.save_dir, "args.json")
        with open(args_path, 'w') as f:
            json.dump(vars(opts), f, indent=2)
        print(f"Arguments saved to: {args_path}")


def run_single_seed(opts, run_id: int, tb_writer=None):
    """
    Run training for a single seed (equivalent to agent.start_training)
    
    Args:
        opts: Options/configuration
        run_id: Seed/run identifier
        tb_writer: TensorBoard writer (optional)
    """
    # Set random seeds
    torch.manual_seed(run_id)
    np.random.seed(run_id)
    
    print("\n" + "="*80)
    print(f"Starting Run {run_id} / {opts.seeds}")
    print("="*80)
    
    # Get environment info
    env_info = get_env_info(opts.env_name)
    state_dim = env_info["state_dim"]
    action_dim = env_info["action_dim"]
    is_continuous = env_info["is_continuous"]
    
    # Get configuration
    config = get_config(opts.env_name)
    
    print(f"Environment: {opts.env_name}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    print(f"  Continuous: {is_continuous}")
    print(f"  Config: B∈[{config.Bmin},{config.Bmax}], b={config.b}, lr={config.lr_model}")
    
    # Calculate number of Byzantine agents
    num_byzantine = opts.num_Byzantine if opts.num_Byzantine > 0 else 0
    byzantine_ratio = num_byzantine / opts.num_worker if opts.num_worker > 0 else 0.0
    
   # Create client factory
    # Create client factory
    client_fn = create_client_fn(
        env_name=opts.env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        num_byzantine=num_byzantine,
        attack_type=opts.attack_type,
        gamma=config.gamma,
        variance_bound=config.sigma,
        # **KORRIGIERT: Fehlende Parameter aus config**
        hidden_units=','.join(map(str, config.hidden_units)),  # (64, 64) -> '64,64'
        activation=config.activation,
        output_activation=config.output_activation,
        max_epi_len=config.max_epi_len,
        device=str(opts.device)
    )
    
    # Create strategy with adaptive batch size sampling
    strategy = FedPGStrategy(
        env_name=opts.env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=config.B,
        batch_size_range=(config.Bmin, config.Bmax) if opts.FedPG_BR else None,
        mini_batch_size=config.b,
        step_size=config.lr_model,
        gamma=config.gamma,
        variance_bound=config.sigma,
        confidence_param=config.delta,
        byzantine_ratio=byzantine_ratio,
        num_agents=opts.num_worker,
        min_available_clients=opts.num_worker,
        evaluate_every=10
    )
    
    # Calculate number of rounds
    # From original: step <= max_trajectories
    # Each round: ~(B * K + b * N_t) / (1 + K) trajectories per agent
    avg_trajectories_per_round = (config.B * opts.num_worker + config.b * config.N) / (1 + opts.num_worker)
    num_rounds = int(config.max_trajectories / avg_trajectories_per_round)
    
    print(f"\nTraining Configuration:")
    print(f"  Workers: {opts.num_worker} (Byzantine: {num_byzantine})")
    print(f"  Attack type: {opts.attack_type}")
    print(f"  Max trajectories: {config.max_trajectories}")
    print(f"  Estimated rounds: {num_rounds}")
    print(f"  Algorithm: {'FedPG-BR' if opts.FedPG_BR else 'SVRPG' if opts.SVRPG else 'GPOMDP'}")
    
    # Configure client resources
    client_resources = {
        "num_cpus": 4,
        "num_gpus": 0.0
    }
    
    # Run Flower simulation
    print("\nStarting Flower simulation...")
    history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=opts.num_worker,
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0.0},
    actor_kwargs={
        "on_actor_init_fn": None
    },
    ray_init_args={"num_cpus": 10},  # **WICHTIG**
)
    
    print("\n" + "="*80)
    print(f"Run {run_id} completed!")
    print("="*80)
    
    # Log to tensorboard if available
    if tb_writer is not None:
        # Log final metrics
        if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
            if history.metrics_distributed:
                for key, values in history.metrics_distributed.items():
                    for round_num, value in values:
                        if tb_writer is not None:
                            tb_writer.add_scalar(f"{key}/run_{run_id}", value, round_num)
        
        if history.losses_distributed:
            for round_num, loss in history.losses_distributed:
                tb_writer.add_scalar(f"loss/run_{run_id}", loss, round_num)
    
    return history


def run_evaluation(opts):
    """
    Run evaluation only (equivalent to agent.start_validating)
    """
    print("\n" + "="*80)
    print("EVALUATION MODE")
    print("="*80)
    
    # Set random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    
    # Get environment info
    env_info = get_env_info(opts.env_name)
    config = get_config(opts.env_name)
    
    # Create policy
    policy = create_policy(
        state_dim=env_info["state_dim"],
        action_dim=env_info["action_dim"],
        env_name=opts.env_name,
        hidden_units=tuple(map(int, config.hidden_units)),
        activation=config.activation,
        output_activation=config.output_activation
    )
    
    # Load checkpoint if provided
    if opts.load_path is not None:
        print(f"Loading checkpoint from: {opts.load_path}")
        checkpoint = torch.load(opts.load_path, map_location='cpu')
        policy.load_state_dict(checkpoint['master'])
        print("Checkpoint loaded successfully")
    
    # Evaluate
    import gym
    from fedpg_br_old import sample_trajectory
    
    env = gym.make(opts.env_name)
    total_rewards = []
    
    print(f"\nEvaluating for {opts.val_size} episodes...")
    for episode in range(opts.val_size):
        trajectory, reward = sample_trajectory(
            env, 
            policy, 
            max_steps=opts.val_max_steps
        )
        total_rewards.append(reward)
        print(f"Episode {episode+1}/{opts.val_size}: {reward:.2f}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print("\n" + "="*80)
    print(f"Evaluation Results:")
    print(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Min: {np.min(total_rewards):.2f}, Max: {np.max(total_rewards):.2f}")
    print("="*80)
    
    env.close()


def run(opts):
    """
    Main run function (equivalent to original run.py)
    
    Args:
        opts: Options from argument parser
    """
    # Pretty print the configuration
    print("\n" + "="*80)
    print("FedPG-BR with Flower - Configuration")
    print("="*80)
    pprint.pprint(vars(opts))
    print("="*80 + "\n")
    
    # Setup directories
    setup_directories(opts)
    
    # Save arguments
    save_args(opts)
    
    # Configure for multiple runs
    assert opts.multiple_run > 0, "multiple_run must be > 0"
    opts.seeds = (np.arange(opts.multiple_run) + opts.seed).tolist()
    print(f"Running {opts.multiple_run} experiment(s) with seeds: {opts.seeds}")
    
    # Set device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    print(f"Using device: {opts.device}")
    
    # Evaluation only mode
    if opts.eval_only:
        run_evaluation(opts)
        return
    
    # Training mode - run for each seed
    all_histories = []
    
    for run_id in opts.seeds:
        # Setup tensorboard for this run
        tb_writer = setup_tensorboard(opts, run_id)
        
        try:
            # Run training
            history = run_single_seed(opts, run_id, tb_writer)
            all_histories.append(history)
            
            # Save results
            if not opts.no_saving:
                results_path = os.path.join(opts.save_dir, f"history_run_{run_id}.json")
                # Save history (simplified version)
                results = {
                    "run_id": run_id,
                    "num_rounds": len(history.losses_distributed) if history.losses_distributed else 0,
                    "final_loss": history.losses_distributed[-1][1] if history.losses_distributed else None
                }
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to: {results_path}")
        
        finally:
            if tb_writer is not None:
                tb_writer.close()
    
    # Print summary
    print("\n" + "="*80)
    print("ALL RUNS COMPLETED")
    print("="*80)
    print(f"Completed {len(all_histories)}/{opts.multiple_run} runs successfully")
    
    if not opts.no_saving:
        print(f"Results saved to: {opts.save_dir}")
    if not opts.no_tb and TENSORBOARD_AVAILABLE:
        print(f"View logs with: tensorboard --logdir {opts.log_dir}")
    
    print("\n✓ Experiment finished successfully!")


if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore", category=Warning)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    # Import options
    from options_flower import get_options
    
    # Run
    run(get_options())
