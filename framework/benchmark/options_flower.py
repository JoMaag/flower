#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options/Arguments for FedPG-BR Flower Implementation
Based on options.py from original implementation
"""

import os
import time
import argparse
from original_configs_v2 import get_config


def get_options(args=None):
    """
    Parse command line arguments
    
    Returns:
        Namespace with all options
    """
    parser = argparse.ArgumentParser('FedPG-BR with Flower')

    # ========================================================================
    # Overall run settings
    # ========================================================================
    parser.add_argument(
        '--env_name', '--env', 
        type=str, 
        default='CartPole-v1', 
        choices=['HalfCheetah-v2', 'LunarLander-v2', 'CartPole-v1'], 
        help='OpenAI Gym environment name'
    )
    
    parser.add_argument(
        '--eval_only', 
        action='store_true', 
        help='Evaluation only (no training)'
    )
    
    parser.add_argument(
        '--no_saving', 
        action='store_true', 
        help='Disable saving checkpoints'
    )
    
    parser.add_argument(
        '--no_tb', 
        action='store_true', 
        help='Disable TensorBoard logging'
    )
    
    parser.add_argument(
        '--render', 
        action='store_true', 
        help='Render environment during evaluation'
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['human', 'rgb'], 
        default='human', 
        help='Render mode'
    )
    
    parser.add_argument(
        '--log_dir', 
        default='logs', 
        help='Directory for TensorBoard logs'
    )
    
    parser.add_argument(
        '--run_name', 
        default='fedpg_flower', 
        help='Name to identify the experiment'
    )
    
    # ========================================================================
    # Multiple runs
    # ========================================================================
    parser.add_argument(
        '--multiple_run', 
        type=int, 
        default=1, 
        help='Number of repeated runs with different seeds'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=0, 
        help='Starting random seed'
    )

    # ========================================================================
    # Federation and Byzantine parameters
    # ========================================================================
    parser.add_argument(
        '--num_worker', 
        type=int, 
        default=10, 
        help='Number of worker agents (K in paper)'
    )
    
    parser.add_argument(
        '--num_Byzantine', 
        type=int, 
        default=0, 
        help='Number of Byzantine agents'
    )
    
    parser.add_argument(
        '--alpha', 
        type=float, 
        default=0.4, 
        help='Maximum fraction of Byzantine agents (must be < 0.5)'
    )
    
    parser.add_argument(
        '--attack_type', 
        type=str, 
        default='random-noise', 
        choices=[
            'zero-gradient', 
            'random-action', 
            'sign-flipping', 
            'reward-flipping', 
            'random-reward', 
            'random-noise', 
            'FedScsPG-attack'
        ],
        help='Type of Byzantine attack'
    )
        
    # ========================================================================
    # RL Algorithms (default GPOMDP)
    # ========================================================================
    parser.add_argument(
        '--SVRPG', 
        action='store_true', 
        help='Run SVRPG algorithm'
    )
    
    parser.add_argument(
        '--FedPG_BR', 
        action='store_true', 
        help='Run FedPG-BR algorithm (with adaptive batch size)'
    )

    # ========================================================================
    # Training and validation
    # ========================================================================
    parser.add_argument(
        '--val_size', 
        type=int, 
        default=10, 
        help='Number of episodes for validation'
    )
    
    parser.add_argument(
        '--val_max_steps', 
        type=int, 
        default=1000, 
        help='Maximum trajectory length for validation'
    )
    
    # ========================================================================
    # Load pre-trained models
    # ========================================================================
    parser.add_argument(
        '--load_path', 
        default=None,
        help='Path to load pre-trained model checkpoint'
    )

    # ========================================================================
    # Custom hyperparameters (override config)
    # ========================================================================
    parser.add_argument(
        '--custom_config',
        action='store_true',
        help='Use custom hyperparameters instead of original config'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )

    # ========================================================================
    # Parse arguments
    # ========================================================================
    opts = parser.parse_args(args)

    # ========================================================================
    # Post-processing
    # ========================================================================
    opts.use_cuda = False  # Flower simulation typically uses CPU
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    
    # Setup directories
    opts.save_dir = os.path.join(
        'outputs',
        '{}'.format(opts.env_name),
        "worker{}_byzantine{}_{}".format(
            opts.num_worker, 
            opts.num_Byzantine, 
            opts.attack_type
        ),
        opts.run_name
    ) if not opts.no_saving else None
    
    opts.log_dir = os.path.join(
        f'{opts.log_dir}',
        '{}'.format(opts.env_name),
        "worker{}_byzantine{}_{}".format(
            opts.num_worker, 
            opts.num_Byzantine, 
            opts.attack_type
        ),
        opts.run_name
    ) if not opts.no_tb else None
    
    # ========================================================================
    # Load environment-specific config (from original options.py)
    # ========================================================================
    if not opts.custom_config:
        config = get_config(opts.env_name)
        
        # Load all config parameters
        opts.max_epi_len = config.max_epi_len
        opts.max_trajectories = config.max_trajectories
        opts.gamma = config.gamma
        opts.min_reward = getattr(config, 'min_reward', 0)
        opts.max_reward = getattr(config, 'max_reward', 1000)
        
        # Network architecture
        opts.hidden_units = ','.join(map(str, config.hidden_units))
        opts.activation = config.activation
        opts.output_activation = config.output_activation
        
        # Training
        opts.do_sample_for_training = config.do_sample_for_training
        opts.lr_model = opts.lr if opts.lr is not None else config.lr_model
        
        # Batch sizes
        opts.B = opts.batch_size if opts.batch_size is not None else config.B
        opts.Bmin = config.Bmin
        opts.Bmax = config.Bmax
        opts.b = config.b
        
        # SVRPG
        opts.N = config.N
        
        # Byzantine filtering
        opts.delta = config.delta
        opts.sigma = config.sigma
        
        print(f"Loaded configuration for {opts.env_name}:")
        print(f"  Batch size: {opts.B} (range: [{opts.Bmin}, {opts.Bmax}])")
        print(f"  Mini-batch: {opts.b}")
        print(f"  Learning rate: {opts.lr_model}")
        print(f"  Gamma: {opts.gamma}")
        print(f"  Sigma: {opts.sigma}, Delta: {opts.delta}")
    
    else:
        print("Using custom configuration")
        # Set defaults if not using config
        if not hasattr(opts, 'max_epi_len'):
            opts.max_epi_len = 1000
        if not hasattr(opts, 'max_trajectories'):
            opts.max_trajectories = 10000
        if not hasattr(opts, 'gamma'):
            opts.gamma = 0.99
        if not hasattr(opts, 'lr_model'):
            opts.lr_model = 1e-3
        if not hasattr(opts, 'B'):
            opts.B = 16
        if not hasattr(opts, 'b'):
            opts.b = 4
        if not hasattr(opts, 'sigma'):
            opts.sigma = 0.1
        if not hasattr(opts, 'delta'):
            opts.delta = 0.6
    
    # ========================================================================
    # Validate options
    # ========================================================================
    assert opts.SVRPG + opts.FedPG_BR <= 1, "Can only use one algorithm at a time"
    
    if opts.SVRPG + opts.FedPG_BR == 0:
        print('Running GPOMDP (baseline)')
    elif opts.FedPG_BR:
        print('Running FedPG-BR')
    else:
        print('Running SVRPG')
    
    assert opts.alpha < 0.5, "Alpha must be < 0.5 (need majority of good agents)"
    
    if opts.num_Byzantine > 0:
        actual_alpha = opts.num_Byzantine / opts.num_worker
        assert actual_alpha < 0.5, f"Too many Byzantine agents: {actual_alpha:.2%} >= 50%"
        print(f"Byzantine agents: {opts.num_Byzantine}/{opts.num_worker} ({actual_alpha:.2%})")
    
    return opts


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("OPTIONS PARSER - Examples")
    print("="*80)
    
    # Example 1: Default CartPole
    print("\n1. Default CartPole:")
    print("-"*40)
    opts1 = get_options(['--env_name', 'CartPole-v1'])
    print(f"Environment: {opts1.env_name}")
    print(f"Batch size: {opts1.B}")
    print(f"Learning rate: {opts1.lr_model}")
    
    # Example 2: With Byzantine agents
    print("\n2. With Byzantine agents:")
    print("-"*40)
    opts2 = get_options([
        '--env_name', 'CartPole-v1',
        '--num_worker', '10',
        '--num_Byzantine', '3',
        '--attack_type', 'sign-flipping',
        '--FedPG_BR'
    ])
    print(f"Workers: {opts2.num_worker}")
    print(f"Byzantine: {opts2.num_Byzantine}")
    print(f"Attack: {opts2.attack_type}")
    print(f"Algorithm: FedPG-BR")
    
    # Example 3: Multiple runs
    print("\n3. Multiple runs:")
    print("-"*40)
    opts3 = get_options([
        '--env_name', 'LunarLander-v2',
        '--multiple_run', '5',
        '--seed', '42'
    ])
    print(f"Multiple runs: {opts3.multiple_run}")
    print(f"Seeds: {opts3.seeds}")
    
    print("\n" + "="*80)
