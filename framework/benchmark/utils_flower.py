#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportGeneralTypeIssues=false
# type: ignore
"""
Utility Functions for FedPG-BR Flower Implementation
Based on utils.py from original implementation
"""

import torch
import numpy as np
from torch.nn import DataParallel
from matplotlib import animation
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# PyTorch Utilities
# ============================================================================

def torch_load_cpu(load_path):
    """
    Load PyTorch checkpoint on CPU
    
    Args:
        load_path: Path to checkpoint file
        
    Returns:
        Loaded checkpoint dictionary
    """
    return torch.load(load_path, map_location=lambda storage, loc: storage)


def get_inner_model(model):
    """
    Get the actual model from DataParallel wrapper if needed
    
    Args:
        model: PyTorch model (possibly wrapped in DataParallel)
        
    Returns:
        Unwrapped model
    """
    return model.module if isinstance(model, DataParallel) else model


def move_to(var, device):
    """
    Move variable(s) to specified device
    
    Args:
        var: Variable, dict of variables, or list of variables
        device: Target device (e.g., 'cuda' or 'cpu')
        
    Returns:
        Variable(s) on target device
    """
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    elif isinstance(var, list):
        return [move_to(v, device) for v in var]
    return var.to(device)


# ============================================================================
# Environment Utilities
# ============================================================================

def env_wrapper(name, obs):
    """
    Wrapper for environment observations
    Can be used to preprocess observations if needed
    
    Args:
        name: Environment name
        obs: Observation from environment
        
    Returns:
        Processed observation
    """
    # Currently just returns obs as-is
    # Can be extended for specific environments
    return obs


def get_env_type(env_name):
    """
    Determine environment type (discrete vs continuous)
    
    Args:
        env_name: Environment name
        
    Returns:
        'discrete' or 'continuous'
    """
    discrete_envs = ['CartPole-v1', 'CartPole-v0', 'LunarLander-v2', 'Acrobot-v1']
    continuous_envs = ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2', 'Ant-v2']
    
    if env_name in discrete_envs:
        return 'discrete'
    elif env_name in continuous_envs:
        return 'continuous'
    else:
        # Default: try to detect from gym
        import gym
        env = gym.make(env_name)
        if isinstance(env.action_space, gym.spaces.Discrete):
            env_type = 'discrete'
        else:
            env_type = 'continuous'
        env.close()
        return env_type


# ============================================================================
# Visualization Utilities
# ============================================================================

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    """
    Save list of frames as animated GIF
    
    Args:
        frames: List of numpy arrays (frames)
        path: Directory to save GIF
        filename: Filename for GIF
    """
    # Ensure path exists
    Path(path).mkdir(parents=True, exist_ok=True)
    
    # Create figure
    plt.figure(
        figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), 
        dpi=72
    )
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
    
    # Create animation
    anim = animation.FuncAnimation(
        plt.gcf(), 
        animate, 
        frames=len(frames), 
        interval=50
    )
    
    # Save
    full_path = Path(path) / filename
    anim.save(str(full_path), writer='imagemagick', fps=120)
    print(f"Animation saved to: {full_path}")


def render_policy(policy, env_name, num_episodes=1, max_steps=1000, save_path=None):
    """
    Render policy in environment and optionally save as GIF
    
    Args:
        policy: Policy network
        env_name: Environment name
        num_episodes: Number of episodes to render
        max_steps: Maximum steps per episode
        save_path: If provided, save frames as GIF
        
    Returns:
        List of episode rewards
    """
    import gym
    from fedpg_br_old import sample_trajectory
    
    env = gym.make(env_name)
    rewards = []
    
    for episode in range(num_episodes):
        frames = []
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Render frame
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                if hasattr(policy, 'forward'):
                    action, _ = policy(state_tensor, sample=False)
                else:
                    action_probs = policy(state_tensor)
                    action = action_probs.argmax().item()
            
            # Take step
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
        
        # Save frames if requested
        if save_path is not None:
            gif_name = f"episode_{episode}.gif"
            save_frames_as_gif(frames, path=save_path, filename=gif_name)
    
    env.close()
    return rewards


# ============================================================================
# Flower-specific Utilities
# ============================================================================

def numpy_to_parameters(arrays):
    """
    Convert list of numpy arrays to Flower Parameters
    
    Args:
        arrays: List of numpy arrays
        
    Returns:
        Flower Parameters object
    """
    from flwr.common import ndarrays_to_parameters
    return ndarrays_to_parameters(arrays)


def parameters_to_numpy(parameters):
    """
    Convert Flower Parameters to list of numpy arrays
    
    Args:
        parameters: Flower Parameters object
        
    Returns:
        List of numpy arrays
    """
    from flwr.common import parameters_to_ndarrays
    return parameters_to_ndarrays(parameters)


def flatten_parameters(state_dict):
    """
    Flatten PyTorch state dict to single numpy array
    
    Args:
        state_dict: PyTorch state dict
        
    Returns:
        Flattened numpy array
    """
    return np.concatenate([v.cpu().numpy().flatten() for v in state_dict.values()])


def unflatten_parameters(flat_params, state_dict):
    """
    Unflatten numpy array to PyTorch state dict
    
    Args:
        flat_params: Flattened numpy array
        state_dict: Reference state dict (for shapes)
        
    Returns:
        State dict with unflattened parameters
    """
    new_state_dict = {}
    offset = 0
    
    for key, value in state_dict.items():
        shape = value.shape
        size = value.numel()
        new_state_dict[key] = torch.from_numpy(
            flat_params[offset:offset + size].reshape(shape)
        )
        offset += size
    
    return new_state_dict


# ============================================================================
# Statistics Utilities
# ============================================================================

def compute_confidence_interval(data, confidence=0.90):
    """
    Compute confidence interval for data
    
    Args:
        data: List or array of values
        confidence: Confidence level (default: 0.90 for 90% CI)
        
    Returns:
        (mean, lower_bound, upper_bound)
    """
    import scipy.stats as st
    
    data = np.array(data)
    mean = np.mean(data)
    
    if len(data) < 2:
        return mean, mean, mean
    
    # Compute confidence interval
    interval = st.t.interval(
        confidence,
        len(data) - 1,
        loc=mean,
        scale=st.sem(data)
    )
    
    return mean, interval[0], interval[1]


def aggregate_run_results(histories):
    """
    Aggregate results from multiple runs
    
    Args:
        histories: List of Flower History objects
        
    Returns:
        Dictionary with aggregated metrics
    """
    results = {
        'num_runs': len(histories),
        'final_losses': [],
        'num_rounds': []
    }
    
    for history in histories:
        if history.losses_distributed:
            results['final_losses'].append(history.losses_distributed[-1][1])
            results['num_rounds'].append(len(history.losses_distributed))
    
    # Compute statistics
    if results['final_losses']:
        mean_loss, lower, upper = compute_confidence_interval(results['final_losses'])
        results['mean_final_loss'] = mean_loss
        results['ci_lower'] = lower
        results['ci_upper'] = upper
    
    return results


# ============================================================================
# Logging Utilities
# ============================================================================

def print_summary(opts, results):
    """
    Print summary of experiment results
    
    Args:
        opts: Options/configuration
        results: Results dictionary
    """
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Environment: {opts.env_name}")
    print(f"Algorithm: {'FedPG-BR' if opts.FedPG_BR else 'SVRPG' if opts.SVRPG else 'GPOMDP'}")
    print(f"Workers: {opts.num_worker} (Byzantine: {opts.num_Byzantine})")
    print(f"Runs completed: {results.get('num_runs', 0)}")
    
    if 'mean_final_loss' in results:
        print(f"\nFinal Loss: {results['mean_final_loss']:.4f}")
        print(f"  90% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    
    print("="*80 + "\n")


# ============================================================================
# File I/O Utilities
# ============================================================================

def save_checkpoint(model, optimizer, epoch, save_path):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': get_inner_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")


def load_checkpoint(model, optimizer, load_path, device='cpu'):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        load_path: Path to checkpoint
        device: Device to load to
        
    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch_load_cpu(load_path)
    
    get_inner_model(model).load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Move optimizer state to device
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    
    # Restore RNG state
    torch.set_rng_state(checkpoint['rng_state'])
    if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    
    print(f"Checkpoint loaded from: {load_path}")
    return checkpoint.get('epoch', 0)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("UTILS - Example Usage")
    print("="*80)
    
    # Test device movement
    print("\n1. Device movement:")
    tensor = torch.randn(3, 4)
    moved = move_to(tensor, 'cpu')
    print(f"Original device: {tensor.device}, Moved device: {moved.device}")
    
    # Test environment type detection
    print("\n2. Environment type detection:")
    print(f"CartPole-v1: {get_env_type('CartPole-v1')}")
    print(f"HalfCheetah-v2: {get_env_type('HalfCheetah-v2')}")
    
    # Test confidence interval
    print("\n3. Confidence intervals:")
    data = [100, 105, 98, 102, 97, 103, 99, 101, 104, 100]
    mean, lower, upper = compute_confidence_interval(data, confidence=0.90)
    print(f"Data: {data}")
    print(f"Mean: {mean:.2f}, 90% CI: [{lower:.2f}, {upper:.2f}]")
    
    print("\n" + "="*80)
