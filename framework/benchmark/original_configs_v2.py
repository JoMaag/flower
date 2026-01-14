"""
Exact Hyperparameters from Original Implementation
Based on options.py from: https://github.com/flint-xf-fan/Byzantine-Federated-RL
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CartPoleConfig:
    """CartPole-v1 Hyperparameters (from options.py lines 94-114)"""
    # Environment
    env_name: str = "CartPole-v1"
    max_epi_len: int = 500
    max_trajectories: int = 5000
    gamma: float = 0.999
    
    # Network architecture
    hidden_units: Tuple[int, ...] = (16, 16)
    activation: str = "ReLU"
    output_activation: str = "Tanh"
    
    # Training
    lr_model: float = 1e-3
    do_sample_for_training: bool = True
    
    # Batch sizes
    B: int = 16  # for SVRPG and GPOMDP
    Bmin: int = 12  # for FedPG-BR
    Bmax: int = 20  # for FedPG-BR
    b: int = 4  # mini batch for SVRPG and FedPG-BR
    
    # SVRPG parameters
    N: int = 3  # inner loop iterations for SVRPG
    
    # Byzantine filtering
    delta: float = 0.6
    sigma: float = 0.06
    
    # Federation
    num_worker: int = 10
    num_byzantine: int = 0
    alpha: float = 0.4
    
    # Validation
    val_size: int = 10
    val_max_steps: int = 1000


@dataclass
class LunarLanderConfig:
    """LunarLander-v2 Hyperparameters (from options.py lines 143-168)"""
    # Environment
    env_name: str = "LunarLander-v2"
    max_epi_len: int = 1000
    max_trajectories: int = 10000
    gamma: float = 0.99
    
    # Network architecture
    hidden_units: Tuple[int, ...] = (64, 64)
    activation: str = "Tanh"
    output_activation: str = "Tanh"
    
    # Training
    lr_model: float = 1e-3
    do_sample_for_training: bool = True
    
    # Batch sizes
    B: int = 32  # for SVRPG and GPOMDP
    Bmin: int = 26  # for FedPG-BR
    Bmax: int = 38  # for FedPG-BR
    b: int = 8  # mini batch for SVRPG and FedPG-BR
    
    # SVRPG parameters
    N: int = 3
    
    # Byzantine filtering
    delta: float = 0.6
    sigma: float = 0.07
    
    # Federation
    num_worker: int = 10
    num_byzantine: int = 0
    alpha: float = 0.4
    
    # Validation
    val_size: int = 10
    val_max_steps: int = 1000


@dataclass
class HalfCheetahConfig:
    """HalfCheetah-v2 Hyperparameters (from options.py lines 117-140)"""
    # Environment
    env_name: str = "HalfCheetah-v2"
    max_epi_len: int = 500
    max_trajectories: int = 10000
    gamma: float = 0.995
    
    # Network architecture
    hidden_units: Tuple[int, ...] = (64, 64)
    activation: str = "Tanh"
    output_activation: str = "Tanh"
    
    # Training
    lr_model: float = 8e-5
    do_sample_for_training: bool = True
    
    # Batch sizes
    B: int = 48  # for SVRPG and GPOMDP
    Bmin: int = 46  # for FedPG-BR
    Bmax: int = 50  # for FedPG-BR
    b: int = 16  # mini batch for SVRPG and FedPG-BR
    
    # SVRPG parameters
    N: int = 3
    
    # Byzantine filtering
    delta: float = 0.6
    sigma: float = 0.9
    
    # Federation
    num_worker: int = 10
    num_byzantine: int = 0
    alpha: float = 0.4
    
    # Validation
    val_size: int = 10
    val_max_steps: int = 1000


def get_config(env_name: str, **kwargs):
    """
    Get configuration for specific environment
    
    Args:
        env_name: Environment name
        **kwargs: Override default parameters
        
    Returns:
        Configuration dataclass
    """
    if env_name == "CartPole-v1":
        config = CartPoleConfig(**kwargs)
    elif env_name == "LunarLander-v2":
        config = LunarLanderConfig(**kwargs)
    elif env_name == "HalfCheetah-v2":
        config = HalfCheetahConfig(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    return config


def get_env_info(env_name: str):
    """
    Get environment information (state_dim, action_dim, is_continuous)
    
    Args:
        env_name: Environment name
        
    Returns:
        dict with state_dim, action_dim, is_continuous
    """
    import gymnasium as gym
    
    env = gym.make(env_name)
    
    state_dim = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        is_continuous = False
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        is_continuous = True
    else:
        raise ValueError(f"Unsupported action space: {type(env.action_space)}")
    
    env.close()
    
    return {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "is_continuous": is_continuous
    }


# ============================================================================
# ATTACK TYPES FROM ORIGINAL
# ============================================================================

ATTACK_TYPES = [
    'zero-gradient',      # Send zero gradient
    'random-action',      # Take random actions (hardware failure simulation)
    'sign-flipping',      # Send negative gradient
    'reward-flipping',    # Flip reward signs
    'random-reward',      # Random rewards
    'random-noise',       # Random Gaussian noise
    'FedScsPG-attack'     # Sophisticated attack (mean + 3*sigma)
]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("EXACT HYPERPARAMETERS FROM ORIGINAL IMPLEMENTATION")
    print("="*80)
    
    # CartPole configuration
    print("\n1. CartPole-v1:")
    print("-" * 40)
    cartpole_config = get_config("CartPole-v1")
    print(f"Batch size range: [{cartpole_config.Bmin}, {cartpole_config.Bmax}]")
    print(f"Mini-batch size: {cartpole_config.b}")
    print(f"Learning rate: {cartpole_config.lr_model}")
    print(f"Gamma: {cartpole_config.gamma}")
    print(f"Sigma (filtering): {cartpole_config.sigma}")
    print(f"Delta (filtering): {cartpole_config.delta}")
    print(f"Hidden units: {cartpole_config.hidden_units}")
    print(f"Max trajectories: {cartpole_config.max_trajectories}")
    
    # LunarLander configuration
    print("\n2. LunarLander-v2:")
    print("-" * 40)
    lunar_config = get_config("LunarLander-v2")
    print(f"Batch size range: [{lunar_config.Bmin}, {lunar_config.Bmax}]")
    print(f"Mini-batch size: {lunar_config.b}")
    print(f"Learning rate: {lunar_config.lr_model}")
    print(f"Gamma: {lunar_config.gamma}")
    print(f"Sigma (filtering): {lunar_config.sigma}")
    print(f"Delta (filtering): {lunar_config.delta}")
    print(f"Hidden units: {lunar_config.hidden_units}")
    print(f"Max trajectories: {lunar_config.max_trajectories}")
    
    # HalfCheetah configuration
    print("\n3. HalfCheetah-v2:")
    print("-" * 40)
    cheetah_config = get_config("HalfCheetah-v2")
    print(f"Batch size range: [{cheetah_config.Bmin}, {cheetah_config.Bmax}]")
    print(f"Mini-batch size: {cheetah_config.b}")
    print(f"Learning rate: {cheetah_config.lr_model}")
    print(f"Gamma: {cheetah_config.gamma}")
    print(f"Sigma (filtering): {cheetah_config.sigma}")
    print(f"Delta (filtering): {cheetah_config.delta}")
    print(f"Hidden units: {cheetah_config.hidden_units}")
    print(f"Max trajectories: {cheetah_config.max_trajectories}")
    
    # Available attack types
    print("\n4. Available Attack Types:")
    print("-" * 40)
    for attack in ATTACK_TYPES:
        print(f"  - {attack}")
    
    print("\n" + "="*80)
    print("Use these exact values to reproduce paper results!")
    print("="*80)
