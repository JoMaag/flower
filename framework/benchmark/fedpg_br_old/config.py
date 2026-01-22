"""
Unified Configuration for FedPG-BR

Combines environment-specific hyperparameters from the original paper.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import gymnasium as gym


ATTACK_TYPES = [
    "zero-gradient",      # Send zero gradient
    "random-action",      # Take random actions (hardware failure)
    "sign-flipping",      # Send negative gradient (* -2.5)
    "reward-flipping",    # Flip reward signs
    "random-reward",      # Shuffle rewards randomly
    "random-noise",       # Add random Gaussian noise
    "FedScsPG-attack",    # Sophisticated attack (mean + 3*sigma)
]


@dataclass
class Config:
    """Configuration for FedPG-BR algorithm."""
    
    # Environment
    env_name: str = "CartPole-v1"
    max_episode_len: int = 500
    max_trajectories: int = 5000
    gamma: float = 0.999
    
    # Network architecture
    hidden_units: Tuple[int, ...] = (16, 16)
    activation: str = "ReLU"
    output_activation: str = "Tanh"
    
    # Training
    lr: float = 1e-3
    
    # Batch sizes (B = fixed, Bmin/Bmax = adaptive for FedPG-BR)
    batch_size: int = 16
    batch_size_min: int = 12
    batch_size_max: int = 20
    mini_batch_size: int = 4  # b in paper
    
    # SVRPG inner loop iterations (N in paper)
    svrpg_iterations: int = 3
    
    # Byzantine filtering parameters
    sigma: float = 0.06  # Variance bound
    delta: float = 0.6   # Confidence parameter
    
    # Federation
    num_workers: int = 10
    num_byzantine: int = 0
    attack_type: str = "random-noise"
    
    # Validation
    val_episodes: int = 10
    val_max_steps: int = 1000
    
    @property
    def alpha(self) -> float:
        """Byzantine ratio (must be < 0.5)."""
        if self.num_workers == 0:
            return 0.0
        return self.num_byzantine / self.num_workers
    
    @property
    def batch_size_range(self) -> Tuple[int, int]:
        """Adaptive batch size range for FedPG-BR."""
        return (self.batch_size_min, self.batch_size_max)
    
    def filtering_threshold(self, batch_size: int) -> float:
        """T_mu = 2*sigma*sqrt(V/B) where V = 2*log(2K/delta)."""
        import numpy as np
        V = 2 * np.log(2 * self.num_workers / self.delta)
        return 2 * self.sigma * np.sqrt(V / batch_size)


# Pre-defined configurations matching the paper (Table 2)
_CONFIGS = {
    "CartPole-v1": Config(
        env_name="CartPole-v1",
        max_episode_len=500,
        max_trajectories=5000,
        gamma=0.999,
        hidden_units=(16, 16),
        activation="ReLU",
        output_activation="Tanh",
        lr=1e-3,
        batch_size=16,
        batch_size_min=12,
        batch_size_max=20,
        mini_batch_size=4,
        svrpg_iterations=3,
        sigma=0.06,
        delta=0.6,
    ),
    "LunarLander-v2": Config(
        env_name="LunarLander-v2",
        max_episode_len=1000,
        max_trajectories=10000,
        gamma=0.99,
        hidden_units=(64, 64),
        activation="Tanh",
        output_activation="Tanh",
        lr=1e-3,
        batch_size=32,
        batch_size_min=26,
        batch_size_max=38,
        mini_batch_size=8,
        svrpg_iterations=3,
        sigma=0.07,
        delta=0.6,
    ),
    "HalfCheetah-v2": Config(
        env_name="HalfCheetah-v2",
        max_episode_len=500,
        max_trajectories=10000,
        gamma=0.995,
        hidden_units=(64, 64),
        activation="Tanh",
        output_activation="Tanh",
        lr=8e-5,
        batch_size=48,
        batch_size_min=46,
        batch_size_max=50,
        mini_batch_size=16,
        svrpg_iterations=3,
        sigma=0.9,
        delta=0.6,
    ),
}


def get_config(env_name: str, **overrides) -> Config:
    """
    Get configuration for environment with optional overrides.
    
    Args:
        env_name: Environment name
        **overrides: Override any config parameter
        
    Returns:
        Config instance
    """
    if env_name not in _CONFIGS:
        raise ValueError(f"Unknown environment: {env_name}. "
                        f"Available: {list(_CONFIGS.keys())}")
    
    # Copy base config
    base = _CONFIGS[env_name]
    config_dict = {
        f.name: getattr(base, f.name) 
        for f in base.__dataclass_fields__.values()
    }
    
    # Apply overrides
    config_dict.update(overrides)
    
    return Config(**config_dict)


def get_env_info(env_name: str) -> dict:
    """
    Get environment dimensions.
    
    Returns:
        dict with state_dim, action_dim, is_continuous
    """
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
        "is_continuous": is_continuous,
    }
