"""
Configuration for FedPG-BR.

Environment-specific hyperparameters from the original paper (Table 2).
"""

from dataclasses import dataclass
from typing import Tuple
import gymnasium as gym


ATTACK_TYPES = [
    "zero-gradient",
    "random-action", 
    "sign-flipping",
    "reward-flipping",
    "random-reward",
    "random-noise",
    "FedScsPG-attack",
]


@dataclass
class Config:
    """Configuration for FedPG-BR algorithm."""
    
    env_name: str = "CartPole-v1"
    max_episode_len: int = 500
    gamma: float = 0.999
    
    # Network
    hidden_units: Tuple[int, ...] = (16, 16)
    activation: str = "ReLU"
    output_activation: str = "Tanh"
    
    # Training
    lr: float = 1e-3
    batch_size: int = 16
    batch_size_min: int = 12
    batch_size_max: int = 20
    mini_batch_size: int = 4
    
    # Byzantine filtering
    sigma: float = 0.06
    delta: float = 0.6
    
    @property
    def batch_size_range(self) -> Tuple[int, int]:
        return (self.batch_size_min, self.batch_size_max)


_CONFIGS = {
    "CartPole-v1": Config(
        env_name="CartPole-v1",
        max_episode_len=500,
        gamma=0.999,
        hidden_units=(16, 16),
        activation="ReLU",
        output_activation="Tanh",
        lr=1e-3,
        batch_size=16,
        batch_size_min=12,
        batch_size_max=20,
        mini_batch_size=4,
        sigma=0.06,
        delta=0.6,
    ),
    "LunarLander-v2": Config(
        env_name="LunarLander-v2",
        max_episode_len=1000,
        gamma=0.99,
        hidden_units=(64, 64),
        activation="Tanh",
        output_activation="Tanh",
        lr=1e-3,
        batch_size=32,
        batch_size_min=26,
        batch_size_max=38,
        mini_batch_size=8,
        sigma=0.07,
        delta=0.6,
    ),
    "HalfCheetah-v2": Config(
        env_name="HalfCheetah-v2",
        max_episode_len=500,
        gamma=0.995,
        hidden_units=(64, 64),
        activation="Tanh",
        output_activation="Tanh",
        lr=8e-5,
        batch_size=48,
        batch_size_min=46,
        batch_size_max=50,
        mini_batch_size=16,
        sigma=0.9,
        delta=0.6,
    ),
}


def get_config(env_name: str) -> Config:
    """Get configuration for environment."""
    if env_name not in _CONFIGS:
        raise ValueError(f"Unknown environment: {env_name}")
    return _CONFIGS[env_name]


def get_env_info(env_name: str) -> dict:
    """Get environment dimensions."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        is_continuous = False
    else:
        action_dim = env.action_space.shape[0]
        is_continuous = True
    
    env.close()
    return {"state_dim": state_dim, "action_dim": action_dim, "is_continuous": is_continuous}
