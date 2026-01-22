"""
FedPG-BR: Fault-Tolerant Federated Reinforcement Learning

Based on: "Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee"
by Flint Xiaofeng Fan et al. (NeurIPS 2021)
"""

from fedpg_br.config import Config, get_config, ATTACK_TYPES
from fedpg_br.policy import MlpPolicy, DiagonalGaussianMlpPolicy, create_policy

__version__ = "0.1.0"
__all__ = [
    "Config",
    "get_config", 
    "ATTACK_TYPES",
    "MlpPolicy",
    "DiagonalGaussianMlpPolicy",
    "create_policy",
]
