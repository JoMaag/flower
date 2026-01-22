"""
Policy Networks for FedPG-BR

- MlpPolicy: Categorical policy for discrete actions (CartPole, LunarLander)
- DiagonalGaussianMlpPolicy: Gaussian policy for continuous actions (HalfCheetah)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from typing import List, Tuple, Optional, Union


def _build_mlp(
    sizes: List[int], 
    activation: type = nn.Tanh, 
    output_activation: type = nn.Identity
) -> nn.Sequential:
    """Build a feedforward neural network."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(act())
    return nn.Sequential(*layers)


def _get_activation(name: str) -> type:
    """Get activation class by name."""
    activations = {
        "Tanh": nn.Tanh,
        "ReLU": nn.ReLU,
        "Identity": nn.Identity,
        "Softmax": nn.Softmax,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name]


class MlpPolicy(nn.Module):
    """
    Categorical MLP Policy for discrete action spaces.
    
    Used for: CartPole-v1, LunarLander-v2
    """
    
    def __init__(
        self,
        sizes: List[int],
        activation: str = "Tanh",
        output_activation: str = "Identity",
    ):
        super().__init__()
        self.sizes = sizes
        self.network = _build_mlp(
            sizes,
            _get_activation(activation),
            _get_activation(output_activation),
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with uniform distribution (matches original)."""
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    def forward(
        self, 
        obs: torch.Tensor, 
        sample: bool = True, 
        fixed_action: Optional[int] = None
    ) -> Tuple[int, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor
            sample: If True, sample action; else argmax
            fixed_action: Use this action (for importance sampling)
            
        Returns:
            (action, log_prob)
        """
        obs = obs.view(-1)
        logits = self.network(obs)
        dist = Categorical(logits=logits)
        
        if fixed_action is not None:
            action = torch.tensor(fixed_action, device=obs.device)
        elif sample:
            action = dist.sample()
        else:
            action = dist.probs.argmax()
        
        return action.item(), dist.log_prob(action)
    
    def get_flat_params(self) -> torch.Tensor:
        """Get flattened parameters."""
        return torch.cat([p.flatten() for p in self.parameters()])
    
    def set_flat_params(self, params: torch.Tensor):
        """Set parameters from flattened tensor."""
        offset = 0
        for p in self.parameters():
            size = p.numel()
            p.data = params[offset:offset + size].view(p.shape)
            offset += size


class DiagonalGaussianMlpPolicy(nn.Module):
    """
    Diagonal Gaussian MLP Policy for continuous action spaces.
    
    Used for: HalfCheetah-v2
    """
    
    LOG_SIGMA_MIN = -20
    LOG_SIGMA_MAX = -2
    
    def __init__(
        self,
        sizes: List[int],
        activation: str = "Tanh",
        gear: float = 1.0,  # Action scaling
    ):
        super().__init__()
        self.sizes = sizes
        self.gear = gear
        
        # Shared feature extractor (all but last layer)
        self.features = _build_mlp(
            sizes[:-1], 
            _get_activation(activation), 
            nn.Identity
        )
        
        # Separate heads for mean and log_std
        self.mu_head = nn.Linear(sizes[-2], sizes[-1], bias=False)
        self.log_sigma_head = nn.Linear(sizes[-2], sizes[-1], bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with uniform distribution."""
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    def forward(
        self,
        obs: torch.Tensor,
        sample: bool = True,
        fixed_action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor
            sample: If True, sample action; else use mean
            fixed_action: Use this action (for importance sampling)
            
        Returns:
            (action, log_prob)
        """
        features = self.features(obs)
        
        mu = torch.tanh(self.mu_head(features)) * self.gear
        log_sigma = torch.clamp(
            self.log_sigma_head(features),
            self.LOG_SIGMA_MIN,
            self.LOG_SIGMA_MAX
        )
        sigma = torch.tanh(log_sigma.exp())
        
        dist = Normal(mu, sigma)
        
        if fixed_action is not None:
            action = torch.tensor(fixed_action, device=obs.device)
        elif sample:
            action = dist.sample()
        else:
            action = mu.detach()
        
        log_prob = dist.log_prob(action)
        log_prob = torch.clamp(log_prob, min=-1e5)  # Numerical stability
        
        return action.numpy(), log_prob.sum()
    
    def get_flat_params(self) -> torch.Tensor:
        """Get flattened parameters."""
        return torch.cat([p.flatten() for p in self.parameters()])
    
    def set_flat_params(self, params: torch.Tensor):
        """Set parameters from flattened tensor."""
        offset = 0
        for p in self.parameters():
            size = p.numel()
            p.data = params[offset:offset + size].view(p.shape)
            offset += size


def create_policy(
    state_dim: int,
    action_dim: int,
    env_name: str,
    hidden_units: Tuple[int, ...] = (64, 64),
    activation: str = "Tanh",
    output_activation: str = "Identity",
) -> Union[MlpPolicy, DiagonalGaussianMlpPolicy]:
    """
    Factory function to create appropriate policy for environment.
    
    Args:
        state_dim: Observation dimension
        action_dim: Action dimension
        env_name: Environment name
        hidden_units: Hidden layer sizes
        activation: Activation function
        output_activation: Output activation (discrete only)
        
    Returns:
        Policy network
    """
    sizes = [state_dim] + list(hidden_units) + [action_dim]
    
    continuous_envs = {"HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2"}
    
    if env_name in continuous_envs:
        return DiagonalGaussianMlpPolicy(sizes=sizes, activation=activation)
    else:
        return MlpPolicy(
            sizes=sizes, 
            activation=activation, 
            output_activation=output_activation
        )
