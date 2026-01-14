"""
Policy Networks from Original Implementation
Based on policy.py from: https://github.com/flint-xf-fan/Byzantine-Federated-RL

Key differences from my initial implementation:
1. Proper initialization (uniform distribution based on param size)
2. DiagonalGaussianMlpPolicy for continuous actions (HalfCheetah)
3. Categorical policy for discrete actions (CartPole, LunarLander)
"""

import numpy as np
import torch
import torch.nn as nn
import math
from torch.distributions.categorical import Categorical
from torch.distributions import Normal


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    """Build a feedforward neural network"""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MlpPolicy(nn.Module):
    """
    Categorical MLP Policy for discrete action spaces
    Used for: CartPole-v1, LunarLander-v2
    """
    def __init__(
        self,
        sizes,
        activation='Tanh',
        output_activation='Identity'
    ):
        super(MlpPolicy, self).__init__()
        
        # Store parameters
        self.activation_name = activation
        self.output_activation_name = output_activation
        
        # Set activation functions
        if activation == 'Tanh':
            self.activation = nn.Tanh
        elif activation == 'ReLU':
            self.activation = nn.ReLU
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")
            
        if output_activation == 'Identity':
            self.output_activation = nn.Identity
        elif output_activation == 'Tanh':
            self.output_activation = nn.Tanh
        elif output_activation == 'ReLU':
            self.output_activation = nn.ReLU
        elif output_activation == 'Softmax':
            self.output_activation = nn.Softmax
        else:
            raise NotImplementedError(f"Output activation {output_activation} not implemented")
            
        # Make policy network
        self.sizes = sizes
        self.logits_net = mlp(self.sizes, self.activation, self.output_activation)
    
        # Init parameters (IMPORTANT: uses uniform initialization like original)
        self.init_parameters()

    def init_parameters(self):
        """Initialize parameters with uniform distribution (from original)"""
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, obs, sample=True, fixed_action=None):
        """
        Forward pass through policy
        
        Args:
            obs: observation tensor
            sample: if True, sample from distribution; else take argmax
            fixed_action: if provided, use this action (for importance sampling)
            
        Returns:
            action (int), log_prob(action)
        """
        obs = obs.view(-1)
        
        # Forward pass the policy net
        logits = self.logits_net(obs)
        
        # Get the policy distribution
        policy = Categorical(logits=logits)
        
        # Take the pre-set action if given (for importance sampling)
        if fixed_action is not None:
            action = torch.tensor(fixed_action, device=obs.device)
        
        # Take random action (sampling)
        elif sample:
            try:
                action = policy.sample()
            except Exception as e:
                print(f"Sampling error: {e}")
                print(f"Logits: {logits}, Obs: {obs}")
                raise
                
        # Take greedy action
        else:
            action = policy.probs.argmax()
        
        return action.item(), policy.log_prob(action)
    
    def get_parameters(self):
        """Get flattened parameters"""
        return torch.cat([p.flatten() for p in self.parameters()])
    
    def set_parameters(self, params):
        """Set parameters from flattened tensor"""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = params[offset:offset + numel].view(p.shape)
            offset += numel


class DiagonalGaussianMlpPolicy(nn.Module):
    """
    Diagonal Gaussian MLP Policy for continuous action spaces
    Used for: HalfCheetah-v2
    """
    def __init__(
        self,
        sizes,
        activation='Tanh',
        output_activation='Tanh',
        geer=1  # Gear ratio for action scaling
    ):
        super(DiagonalGaussianMlpPolicy, self).__init__()

        # Store parameters
        self.activation_name = activation
        self.output_activation_name = output_activation

        if activation == 'Tanh':
            self.activation = nn.Tanh
        elif activation == 'ReLU':
            self.activation = nn.ReLU
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

        # Make policy network
        self.sizes = sizes
        self.geer = geer
        
        # Shared feature extractor
        self.logits_net = mlp(self.sizes[:-1], self.activation, nn.Identity)
        
        # Separate heads for mean and log_std
        self.mu_net = nn.Linear(self.sizes[-2], self.sizes[-1], bias=False)
        self.log_sigma_net = nn.Linear(self.sizes[-2], self.sizes[-1], bias=False)
        
        # Clipping bounds for log_sigma (stability)
        self.LOG_SIGMA_MIN = -20
        self.LOG_SIGMA_MAX = -2
        
        # Init parameters
        self.init_parameters()

    def init_parameters(self):
        """Initialize parameters with uniform distribution"""
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, obs, sample=True, fixed_action=None):
        """
        Forward pass through policy
        
        Args:
            obs: observation tensor
            sample: if True, sample from distribution; else use mean
            fixed_action: if provided, use this action (for importance sampling)
            
        Returns:
            action (numpy array), log_prob(action)
        """
        # Forward pass the policy net
        logits = self.logits_net(obs)

        # Get the mean (mu)
        mu = torch.tanh(self.mu_net(logits)) * self.geer

        # Get the std (sigma)
        log_sigma = torch.clamp(
            self.log_sigma_net(logits), 
            self.LOG_SIGMA_MIN, 
            self.LOG_SIGMA_MAX
        )
        sigma = torch.tanh(log_sigma.exp())

        # Get the policy distribution
        policy = Normal(mu, sigma)

        # Take the pre-set action (for importance sampling)
        if fixed_action is not None:
            action = torch.tensor(fixed_action, device=obs.device)
        else:
            if sample:
                action = policy.sample()
            else:
                action = mu.detach()
        
        # Compute log probability
        ll = policy.log_prob(action)
        
        # Avoid NaN (clamp very negative log probs)
        ll[ll < -1e5] = -1e5
        
        return action.numpy(), ll.sum()
    
    def get_parameters(self):
        """Get flattened parameters"""
        return torch.cat([p.flatten() for p in self.parameters()])
    
    def set_parameters(self, params):
        """Set parameters from flattened tensor"""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = params[offset:offset + numel].view(p.shape)
            offset += numel


def create_policy(
    state_dim: int,
    action_dim: int,
    env_name: str,
    hidden_units=(64, 64),
    activation='Tanh',
    output_activation='Identity'
):
    """
    Factory function to create appropriate policy for environment
    
    Args:
        state_dim: Dimension of observation space
        action_dim: Dimension of action space
        env_name: Environment name
        hidden_units: Tuple of hidden layer sizes
        activation: Activation function name
        output_activation: Output activation function name
        
    Returns:
        Policy network (MlpPolicy or DiagonalGaussianMlpPolicy)
    """
    sizes = [state_dim] + list(hidden_units) + [action_dim]
    
    # Continuous action space environments
    if env_name in ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']:
        print(f"Creating DiagonalGaussianMlpPolicy for {env_name}")
        return DiagonalGaussianMlpPolicy(
            sizes=sizes,
            activation=activation,
            output_activation=output_activation,
            geer=1  # Can be tuned per environment
        )
    
    # Discrete action space environments
    elif env_name in ['CartPole-v1', 'LunarLander-v2']:
        print(f"Creating MlpPolicy (Categorical) for {env_name}")
        return MlpPolicy(
            sizes=sizes,
            activation=activation,
            output_activation=output_activation
        )
    
    else:
        raise ValueError(f"Unknown environment: {env_name}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("POLICY NETWORKS - Original Implementation")
    print("="*80)
    
    # Test Categorical Policy (CartPole)
    print("\n1. Categorical Policy (CartPole-v1):")
    print("-"*40)
    cartpole_policy = create_policy(
        state_dim=4,
        action_dim=2,
        env_name='CartPole-v1',
        hidden_units=(16, 16),
        activation='ReLU',
        output_activation='Tanh'
    )
    print(f"Network: {cartpole_policy.sizes}")
    print(f"Parameters: {sum(p.numel() for p in cartpole_policy.parameters())}")
    
    # Test forward pass
    obs = torch.randn(4)
    action, log_prob = cartpole_policy(obs, sample=True)
    print(f"Sample action: {action}, log_prob: {log_prob.item():.4f}")
    
    # Test Gaussian Policy (HalfCheetah)
    print("\n2. Gaussian Policy (HalfCheetah-v2):")
    print("-"*40)
    cheetah_policy = create_policy(
        state_dim=17,
        action_dim=6,
        env_name='HalfCheetah-v2',
        hidden_units=(64, 64),
        activation='Tanh',
        output_activation='Tanh'
    )
    print(f"Network: {cheetah_policy.sizes}")
    print(f"Parameters: {sum(p.numel() for p in cheetah_policy.parameters())}")
    
    # Test forward pass
    obs = torch.randn(17)
    action, log_prob = cheetah_policy(obs, sample=True)
    print(f"Sample action shape: {action.shape}, log_prob: {log_prob.item():.4f}")
    
    print("\n" + "="*80)
