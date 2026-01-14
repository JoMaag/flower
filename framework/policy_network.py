# policy_network.py - PYTORCH VERSION
# pyright: reportGeneralTypeIssues=false
# type: ignore
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import numpy as np


class CategoricalPolicy(nn.Module):
    """
    Linear policy for discrete action spaces (like CartPole).
    π_θ(a|s) = softmax(W @ s + b)
    
    Uses PyTorch nn.Linear for automatic gradient computation.
    """
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Linear layer: y = Wx + b
        self.linear = nn.Linear(state_dim, action_dim)
        
        # Initialize like paper code
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights uniformly (like paper)."""
        for param in self.parameters():
            stdv = 1. / np.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    @property
    def param_size(self):
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, state, sample=True, fixed_action=None):
        """
        Compute action and log probability.
        
        Args:
            state: observation (numpy or torch tensor)
            sample: if True, sample action; else take argmax
            fixed_action: if given, compute log prob for this action
        
        Returns:
            action (int), log_prob (torch.Tensor)
        """
        # Convert to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        
        # Ensure 1D
        state = state.view(-1)
        
        # Compute logits
        logits = self.linear(state)
        
        # Create categorical distribution
        policy = Categorical(logits=logits)
        
        # Get action
        if fixed_action is not None:
            action = torch.tensor(fixed_action, dtype=torch.long)
        elif sample:
            action = policy.sample()
        else:
            action = policy.probs.argmax()
        
        # Compute log probability
        log_prob = policy.log_prob(action)
        
        return action.item(), log_prob
    
    def sample_action(self, state):
        """Sample action from π(·|state). Returns int."""
        with torch.no_grad():
            action, _ = self.forward(state, sample=True)
        return action
    
    def log_prob(self, state, action, theta=None):
        """
        Compute log π_θ(action|state).
        
        Args:
            state: observation
            action: action taken
            theta: optional parameters (if None, use current)
        
        Returns:
            log_prob (torch.Tensor)
        """
        if theta is not None:
            # Temporarily set weights
            old_params = self.get_weights()
            self.set_weights(theta)
        
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        
        state = state.view(-1)
        logits = self.linear(state)
        policy = Categorical(logits=logits)
        
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)
        
        log_prob = policy.log_prob(action)
        
        if theta is not None:
            # Restore weights
            self.set_weights(old_params)
        
        return log_prob
    
    def log_prob_gradient(self, state, action, theta):
        """
        Compute ∇_θ log π_θ(action|state).
        
        PyTorch computes this automatically via backprop!
        
        Returns:
            gradient (numpy array, flattened)
        """
        # Set parameters
        self.set_weights(theta)
        self.zero_grad()
        
        # Compute log prob
        log_p = self.log_prob(state, action)
        
        # Backprop to get gradient
        log_p.backward()
        
        # Extract gradient
        grad = torch.cat([
            p.grad.flatten() if p.grad is not None 
            else torch.zeros_like(p.flatten())
            for p in self.parameters()
        ])
        
        return grad.detach().numpy()
    
    def get_weights(self):
        """Get flattened parameters as numpy array."""
        return torch.cat([
            p.flatten() for p in self.parameters()
        ]).detach().numpy()
    
    def set_weights(self, theta):
        """Set parameters from flattened array."""
        if isinstance(theta, np.ndarray):
            theta = torch.from_numpy(theta).float()
        
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = theta[idx:idx+numel].view(p.shape)
            idx += numel


class GaussianPolicy(nn.Module):
    """
    Gaussian policy for continuous action spaces (like HalfCheetah).
    π_θ(a|s) = N(μ_θ(s), σ²)
    
    μ_θ(s) = W @ s + b
    """
    
    def __init__(self, state_dim, action_dim, log_std=-0.5):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Mean network
        self.linear = nn.Linear(state_dim, action_dim)
        
        # Fixed log std (not learned)
        self.log_std = torch.tensor(log_std)
        self.std = torch.exp(self.log_std)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights uniformly."""
        for param in self.parameters():
            stdv = 1. / np.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    @property
    def param_size(self):
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, state, sample=True, fixed_action=None):
        """
        Compute action and log probability.
        
        Returns:
            action (numpy array), log_prob (torch.Tensor)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        
        state = state.view(-1)
        
        # Compute mean
        mean = self.linear(state)
        
        # Create Gaussian distribution
        policy = Normal(mean, self.std)
        
        # Get action
        if fixed_action is not None:
            if not isinstance(fixed_action, torch.Tensor):
                action = torch.as_tensor(fixed_action, dtype=torch.float32)
            else:
                action = fixed_action
        elif sample:
            action = policy.sample()
        else:
            action = mean.detach()
        
        # Compute log probability
        log_prob = policy.log_prob(action).sum()  # Sum over action dims
        
        return action.detach().numpy(), log_prob
    
    def sample_action(self, state):
        """Sample action from π(·|state). Returns numpy array."""
        with torch.no_grad():
            action, _ = self.forward(state, sample=True)
        return action
    
    def log_prob(self, state, action, theta=None):
        """
        Compute log π_θ(action|state).
        
        Returns:
            log_prob (torch.Tensor, scalar)
        """
        if theta is not None:
            old_params = self.get_weights()
            self.set_weights(theta)
        
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        
        state = state.view(-1)
        mean = self.linear(state)
        policy = Normal(mean, self.std)
        log_prob = policy.log_prob(action).sum()
        
        if theta is not None:
            self.set_weights(old_params)
        
        return log_prob
    
    def log_prob_gradient(self, state, action, theta):
        """
        Compute ∇_θ log π_θ(action|state).
        
        Returns:
            gradient (numpy array, flattened)
        """
        self.set_weights(theta)
        self.zero_grad()
        
        log_p = self.log_prob(state, action)
        log_p.backward()
        
        grad = torch.cat([
            p.grad.flatten() if p.grad is not None 
            else torch.zeros_like(p.flatten())
            for p in self.parameters()
        ])
        
        return grad.detach().numpy()
    
    def get_weights(self):
        """Get flattened parameters as numpy array."""
        return torch.cat([
            p.flatten() for p in self.parameters()
        ]).detach().numpy()
    
    def set_weights(self, theta):
        """Set parameters from flattened array."""
        if isinstance(theta, np.ndarray):
            theta = torch.from_numpy(theta).float()
        
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = theta[idx:idx+numel].view(p.shape)
            idx += numel