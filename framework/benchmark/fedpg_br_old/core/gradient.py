"""
Policy gradient computation for FedPG-BR.

Implements GPOMDP-style gradient estimation.
"""

import torch
from typing import Tuple, Optional

from fedpg_br.core.trajectory import Trajectory, compute_returns


def compute_log_probs(trajectory: Trajectory, policy, device: str = "cpu") -> torch.Tensor:
    """
    Compute log probabilities for trajectory actions under given policy.
    
    This is used for importance sampling - evaluating policy_0 on
    actions taken by policy_n.
    
    Args:
        trajectory: Trajectory with states and actions
        policy: Policy network to evaluate
        device: Computation device
        
    Returns:
        Tensor of log probabilities [trajectory_length]
    """
    log_probs = []
    
    for state, action in zip(trajectory.states, trajectory.actions):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
        _, log_prob = policy(state_tensor, sample=False, fixed_action=action)
        log_probs.append(log_prob)
    
    return torch.stack(log_probs)


def compute_policy_gradient(
    trajectory: Trajectory,
    policy,
    gamma: float,
    returns: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute policy gradient using GPOMDP estimator.
    
    g(τ|θ) = Σ_t (Σ_{t'≤t} ∇log π(a_t'|s_t')) * (γ^t * r_t - baseline)
    
    Args:
        trajectory: Trajectory data
        policy: Policy network
        gamma: Discount factor
        returns: Pre-computed returns (optional)
        device: Computation device
        
    Returns:
        (gradient_vector, log_probs)
    """
    if returns is None:
        returns = compute_returns(trajectory, gamma)
    
    # Compute log probabilities with gradient tracking
    log_probs = []
    for state, action in zip(trajectory.states, trajectory.actions):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
        _, log_prob = policy(state_tensor, sample=False, fixed_action=action)
        log_probs.append(log_prob)
    
    log_probs = torch.stack(log_probs)
    
    # Policy gradient loss: -E[log π(a|s) * A(s,a)]
    loss = -(log_probs * returns).mean()
    
    # Backpropagate to get gradient
    policy.zero_grad()
    loss.backward()
    
    # Extract flattened gradient
    gradient = torch.cat([p.grad.flatten().clone() for p in policy.parameters()])
    
    return gradient, log_probs
