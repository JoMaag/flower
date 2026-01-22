"""
Trajectory sampling and return computation for policy gradient.
"""

import torch
import numpy as np
import gymnasium as gym
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Trajectory:
    """Container for a single trajectory."""
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    
    def add(self, state: np.ndarray, action: int, reward: float, log_prob: float):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def __len__(self) -> int:
        return len(self.rewards)
    
    @property
    def total_reward(self) -> float:
        return sum(self.rewards)


def sample_trajectory(
    env: gym.Env,
    policy,
    max_steps: int = 1000,
    sample: bool = True,
    device: str = "cpu",
) -> Tuple[Trajectory, float]:
    """
    Sample a single trajectory using the policy.
    
    Args:
        env: Gymnasium environment
        policy: Policy network (MlpPolicy or DiagonalGaussianMlpPolicy)
        max_steps: Maximum steps per episode
        sample: If True, sample actions; else greedy
        device: Computation device
        
    Returns:
        (trajectory, total_reward)
    """
    trajectory = Trajectory()
    
    # Handle both old and new gym API
    reset_result = env.reset()
    state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    
    for _ in range(max_steps):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            action, log_prob = policy(state_tensor, sample=sample)
        
        # Handle log_prob tensor
        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.item()
        
        # Step environment
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_result
        
        trajectory.add(state, action, reward, log_prob)
        state = next_state
        
        if done:
            break
    
    return trajectory, trajectory.total_reward


def compute_returns(
    trajectory: Trajectory,
    gamma: float,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute discounted returns (reward-to-go).
    
    Args:
        trajectory: Trajectory with rewards
        gamma: Discount factor
        normalize: If True, normalize returns (advantage estimation)
        
    Returns:
        Tensor of returns for each timestep
    """
    returns = []
    R = 0.0
    
    # Compute returns in reverse
    for reward in reversed(trajectory.rewards):
        R = reward + gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns, dtype=torch.float32)
    
    if normalize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns
