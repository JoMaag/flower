"""
Worker (RL Agent) for Federated Learning.

Each worker:
- Has its own environment copy
- Samples trajectories
- Computes gradients
- Can be Byzantine (faulty/adversarial)
"""

import logging
import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Discrete
from typing import List, Tuple, Optional

from fedpg_br.policy import MlpPolicy, DiagonalGaussianMlpPolicy

logger = logging.getLogger(__name__)


class Worker:
    """
    Federated RL Worker/Agent.
    
    Samples trajectories and computes policy gradients.
    Can simulate Byzantine behavior with various attack types.
    """
    
    def __init__(
        self,
        worker_id: int,
        env_name: str,
        hidden_units: Tuple[int, ...],
        gamma: float,
        activation: str = "Tanh",
        output_activation: str = "Identity",
        is_byzantine: bool = False,
        attack_type: Optional[str] = None,
        max_episode_len: int = 1000,
        device: str = "cpu",
    ):
        """
        Args:
            worker_id: Unique identifier
            env_name: Gymnasium environment name
            hidden_units: Hidden layer sizes
            gamma: Discount factor
            activation: Activation function
            output_activation: Output activation (discrete only)
            is_byzantine: If True, apply attacks
            attack_type: Type of Byzantine attack
            max_episode_len: Max steps per episode
            device: Computation device
        """
        self.id = worker_id
        self.gamma = gamma
        self.is_byzantine = is_byzantine
        self.attack_type = attack_type
        self.max_episode_len = max_episode_len
        self.device = device
        
        # Create environment
        self.env = gym.make(env_name)
        self.env_name = env_name
        
        # Get dimensions
        obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            action_dim = self.env.action_space.n
            self.policy = MlpPolicy(
                sizes=[obs_dim] + list(hidden_units) + [action_dim],
                activation=activation,
                output_activation=output_activation,
            )
        else:
            action_dim = self.env.action_space.shape[0]
            self.policy = DiagonalGaussianMlpPolicy(
                sizes=[obs_dim] + list(hidden_units) + [action_dim],
                activation=activation,
            )
        
        self.policy.to(device)
    
    def load_parameters(self, state_dict: dict):
        """Load parameters from server."""
        self.policy.load_state_dict(state_dict)
    
    def collect_trajectories(
        self,
        num_trajectories: int,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[int]]:
        """
        Collect trajectories for training.
        
        Args:
            num_trajectories: Number of trajectories to collect (B)
            sample: If True, sample actions; else greedy
            
        Returns:
            (advantages, log_probs, episode_returns, episode_lengths)
        """
        all_advantages = []
        all_log_probs = []
        all_returns = []
        all_lengths = []
        
        while len(all_returns) < num_trajectories:
            # Reset environment
            reset_result = self.env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            
            episode_rewards = []
            episode_log_probs = []
            
            for _ in range(self.max_episode_len):
                # Get action
                state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
                
                if self.is_byzantine and self.attack_type == "random-action":
                    # Random action attack (simulates hardware failure)
                    if isinstance(self.env.action_space, Discrete):
                        fixed_action = 0
                    else:
                        fixed_action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
                    action, log_prob = self.policy(state_tensor, sample=sample, fixed_action=fixed_action)
                else:
                    action, log_prob = self.policy(state_tensor, sample=sample)
                
                # Step
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    state, reward, done, _ = step_result
                
                # Reward attacks
                if self.is_byzantine and self.attack_type == "reward-flipping":
                    reward = -reward
                
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob)
                
                if done:
                    break
            
            # Compute returns
            if self.is_byzantine and self.attack_type == "random-reward":
                random.shuffle(episode_rewards)
            
            returns = []
            R = 0.0
            for r in reversed(episode_rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            
            returns = torch.tensor(returns, dtype=torch.float32)
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            all_advantages.extend(advantages.tolist())
            all_log_probs.extend(episode_log_probs)
            all_returns.append(sum(episode_rewards))
            all_lengths.append(len(episode_rewards))
        
        return (
            torch.tensor(all_advantages, dtype=torch.float32).to(self.device),
            torch.stack(all_log_probs),
            all_returns,
            all_lengths,
        )
    
    def compute_gradient(
        self,
        num_trajectories: int,
        sample: bool = True,
    ) -> Tuple[List[torch.Tensor], float, float, float]:
        """
        Compute policy gradient from B trajectories.
        
        Args:
            num_trajectories: Number of trajectories (B)
            sample: If True, sample actions
            
        Returns:
            (gradient_list, loss, avg_return, avg_length)
        """
        advantages, log_probs, returns, lengths = self.collect_trajectories(
            num_trajectories, sample
        )
        
        # Policy gradient loss
        loss = -(log_probs * advantages).mean()
        
        # Compute gradient
        self.policy.zero_grad()
        loss.backward()
        
        # Apply gradient attacks
        gradients = []
        for param in self.policy.parameters():
            if not self.is_byzantine or self.attack_type is None:
                gradients.append(param.grad.clone())
            elif self.attack_type == "zero-gradient":
                gradients.append(torch.zeros_like(param.grad))
            elif self.attack_type == "random-noise":
                noise_scale = (param.grad.max() - param.grad.min()) * 3
                noise = (torch.rand_like(param.grad) * 2 - 1) * noise_scale
                gradients.append(param.grad + noise)
            elif self.attack_type == "sign-flipping":
                gradients.append(-2.5 * param.grad)
            else:
                # reward-flipping, random-action, random-reward handled during collection
                gradients.append(param.grad.clone())
        
        return gradients, loss.item(), np.mean(returns), np.mean(lengths)
    
    def evaluate(
        self,
        num_episodes: int = 10,
        max_steps: int = 1000,
    ) -> Tuple[float, float]:
        """
        Evaluate current policy.
        
        Returns:
            (total_reward, episode_length)
        """
        total_reward = 0.0
        total_length = 0
        
        for _ in range(num_episodes):
            reset_result = self.env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            episode_reward = 0.0
            
            for step in range(max_steps):
                state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    action, _ = self.policy(state_tensor, sample=False)
                
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    state, reward, done, _ = step_result
                
                episode_reward += reward
                
                if done:
                    break
            
            total_reward += episode_reward
            total_length += step + 1
        
        return total_reward / num_episodes, total_length / num_episodes
    
    def to(self, device: str) -> "Worker":
        """Move to device."""
        self.device = device
        self.policy.to(device)
        return self
