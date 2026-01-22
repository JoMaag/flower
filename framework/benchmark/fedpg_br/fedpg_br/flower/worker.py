"""Worker (RL Agent) for Federated Learning."""

import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Discrete
from typing import List, Tuple, Optional

from fedpg_br.policy import MlpPolicy, DiagonalGaussianMlpPolicy
from fedpg_br.core.attacks import (
    apply_attack, reset_attack_state, AttackConfig,
    ATTACK_REGISTRY, PAPER_ATTACK_TYPES
)


class Worker:
    """Federated RL Worker that samples trajectories and computes gradients.
    
    Implements the agent behavior from FedPG-BR paper, including
    Byzantine attack simulation for testing fault tolerance.
    """
    
    def __init__(self, worker_id: int, env_name: str, hidden_units: Tuple[int, ...],
                 gamma: float, activation: str = "Tanh", output_activation: str = "Identity",
                 is_byzantine: bool = False, attack_type: Optional[str] = None,
                 max_episode_len: int = 1000, device: str = "cpu",
                 attack_config: Optional[AttackConfig] = None):
        
        self.id = worker_id
        self.gamma = gamma
        self.is_byzantine = is_byzantine
        self.attack_type = attack_type
        self.max_episode_len = max_episode_len
        self.device = device
        self.attack_config = attack_config or AttackConfig()
        
        # Create environment
        self.env = gym.make(env_name)
        obs_dim = self.env.observation_space.shape[0]
        
        # Create policy based on action space type
        if isinstance(self.env.action_space, Discrete):
            action_dim = self.env.action_space.n
            self.policy = MlpPolicy(
                [obs_dim] + list(hidden_units) + [action_dim], 
                activation, 
                output_activation
            )
            self.is_discrete = True
        else:
            action_dim = self.env.action_space.shape[0]
            self.policy = DiagonalGaussianMlpPolicy(
                [obs_dim] + list(hidden_units) + [action_dim], 
                activation
            )
            self.is_discrete = False
        
        self.policy.to(device)
        
        # Log Byzantine status
        if self.is_byzantine:
            print(f"Worker {worker_id}: Byzantine agent with attack '{attack_type}'")
    
    def _sample_action(self, state_tensor: torch.Tensor, sample: bool = True) -> Tuple[any, torch.Tensor]:
        """Sample action, potentially using random action for RA attack."""
        
        # Random Action (RA) attack: ignore policy, take random actions
        if self.is_byzantine and self.attack_type == "random-action":
            # Get random action from environment
            random_action = self.env.action_space.sample()
            
            # Still need to get log_prob from policy for gradient computation
            action, log_prob = self.policy(state_tensor, sample=sample)
            return random_action, log_prob
        
        # Normal policy sampling
        action, log_prob = self.policy(state_tensor, sample=sample)
        return action, log_prob
    
    def _apply_reward_attack(self, rewards: List[float]) -> List[float]:
        """Apply reward-based attacks if applicable."""
        if not self.is_byzantine:
            return rewards
        
        if self.attack_type == "reward-flipping":
            return [-r for r in rewards]
        
        return rewards
    
    def compute_gradient(self, num_trajectories: int, sample: bool = True) -> Tuple[List[torch.Tensor], float, float, float]:
        """Sample trajectories and compute policy gradient.
        
        Args:
            num_trajectories: Number of trajectories (episodes) to sample
            sample: Whether to sample stochastically from policy
            
        Returns:
            Tuple of (gradients, loss, avg_return, avg_length)
        """
        all_advantages = []
        all_log_probs = []
        all_returns = []
        all_lengths = []
        
        # Sample trajectories
        while len(all_returns) < num_trajectories:
            reset_result = self.env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            
            episode_rewards = []
            episode_log_probs = []
            
            for step in range(self.max_episode_len):
                state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
                
                # Get action (potentially random for RA attack)
                action, log_prob = self._sample_action(state_tensor, sample=sample)
                
                # Environment step
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    state, reward, done, _ = step_result
                
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob)
                
                if done:
                    break
            
            # Apply reward-based attacks
            episode_rewards = self._apply_reward_attack(episode_rewards)
            
            # Compute returns (discounted cumulative rewards)
            returns = []
            R = 0.0
            for r in reversed(episode_rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            
            # Normalize advantages
            returns = torch.tensor(returns, dtype=torch.float32)
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            all_advantages.extend(advantages.tolist())
            all_log_probs.extend(episode_log_probs)
            all_returns.append(sum(episode_rewards))
            all_lengths.append(len(episode_rewards))
        
        # Compute policy gradient loss
        advantages = torch.tensor(all_advantages, dtype=torch.float32).to(self.device)
        log_probs = torch.stack(all_log_probs)
        
        # REINFORCE loss: -E[log π(a|s) * A(s,a)]
        loss = -(log_probs * advantages).mean()
        
        # Backpropagate to compute gradients
        self.policy.zero_grad()
        loss.backward()
        
        # Extract gradients
        gradients = [param.grad.clone() for param in self.policy.parameters()]
        
        # Apply gradient-based Byzantine attacks
        if self.is_byzantine and self.attack_type is not None:
            if self.attack_type not in ["random-action", "reward-flipping"]:
                # These attacks modify gradients directly
                gradients = apply_attack(
                    self.attack_type, 
                    gradients, 
                    worker_id=self.id,
                    config=self.attack_config
                )
        
        return gradients, loss.item(), np.mean(all_returns), np.mean(all_lengths)
    
    def evaluate(self, num_episodes: int = 10, max_steps: int = 1000) -> Tuple[float, float]:
        """Evaluate current policy without exploration.
        
        Args:
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (average_reward, average_length)
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
    
    def close(self):
        """Clean up environment."""
        self.env.close()
