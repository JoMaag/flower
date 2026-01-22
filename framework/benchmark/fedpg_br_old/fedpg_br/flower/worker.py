"""Worker (RL Agent) for Federated Learning."""

import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Discrete
from typing import List, Tuple, Optional

from fedpg_br.policy import MlpPolicy, DiagonalGaussianMlpPolicy


class Worker:
    """Federated RL Worker that samples trajectories and computes gradients."""
    
    def __init__(self, worker_id: int, env_name: str, hidden_units: Tuple[int, ...],
                 gamma: float, activation: str = "Tanh", output_activation: str = "Identity",
                 is_byzantine: bool = False, attack_type: Optional[str] = None,
                 max_episode_len: int = 1000, device: str = "cpu"):
        
        self.id = worker_id
        self.gamma = gamma
        self.is_byzantine = is_byzantine
        self.attack_type = attack_type
        self.max_episode_len = max_episode_len
        self.device = device
        
        self.env = gym.make(env_name)
        obs_dim = self.env.observation_space.shape[0]
        
        if isinstance(self.env.action_space, Discrete):
            action_dim = self.env.action_space.n
            self.policy = MlpPolicy([obs_dim] + list(hidden_units) + [action_dim], activation, output_activation)
        else:
            action_dim = self.env.action_space.shape[0]
            self.policy = DiagonalGaussianMlpPolicy([obs_dim] + list(hidden_units) + [action_dim], activation)
        
        self.policy.to(device)
    
    def load_parameters(self, state_dict: dict):
        self.policy.load_state_dict(state_dict)
    
    def compute_gradient(self, num_trajectories: int, sample: bool = True) -> Tuple[List[torch.Tensor], float, float, float]:
        all_advantages = []
        all_log_probs = []
        all_returns = []
        all_lengths = []
        
        while len(all_returns) < num_trajectories:
            reset_result = self.env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            
            episode_rewards = []
            episode_log_probs = []
            
            for _ in range(self.max_episode_len):
                state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
                
                if self.is_byzantine and self.attack_type == "random-action":
                    fixed_action = 0 if isinstance(self.env.action_space, Discrete) else np.zeros(self.env.action_space.shape[0], dtype=np.float32)
                    action, log_prob = self.policy(state_tensor, sample=sample, fixed_action=fixed_action)
                else:
                    action, log_prob = self.policy(state_tensor, sample=sample)
                
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    state, reward, done, _ = step_result
                
                if self.is_byzantine and self.attack_type == "reward-flipping":
                    reward = -reward
                
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob)
                
                if done:
                    break
            
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
        
        advantages = torch.tensor(all_advantages, dtype=torch.float32).to(self.device)
        log_probs = torch.stack(all_log_probs)
        
        loss = -(log_probs * advantages).mean()
        self.policy.zero_grad()
        loss.backward()
        
        gradients = []
        for param in self.policy.parameters():
            if not self.is_byzantine or self.attack_type is None:
                gradients.append(param.grad.clone())
            elif self.attack_type == "zero-gradient":
                gradients.append(torch.zeros_like(param.grad))
            elif self.attack_type == "random-noise":
                noise = (torch.rand_like(param.grad) * 2 - 1) * (param.grad.max() - param.grad.min()) * 3
                gradients.append(param.grad + noise)
            elif self.attack_type == "sign-flipping":
                gradients.append(-2.5 * param.grad)
            else:
                gradients.append(param.grad.clone())
        
        return gradients, loss.item(), np.mean(all_returns), np.mean(all_lengths)
    
    def evaluate(self, num_episodes: int = 10, max_steps: int = 1000) -> Tuple[float, float]:
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
