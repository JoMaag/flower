#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportGeneralTypeIssues=false
# type: ignore
"""
Worker Implementation for FedPG-BR Flower
Based on worker.py from original implementation

Worker = Individual RL Agent that:
- Samples trajectories from environment
- Computes gradients
- Can be Byzantine (faulty/adversarial)
"""

import torch
import numpy as np
import gymnasium as gym
from gym.spaces import Discrete
import random

from original_policy import MlpPolicy, DiagonalGaussianMlpPolicy
from utils_flower import get_inner_model, save_frames_as_gif, env_wrapper


class Worker:
    """
    Worker/Agent for Federated RL
    
    Each worker:
    - Has its own copy of the environment
    - Samples trajectories using current policy
    - Computes GPOMDP gradients
    - Can be Byzantine (attacks)
    """
    
    def __init__(
        self,
        id: int,
        is_byzantine: bool,
        env_name: str,
        hidden_units: str,
        gamma: float,
        activation: str = 'Tanh',
        output_activation: str = 'Identity',
        attack_type: str = None,
        max_epi_len: int = 1000,
        device: str = 'cpu'
    ):
        """
        Initialize Worker
        
        Args:
            id: Worker ID
            is_byzantine: If True, this is a Byzantine agent
            env_name: Environment name
            hidden_units: Hidden layer sizes (e.g., '64,64')
            gamma: Discount factor
            activation: Activation function
            output_activation: Output activation
            attack_type: Type of Byzantine attack
            max_epi_len: Maximum episode length
            device: Device for computation
        """
        super(Worker, self).__init__()
        
        # Setup
        self.id = id
        self.is_byzantine = is_byzantine
        self.gamma = gamma
        self.attack_type = attack_type
        self.max_epi_len = max_epi_len
        self.device = device
        
        # Make environment
        self.env_name = env_name
        self.env = gym.make(env_name)
        
        # Get dimensions
        obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            n_acts = self.env.action_space.n
        else:
            n_acts = self.env.action_space.shape[0]
        
        # Build network architecture
        hidden_sizes = list(eval(hidden_units))
        self.sizes = [obs_dim] + hidden_sizes + [n_acts]
        
        # Get policy network
        if isinstance(self.env.action_space, Discrete):
            self.logits_net = MlpPolicy(
                self.sizes, 
                activation, 
                output_activation
            )
        else:
            self.logits_net = DiagonalGaussianMlpPolicy(
                self.sizes, 
                activation
            )
        
        # Print architecture for first worker
        if self.id == 1:
            print(f"\nWorker {id} Policy Architecture:")
            print(self.logits_net)
            print(f"Total parameters: {sum(p.numel() for p in self.logits_net.parameters())}\n")
    
    def load_param_from_master(self, param):
        """
        Load parameters from master/server
        
        Args:
            param: State dict from master
        """
        model_actor = get_inner_model(self.logits_net)
        model_actor.load_state_dict({**model_actor.state_dict(), **param})
    
    def rollout(
        self, 
        device: str, 
        max_steps: int = 1000, 
        render: bool = False, 
        env=None, 
        obs=None, 
        sample: bool = True, 
        mode: str = 'human', 
        save_dir: str = './', 
        filename: str = '.'
    ):
        """
        Execute one episode rollout
        
        Args:
            device: Device for computation
            max_steps: Maximum steps
            render: If True, render environment
            env: Environment (if None, use self.env)
            obs: Initial observation (if None, reset env)
            sample: If True, sample actions; else greedy
            mode: Render mode ('human' or 'rgb')
            save_dir: Directory to save GIF
            filename: Filename for GIF
            
        Returns:
            (episode_reward, episode_length, rewards_list)
        """
        if env is None and obs is None:
            env = self.env
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
        
        done = False
        ep_rew = []
        frames = []
        step = 0
        
        while not done and step < max_steps:
            step += 1
            
            if render:
                if mode == 'rgb':
                    frames.append(env.render(mode="rgb_array"))
                else:
                    env.render()
            
            # Get action from policy
            obs = env_wrapper(env.unwrapped.spec.id, obs)
            action = self.logits_net(
                torch.as_tensor(obs, dtype=torch.float32).to(device), 
                sample=sample
            )[0]
            
            # Handle both old and new gym API
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, rew, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs, rew, done, _ = step_result
            
            ep_rew.append(rew)
        
        if mode == 'rgb':
            save_frames_as_gif(frames, save_dir, filename)
        
        return np.sum(ep_rew), len(ep_rew), ep_rew
    
    def collect_experience_for_training(
        self, 
        B: int, 
        device: str, 
        record: bool = False, 
        sample: bool = True, 
        attack_type: str = None
    ):
        """
        Collect B trajectories for training
        
        Args:
            B: Number of trajectories to collect
            device: Device for computation
            record: If True, record states and actions
            sample: If True, sample actions; else greedy
            attack_type: Attack type (overrides self.attack_type)
            
        Returns:
            If record=False: (weights, log_probs, returns, lengths)
            If record=True: (weights, log_probs, returns, lengths, states, actions)
        """
        # Lists for logging
        batch_weights = []      # R(tau) weighting for policy gradient
        batch_rets = []         # Episode returns
        batch_lens = []         # Episode lengths
        batch_log_prob = []     # Log probabilities
        
        # Recording lists
        if record:
            batch_states = []
            batch_actions = []
        
        # Reset episode variables
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        
        done = False
        ep_rews = []
        t = 1
        
        # Override attack type if provided
        if attack_type is None:
            attack_type = self.attack_type
        
        # Collect experience
        while True:
            # Record state
            if record:
                batch_states.append(obs)
            
            # Wrap observation
            obs = env_wrapper(self.env_name, obs)
            
            # Get action - with attack if Byzantine
            if self.is_byzantine and attack_type == 'random-action':
                # Random action attack (hardware failure simulation)
                act_rnd = self.env.action_space.sample()
                if isinstance(act_rnd, int):  # Discrete
                    act_rnd = 0
                else:  # Continuous
                    act_rnd = np.zeros(
                        len(self.env.action_space.sample()), 
                        dtype=np.float32
                    )
                act, log_prob = self.logits_net(
                    torch.as_tensor(obs, dtype=torch.float32).to(device), 
                    sample=sample, 
                    fixed_action=act_rnd
                )
            else:
                # Normal action
                act, log_prob = self.logits_net(
                    torch.as_tensor(obs, dtype=torch.float32).to(device), 
                    sample=sample
                )
            
            # Take step
            step_result = self.env.step(act)
            if len(step_result) == 5:
                obs, rew, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, rew, done, info = step_result
            
            # Reward attack
            if self.is_byzantine and attack_type == 'reward-flipping':
                rew = -rew
            
            # Increment timestep
            t += 1
            
            # Save log prob and reward
            batch_log_prob.append(log_prob)
            ep_rews.append(rew)
            
            # Record action
            if record:
                batch_actions.append(act)
            
            # Episode done
            if done or len(ep_rews) >= self.max_epi_len:
                # Record episode stats
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                
                # Compute returns (discounted rewards)
                returns = []
                R = 0
                
                # Random reward attack
                if self.is_byzantine and attack_type == 'random-reward':
                    random.shuffle(ep_rews)
                
                # Compute discounted returns
                for r in ep_rews[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)
                
                returns = torch.tensor(returns, dtype=torch.float32)
                
                # Advantage (return whitening)
                advantage = (returns - returns.mean()) / (returns.std() + 1e-20)
                batch_weights += advantage
                
                # Check if we have enough trajectories
                if len(batch_lens) >= B:
                    break
                
                # Reset for next episode
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result
                else:
                    obs = reset_result
                done, ep_rews, t = False, [], 1
        
        # Convert to tensors
        weights = torch.as_tensor(batch_weights, dtype=torch.float32).to(device)
        logp = torch.stack(batch_log_prob)
        
        if record:
            return weights, logp, batch_rets, batch_lens, batch_states, batch_actions
        else:
            return weights, logp, batch_rets, batch_lens
    
    def train_one_epoch(self, B: int, device: str, sample: bool):
        """
        Train for one epoch (collect B trajectories and compute gradient)
        
        Args:
            B: Number of trajectories
            device: Device for computation
            sample: If True, sample actions; else greedy
            
        Returns:
            (gradient_list, loss, avg_return, avg_length)
        """
        # Collect experience
        weights, logp, batch_rets, batch_lens = self.collect_experience_for_training(
            B, 
            device, 
            sample=sample, 
            attack_type=self.attack_type
        )
        
        # Compute policy gradient loss
        batch_loss = -(logp * weights).mean()
        
        # Compute gradient
        self.logits_net.zero_grad()
        batch_loss.backward()
        
        # Apply Byzantine attack to gradient
        if self.is_byzantine and self.attack_type is not None:
            grad = []
            for item in self.parameters():
                if self.attack_type == 'zero-gradient':
                    # Send zero gradient (stalling attack)
                    grad.append(item.grad * 0)
                
                elif self.attack_type == 'random-noise':
                    # Random noise attack
                    rnd = (torch.rand(item.grad.shape, device=item.device) * 2 - 1) * \
                          (item.grad.max().data - item.grad.min().data) * 3
                    grad.append(item.grad + rnd)
                
                elif self.attack_type == 'sign-flipping':
                    # Sign flipping attack
                    grad.append(-2.5 * item.grad)
                
                elif self.attack_type in ['reward-flipping', 'random-action', 'random-reward']:
                    # These attacks happen during trajectory collection
                    grad.append(item.grad)
                
                elif self.attack_type == 'FedScsPG-attack':
                    # Sophisticated attack (handled at server)
                    grad.append(item.grad)
                
                else:
                    raise NotImplementedError(f"Attack type {self.attack_type} not implemented")
        else:
            # Return true gradient
            grad = [item.grad for item in self.parameters()]
        
        return grad, batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens)
    
    def to(self, device: str):
        """Move worker to device"""
        self.device = device
        self.logits_net.to(device)
        return self
    
    def eval(self):
        """Set to evaluation mode"""
        self.logits_net.eval()
        return self
    
    def train(self):
        """Set to training mode"""
        self.logits_net.train()
        return self
    
    def parameters(self):
        """Get policy parameters"""
        return self.logits_net.parameters()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("WORKER - Example Usage")
    print("="*80)
    
    # Create a good worker
    print("\n1. Creating Good Worker:")
    print("-"*40)
    worker_good = Worker(
        id=1,
        is_byzantine=False,
        env_name='CartPole-v1',
        hidden_units='16,16',
        gamma=0.999,
        activation='ReLU',
        output_activation='Tanh',
        max_epi_len=500,
        device='cpu'
    )
    
    # Test rollout
    print("\n2. Testing Rollout:")
    print("-"*40)
    reward, length, _ = worker_good.rollout('cpu', max_steps=500)
    print(f"Episode reward: {reward:.2f}, length: {length}")
    
    # Test trajectory collection
    print("\n3. Collecting Trajectories:")
    print("-"*40)
    weights, logp, returns, lengths = worker_good.collect_experience_for_training(
        B=5, 
        device='cpu', 
        record=False
    )
    print(f"Collected {len(returns)} trajectories")
    print(f"Average return: {np.mean(returns):.2f}")
    print(f"Average length: {np.mean(lengths):.2f}")
    
    # Test training
    print("\n4. Training One Epoch:")
    print("-"*40)
    grad, loss, avg_ret, avg_len = worker_good.train_one_epoch(
        B=5, 
        device='cpu', 
        sample=True
    )
    print(f"Loss: {loss:.4f}")
    print(f"Avg return: {avg_ret:.2f}")
    print(f"Gradient computed: {len(grad)} parameter groups")
    
    # Create a Byzantine worker
    print("\n5. Creating Byzantine Worker:")
    print("-"*40)
    worker_byz = Worker(
        id=2,
        is_byzantine=True,
        env_name='CartPole-v1',
        hidden_units='16,16',
        gamma=0.999,
        attack_type='sign-flipping',
        max_epi_len=500,
        device='cpu'
    )
    print(f"Byzantine worker with attack: {worker_byz.attack_type}")
    
    # Test Byzantine gradient
    grad_byz, loss_byz, _, _ = worker_byz.train_one_epoch(
        B=5, 
        device='cpu', 
        sample=True
    )
    print(f"Byzantine gradient computed (attack applied)")
    
    print("\n" + "="*80)
