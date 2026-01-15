#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportGeneralTypeIssues=false
# type: ignore
"""
Flower Client for FedPG-BR
"""

import torch
import numpy as np
import gymnasium as gym
from typing import List, Tuple, Dict
from flwr.client import NumPyClient
from flwr.common import NDArrays

# Import Worker from Betreuer's code
from worker_flower import Worker
from utils_flower import get_inner_model


class FedPGClient(NumPyClient):
    """
    Flower Client for FedPG-BR Algorithm
    
    Each client is a Worker that:
    - Receives policy parameters from server (theta_t_0)
    - Samples B_t trajectories
    - Computes gradient mu_k_t
    - Returns gradient to server
    """
    
    def __init__(
        self,
        client_id: int,
        env_name: str,
        state_dim: int,
        action_dim: int,
        is_byzantine: bool = False,
        attack_type: str = "random_noise",
        gamma: float = 0.99,
        variance_bound: float = 0.1,
        hidden_units: str = "64,64",
        activation: str = "Tanh",
        output_activation: str = "Tanh",
        max_epi_len: int = 1000,
        device: str = "cpu"
    ):
        self.client_id = client_id
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_byzantine = is_byzantine
        self.attack_type = attack_type
        self.gamma = gamma
        self.variance_bound = variance_bound
        self.device = device
        
        # **KORRIGIERT: Verwende Worker Klasse vom Betreuer**
        self.worker = Worker(
            id=client_id,
            is_byzantine=is_byzantine,
            env_name=env_name,
            hidden_units=hidden_units,
            gamma=gamma,
            activation=activation,
            output_activation=output_activation,
            attack_type=attack_type if is_byzantine else None,
            max_epi_len=max_epi_len,
            device=device
        )
        
        # Für Kompatibilität mit bestehendem Code
        self.policy = self.worker.logits_net
        self.env = self.worker.env
        
        print(f"Client {client_id} initialized ({'Byzantine' if is_byzantine else 'Good'})")
    
    def get_parameters(self, config: Dict[str, any]) -> List[np.ndarray]:
        """Return current policy parameters"""
        return [val.cpu().numpy() for val in self.policy.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set policy parameters from server"""
        params_dict = zip(self.policy.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.from_numpy(new_param).to(self.device)
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, any]]:
        """
        FedPG-BR fit method:
        1. Receive policy parameters theta_t_0 from server
        2. Sample B_t trajectories using theta_t_0
        3. Compute gradient mu_k_t = (1/B_t) * sum of gradients
        4. Return gradient as "parameters"
        
        Args:
            parameters: Policy parameters from server (theta_t_0)
            config: Contains batch_size B_t
            
        Returns:
            (gradient_as_numpy_arrays, num_samples, metrics)
        """
        # Set policy to server parameters
        self.set_parameters(parameters)
        
        # Get batch size from config
        batch_size = config.get("batch_size", 16)
        
        # Compute gradient (Byzantine or honest)
        gradient, num_samples, metrics = self._compute_gradient(batch_size)
        
        # Convert gradient tensor to list of numpy arrays (Flower format)
        gradient_numpy = self._tensor_to_numpy_list(gradient)
        
        return gradient_numpy, num_samples, metrics
    
    def _compute_gradient(self, batch_size: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Compute gradient (Byzantine or honest)
        """
        if self.is_byzantine:
            # Byzantine gradient is handled in Worker.train_one_epoch()
            # via attack_type parameter
            pass
        
        # **KORRIGIERT: Verwende Worker's train_one_epoch() wie Betreuer**
        grad_list, loss, avg_ret, avg_len = self.worker.train_one_epoch(
            B=batch_size,
            device=self.device,
            sample=True
        )
        
        # Convert grad list to single tensor
        grad_tensor = torch.cat([g.flatten() for g in grad_list])
        
        metrics = {
            "is_byzantine": self.is_byzantine,
            "attack_type": self.attack_type if self.is_byzantine else "none",
            "loss": loss,
            "avg_trajectory_reward": avg_ret,
            "avg_trajectory_length": avg_len,
            "gradient_norm": torch.norm(grad_tensor).item(),
            "client_id": self.client_id
        }
        
        return grad_tensor, batch_size, metrics
    
    def _tensor_to_numpy_list(self, tensor: torch.Tensor) -> List[np.ndarray]:
        """
        Convert flattened gradient tensor to list of numpy arrays
        matching policy parameter shapes
        """
        numpy_list = []
        offset = 0
        
        for param in self.policy.parameters():
            numel = param.numel()
            param_grad = tensor[offset:offset + numel].view(param.shape)
            numpy_list.append(param_grad.cpu().numpy())
            offset += numel
        
        return numpy_list
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, any]) -> Tuple[float, int, Dict[str, any]]:
        """
        Evaluate policy (optional for FedPG-BR)
        """
        self.set_parameters(parameters)
        
        # Run evaluation episodes
        num_episodes = config.get("num_eval_episodes", 10)
        max_steps = config.get("max_eval_steps", 1000)
        
        total_reward = 0.0
        total_length = 0
        
        for _ in range(num_episodes):
            reward, length, _ = self.worker.rollout(
                device=self.device,
                max_steps=max_steps,
                render=False
            )
            total_reward += reward
            total_length += length
        
        avg_reward = total_reward / num_episodes
        avg_length = total_length / num_episodes
        
        metrics = {
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "client_id": self.client_id
        }
        
        # Return loss (negative reward), num_examples, metrics
        return -avg_reward, num_episodes, metrics


def create_client_fn(
    env_name: str,
    state_dim: int,
    action_dim: int,
    num_byzantine: int = 0,
    attack_type: str = "random_noise",
    gamma: float = 0.99,
    variance_bound: float = 0.1,
    hidden_units: str = "64,64",
    activation: str = "Tanh",
    output_activation: str = "Tanh",
    max_epi_len: int = 1000,
    device: str = "cpu"
):
    """
    Factory function to create Flower clients
    
    Args:
        env_name: Environment name
        state_dim: State dimension
        action_dim: Action dimension
        num_byzantine: Number of Byzantine clients
        attack_type: Type of Byzantine attack
        gamma: Discount factor
        variance_bound: Sigma for Byzantine filtering
        hidden_units: Hidden layer sizes
        activation: Activation function
        output_activation: Output activation
        max_epi_len: Maximum episode length
        device: Device for computation
        
    Returns:
        client_fn: Function that creates a client given cid
    """
    def client_fn(cid: str) -> FedPGClient:
        """Create a Flower client"""
        client_id = int(cid)
        is_byzantine = client_id < num_byzantine
        
        return FedPGClient(
            client_id=client_id,
            env_name=env_name,
            state_dim=state_dim,
            action_dim=action_dim,
            is_byzantine=is_byzantine,
            attack_type=attack_type,
            gamma=gamma,
            variance_bound=variance_bound,
            hidden_units=hidden_units,
            activation=activation,
            output_activation=output_activation,
            max_epi_len=max_epi_len,
            device=device
        )
    
    return client_fn