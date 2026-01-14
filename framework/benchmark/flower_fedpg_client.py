# pyright: reportGeneralTypeIssues=false
# type: ignore

"""
Flower Client Implementation for FedPG-BR
Each client represents an RL agent that samples trajectories and computes gradients
"""


import flwr as fl
import torch
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

from fedpg_br import (
    PolicyNetwork, 
    TrajectoryBuffer,
    sample_trajectory,
    compute_gpomdp_gradient
)


class RLClient(fl.client.NumPyClient):
    """
    Flower Client for Federated Reinforcement Learning
    
    Key differences from standard FL:
    - Clients send GRADIENTS, not model parameters
    - Clients sample trajectories from their local environment
    - No local training loop (just gradient computation)
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
        variance_bound: float = 0.1
    ):
        self.client_id = client_id
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_byzantine = is_byzantine
        self.attack_type = attack_type
        self.gamma = gamma
        self.variance_bound = variance_bound
        
        # Create local policy (will be updated with server parameters)
        self.policy = PolicyNetwork(state_dim, action_dim)
        
        # Create environment
        self.env = gym.make(env_name)
        
        print(f"Client {client_id} initialized ({'Byzantine' if is_byzantine else 'Good'})")
    
    def get_parameters(self, config: Dict[str, any]) -> List[np.ndarray]:
        """
        Get current policy parameters (not used in FedPG-BR)
        In our case, we don't send parameters back, only gradients
        """
        return [val.cpu().numpy() for val in self.policy.state_dict().values()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set policy parameters from server (theta_t_0 broadcast)
        """
        params_dict = zip(self.policy.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.policy.load_state_dict(state_dict, strict=True)
    
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
        4. Return gradient as "parameters" (Flower expects this format)
        
        Args:
            parameters: Policy parameters from server (theta_t_0)
            config: Contains batch_size B_t and other hyperparameters
            
        Returns:
            (gradient_as_numpy_arrays, num_samples, metrics)
        """
        # Set policy to server parameters
        self.set_parameters(parameters)
        
        # Get batch size from config
        batch_size = config.get("batch_size", 16)
        
        # If Byzantine, return adversarial gradient
        if self.is_byzantine:
            gradient = self._generate_byzantine_gradient()
            num_samples = batch_size  # Pretend we sampled
            metrics = {"is_byzantine": True, "attack_type": self.attack_type}
        else:
            # Good agent: sample trajectories and compute gradient
            gradient, num_samples, metrics = self._compute_honest_gradient(batch_size)
        
        # Convert gradient tensor to list of numpy arrays (Flower format)
        gradient_numpy = self._tensor_to_numpy_list(gradient)
        
        return gradient_numpy, num_samples, metrics
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, any]
    ) -> Tuple[float, int, Dict[str, any]]:
        """
        Evaluate current policy
        
        Returns:
            (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)
        
        num_eval_episodes = config.get("num_eval_episodes", 10)
        total_rewards = []
        
        for _ in range(num_eval_episodes):
            _, reward = sample_trajectory(self.env, self.policy)
            total_rewards.append(reward)
        
        avg_reward = np.mean(total_rewards)
        
        # In RL, we return negative reward as "loss" (lower is better for Flower)
        loss = -avg_reward
        
        metrics = {
            "avg_reward": avg_reward,
            "std_reward": np.std(total_rewards),
            "client_id": self.client_id
        }
        
        return loss, num_eval_episodes, metrics
    
    def _compute_honest_gradient(self, batch_size: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Compute honest gradient for good agent
        Algorithm 1, lines 5-6
        """
        gradients = []
        total_reward = 0.0
        
        for _ in range(batch_size):
            # Sample trajectory using current policy
            trajectory, reward = sample_trajectory(self.env, self.policy)
            
            # Compute GPOMDP gradient
            grad = compute_gpomdp_gradient(trajectory, self.policy, self.gamma)
            gradients.append(grad)
            total_reward += reward
        
        # mu_k_t = (1/B_t) * sum of gradients
        mu_k_t = torch.mean(torch.stack(gradients), dim=0)
        
        metrics = {
            "is_byzantine": False,
            "avg_trajectory_reward": total_reward / batch_size,
            "gradient_norm": torch.norm(mu_k_t).item(),
            "client_id": self.client_id
        }
        
        return mu_k_t, batch_size, metrics
    
    def _generate_byzantine_gradient(self) -> torch.Tensor:
        """
        Generate Byzantine gradient based on attack type
        """
        param_dim = sum(p.numel() for p in self.policy.parameters())
        
        if self.attack_type == "random_noise":
            # Random noise attack
            gradient = torch.randn(param_dim) * self.variance_bound * 10
            
        elif self.attack_type == "sign_flip":
            # Sign flipping attack: compute honest gradient and flip sign
            honest_grad, _, _ = self._compute_honest_gradient(batch_size=4)
            gradient = -2.5 * honest_grad
            
        elif self.attack_type == "zero":
            # Send zero gradient (stall learning)
            gradient = torch.zeros(param_dim)
            
        else:
            # Default: random noise
            gradient = torch.randn(param_dim) * self.variance_bound * 10
        
        return gradient
    
    def _tensor_to_numpy_list(self, tensor: torch.Tensor) -> List[np.ndarray]:
        """
        Convert flattened gradient tensor to list of numpy arrays
        (matching the shape of policy parameters)
        """
        numpy_list = []
        offset = 0
        
        for param in self.policy.parameters():
            numel = param.numel()
            param_grad = tensor[offset:offset + numel].reshape(param.shape)
            numpy_list.append(param_grad.detach().cpu().numpy())
            offset += numel
        
        return numpy_list


def create_client_fn(
    env_name: str,
    state_dim: int,
    action_dim: int,
    num_byzantine: int = 0,
    attack_type: str = "random_noise",
    gamma: float = 0.99,
    variance_bound: float = 0.1
):
    """
    Factory function to create client_fn for Flower simulation
    
    Args:
        env_name: Gym environment name
        state_dim: State space dimension
        action_dim: Action space dimension
        num_byzantine: Number of Byzantine clients
        attack_type: Type of Byzantine attack
        
    Returns:
        client_fn that takes cid and returns a Flower client
    """
    def client_fn(cid: str) -> RLClient:
        client_id = int(cid)
        is_byzantine = client_id < num_byzantine
        
        return RLClient(
            client_id=client_id,
            env_name=env_name,
            state_dim=state_dim,
            action_dim=action_dim,
            is_byzantine=is_byzantine,
            attack_type=attack_type,
            gamma=gamma,
            variance_bound=variance_bound
        )
    
    return client_fn
