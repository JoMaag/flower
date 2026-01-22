"""
Flower Client for FedPG-BR.

Each client wraps a Worker and handles Flower communication protocol.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Callable

from flwr.client import NumPyClient, Client
from flwr.common import Context

from fedpg_br.flower.worker import Worker

logger = logging.getLogger(__name__)


class FedPGClient(NumPyClient):
    """
    Flower Client for FedPG-BR Algorithm.
    
    Protocol:
    1. Receive policy parameters θ_t_0 from server
    2. Sample B_t trajectories
    3. Compute gradient μ_k_t
    4. Return gradient (not parameters!) to server
    """
    
    def __init__(
        self,
        worker: Worker,
    ):
        """
        Args:
            worker: Worker instance for this client
        """
        self.worker = worker
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current policy parameters."""
        return [p.cpu().detach().numpy() for p in self.worker.policy.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set policy parameters from server."""
        for param, new_value in zip(self.worker.policy.parameters(), parameters):
            param.data = torch.from_numpy(new_value).to(self.worker.device)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Compute gradient for this round.
        
        Args:
            parameters: Policy parameters from server (θ_t_0)
            config: Contains batch_size
            
        Returns:
            (gradient_as_numpy, num_samples, metrics)
        """
        # Load server parameters
        self.set_parameters(parameters)
        
        batch_size = config.get("batch_size", 16)
        
        # Compute gradient
        grad_list, loss, avg_return, avg_length = self.worker.compute_gradient(
            num_trajectories=batch_size,
            sample=True,
        )
        
        # Convert to numpy (matching parameter shapes)
        gradient_numpy = [g.cpu().numpy() for g in grad_list]
        
        metrics = {
            "loss": loss,
            "avg_return": avg_return,
            "avg_length": avg_length,
            "is_byzantine": self.worker.is_byzantine,
            "client_id": self.worker.id,
        }
        
        return gradient_numpy, batch_size, metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate policy.
        
        Returns:
            (loss, num_examples, metrics)
        """
        self.set_parameters(parameters)
        
        num_episodes = config.get("num_episodes", 10)
        max_steps = config.get("max_steps", 1000)
        
        avg_reward, avg_length = self.worker.evaluate(num_episodes, max_steps)
        
        metrics = {
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "client_id": self.worker.id,
        }
        
        # Return negative reward as loss
        return -avg_reward, num_episodes, metrics


def create_client_fn(
    env_name: str,
    hidden_units: Tuple[int, ...],
    gamma: float,
    activation: str = "Tanh",
    output_activation: str = "Identity",
    num_byzantine: int = 0,
    attack_type: str = "random-noise",
    max_episode_len: int = 1000,
    device: str = "cpu",
) -> Callable[[Context], Client]:
    """
    Factory function for creating Flower clients.
    
    Args:
        env_name: Environment name
        hidden_units: Hidden layer sizes
        gamma: Discount factor
        activation: Activation function
        output_activation: Output activation
        num_byzantine: Number of Byzantine clients (assigned to lowest IDs)
        attack_type: Byzantine attack type
        max_episode_len: Max episode length
        device: Computation device
        
    Returns:
        Function that creates a Client given Context
    """
    def client_fn(context: Context) -> Client:
        # Extract client_id from context (Flower simulation passes partition-id)
        partition_id = context.node_config.get("partition-id", None)
        if partition_id is not None:
            client_id = int(partition_id)
        else:
            # Fallback: try node_id
            client_id = int(context.node_id) if context.node_id else 0
        
        is_byzantine = client_id < num_byzantine
        
        worker = Worker(
            worker_id=client_id,
            env_name=env_name,
            hidden_units=hidden_units,
            gamma=gamma,
            activation=activation,
            output_activation=output_activation,
            is_byzantine=is_byzantine,
            attack_type=attack_type if is_byzantine else None,
            max_episode_len=max_episode_len,
            device=device,
        )
        
        return FedPGClient(worker).to_client()
    
    return client_fn
