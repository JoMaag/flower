"""
Flower ClientApp for FedPG-BR.
"""

import torch
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from fedpg_br.config import get_config, get_env_info
from fedpg_br.flower.worker import Worker


class FedPGClient(NumPyClient):
    """FedPG-BR Client that computes and returns gradients."""
    
    def __init__(self, worker: Worker):
        self.worker = worker
    
    def get_parameters(self, config):
        return [p.cpu().detach().numpy() for p in self.worker.policy.parameters()]
    
    def set_parameters(self, parameters):
        for param, new_value in zip(self.worker.policy.parameters(), parameters):
            param.data = torch.from_numpy(new_value).to(self.worker.device)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        batch_size = config.get("batch_size", 16)
        
        grad_list, loss, avg_return, avg_length = self.worker.compute_gradient(batch_size, sample=True)
        gradient_numpy = [g.cpu().numpy() for g in grad_list]
        
        return gradient_numpy, batch_size, {
            "loss": float(loss),
            "avg_return": float(avg_return),
            "avg_length": float(avg_length),
            "is_byzantine": self.worker.is_byzantine,
        }
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        num_episodes = config.get("num_episodes", 10)
        avg_reward, avg_length = self.worker.evaluate(num_episodes)
        return float(-avg_reward), num_episodes, {"avg_reward": float(avg_reward)}


def client_fn(context: Context):
    """Create a FedPG-BR client."""
    # Get config from run
    run_config = context.run_config
    env_name = run_config.get("env", "CartPole-v1")
    num_byzantine = int(run_config.get("num-byzantine", 0))
    attack_type = run_config.get("attack-type", "random-noise")
    
    # Get partition ID (client ID)
    partition_id = context.node_config.get("partition-id", 0)
    client_id = int(partition_id)
    is_byzantine = client_id < num_byzantine
    
    # Load environment config
    config = get_config(env_name)
    
    # Create worker
    worker = Worker(
        worker_id=client_id,
        env_name=env_name,
        hidden_units=config.hidden_units,
        gamma=config.gamma,
        activation=config.activation,
        output_activation=config.output_activation,
        is_byzantine=is_byzantine,
        attack_type=attack_type if is_byzantine else None,
        max_episode_len=config.max_episode_len,
        device="cpu",
    )
    
    return FedPGClient(worker).to_client()


# Create Flower ClientApp
app = ClientApp(client_fn=client_fn)
