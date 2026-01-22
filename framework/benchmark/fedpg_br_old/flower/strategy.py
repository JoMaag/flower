"""
Flower Strategy for FedPG-BR (Algorithm 1).

Implements:
- Byzantine filtering (Algorithm 1.1)
- SCSG inner loop with importance sampling
- Adaptive batch size sampling
"""

import logging
import numpy as np
import torch
import gymnasium as gym
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from fedpg_br.policy import create_policy
from fedpg_br.core.byzantine import ByzantineFilter
from fedpg_br.core.trajectory import sample_trajectory, compute_returns
from fedpg_br.core.gradient import compute_policy_gradient, compute_log_probs

logger = logging.getLogger(__name__)


class FedPGStrategy(Strategy):
    """
    FedPG-BR Strategy for Flower.
    
    Algorithm 1 from paper:
    1. Broadcast θ_t_0 to all agents
    2. Receive gradients μ_k_t from agents
    3. Apply Byzantine filtering (Algorithm 1.1)
    4. Perform SCSG inner loop at server
    5. Update server policy
    """
    
    def __init__(
        self,
        env_name: str,
        state_dim: int,
        action_dim: int,
        hidden_units: Tuple[int, ...] = (64, 64),
        activation: str = "Tanh",
        output_activation: str = "Identity",
        batch_size: int = 16,
        batch_size_range: Optional[Tuple[int, int]] = None,
        mini_batch_size: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.99,
        sigma: float = 0.1,
        delta: float = 0.6,
        num_agents: int = 10,
        byzantine_ratio: float = 0.0,
        evaluate_every: int = 10,
    ):
        """
        Args:
            env_name: Environment name
            state_dim: Observation dimension
            action_dim: Action dimension
            hidden_units: Policy network hidden layers
            activation: Activation function
            output_activation: Output activation
            batch_size: Default batch size (B)
            batch_size_range: If set, sample B from [min, max] (FedPG-BR)
            mini_batch_size: Mini-batch for SCSG (b)
            lr: Learning rate
            gamma: Discount factor
            sigma: Variance bound for Byzantine filtering
            delta: Confidence parameter
            num_agents: Number of agents (K)
            byzantine_ratio: Fraction of Byzantine agents (α)
            evaluate_every: Evaluate every N rounds
        """
        super().__init__()
        
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.batch_size_range = batch_size_range
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.num_agents = num_agents
        self.evaluate_every = evaluate_every
        
        # Server policy
        self.policy = create_policy(
            state_dim=state_dim,
            action_dim=action_dim,
            env_name=env_name,
            hidden_units=hidden_units,
            activation=activation,
            output_activation=output_activation,
        )
        
        # Adam optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Byzantine filter
        self.byzantine_filter = ByzantineFilter(
            sigma=sigma,
            delta=delta,
            num_agents=num_agents,
            alpha=byzantine_ratio,
        )
        
        # Server environment (for SCSG inner loop)
        self.env = gym.make(env_name)
        
        # Tracking
        self.current_round = 0
        self.theta_t_0 = None  # Snapshot at round start
        self.policy_snapshots = []  # For random output selection (Algorithm 1, line 14)
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        ndarrays = [p.cpu().detach().numpy() for p in self.policy.parameters()]
        return ndarrays_to_parameters(ndarrays)
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """
        Configure training round.
        
        Algorithm 1, line 3: θ_t_0 ← θ̃_{t-1}, broadcast to all agents
        """
        self.current_round = server_round
        
        # Save snapshot
        self.theta_t_0 = self.policy.get_flat_params().clone()
        
        # Sample batch size (adaptive for FedPG-BR)
        if self.batch_size_range is not None:
            current_batch = np.random.randint(
                self.batch_size_range[0],
                self.batch_size_range[1] + 1
            )
        else:
            current_batch = self.batch_size
        
        self._current_batch_size = current_batch
        
        # Sample all clients
        clients = client_manager.sample(
            num_clients=self.num_agents,
            min_num_clients=self.num_agents,
        )
        
        config = {"batch_size": current_batch, "round": server_round}
        fit_ins = fl.common.FitIns(parameters, config)
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate gradients with Byzantine filtering and SCSG update.
        
        Algorithm 1, lines 7-13
        """
        if not results:
            return None, {}
        
        batch_size = self._current_batch_size
        
        # Extract gradients from clients
        gradients = []
        for _, fit_res in results:
            grad_numpy = parameters_to_ndarrays(fit_res.parameters)
            grad_tensor = torch.cat([
                torch.from_numpy(g).flatten() for g in grad_numpy
            ]).float()
            gradients.append(grad_tensor)
        
        # Line 7: Byzantine filtering and aggregation
        mu_t, good_agents = self.byzantine_filter.aggregate(gradients, batch_size)
        
        # Line 8: Sample N_t from geometric distribution
        p_geom = self.mini_batch_size / (batch_size + self.mini_batch_size)
        N_t = np.random.geometric(p_geom)
        
        # Lines 9-12: SCSG inner loop
        theta_t_n = self.theta_t_0.clone()
        actual_steps = 0
        
        for n in range(N_t):
            v_t_n, ratio = self._scsg_step(theta_t_n, self.theta_t_0, mu_t)
            
            # Early stopping if importance ratios diverge
            if ratio < 0.995 or ratio > 1.005:
                logger.debug(f"Round {server_round}, step {n}: early stop (ratio={ratio:.4f})")
                break
            
            # Update via optimizer
            self.policy.set_flat_params(theta_t_n)
            self.optimizer.zero_grad()
            
            offset = 0
            for param in self.policy.parameters():
                size = param.numel()
                param.grad = v_t_n[offset:offset + size].view(param.shape).clone()
                offset += size
            
            self.optimizer.step()
            theta_t_n = self.policy.get_flat_params()
            actual_steps = n + 1
        
        # Line 13: θ̃_t ← θ_t_{N_t}
        self.policy.set_flat_params(theta_t_n)
        self.policy_snapshots.append(theta_t_n.clone())
        
        # Metrics
        metrics = {
            "num_good_agents": len(good_agents),
            "num_byzantine_detected": len(results) - len(good_agents),
            "scsg_steps": actual_steps,
            "gradient_norm": torch.norm(mu_t).item(),
            "batch_size": batch_size,
        }
        
        ndarrays = [p.cpu().detach().numpy() for p in self.policy.parameters()]
        return ndarrays_to_parameters(ndarrays), metrics
    
    def _scsg_step(
        self,
        theta_n: torch.Tensor,
        theta_0: torch.Tensor,
        mu_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute semi-stochastic gradient (Algorithm 1, line 11).
        
        v_t_n = (1/b) Σ [g(τ|θ_n) - ω(τ|θ_n,θ_0) g(τ|θ_0)] + μ_t
        
        Returns:
            (gradient, mean_importance_ratio)
        """
        # Create policy copies
        policy_n = create_policy(
            self.state_dim, self.action_dim, self.env_name,
            hidden_units=tuple(self.policy.sizes[1:-1]) if hasattr(self.policy, 'sizes') else (64, 64),
        )
        policy_n.set_flat_params(theta_n)
        
        policy_0 = create_policy(
            self.state_dim, self.action_dim, self.env_name,
            hidden_units=tuple(self.policy.sizes[1:-1]) if hasattr(self.policy, 'sizes') else (64, 64),
        )
        policy_0.set_flat_params(theta_0)
        
        all_grad_new = []
        all_grad_old = []
        all_ratios = []
        
        for _ in range(self.mini_batch_size):
            # Sample trajectory with policy_n
            trajectory, _ = sample_trajectory(self.env, policy_n)
            returns = compute_returns(trajectory, self.gamma)
            
            # Gradient for policy_n
            grad_new, log_probs_n = compute_policy_gradient(
                trajectory, policy_n, self.gamma, returns
            )
            
            # Log probs for policy_0 on SAME actions (importance sampling)
            log_probs_0 = compute_log_probs(trajectory, policy_0)
            
            # Importance ratios
            ratios = torch.exp(log_probs_0.detach() - log_probs_n.detach())
            all_ratios.append(ratios.mean().item())
            
            # Weighted gradient for policy_0
            loss_0 = -(log_probs_0 * returns * ratios).mean()
            policy_0.zero_grad()
            loss_0.backward()
            grad_old = torch.cat([p.grad.flatten().clone() for p in policy_0.parameters()])
            
            all_grad_new.append(grad_new)
            all_grad_old.append(grad_old)
        
        # Average and apply SCSG correction
        avg_grad_new = torch.mean(torch.stack(all_grad_new), dim=0)
        avg_grad_old = torch.mean(torch.stack(all_grad_old), dim=0)
        mean_ratio = np.mean(all_ratios)
        
        v_t_n = avg_grad_new - avg_grad_old + mu_t
        
        return v_t_n, mean_ratio
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """Configure evaluation round."""
        if server_round % self.evaluate_every != 0:
            return []
        
        sample_size = min(5, self.num_agents)
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min(3, sample_size),
        )
        
        config = {"num_episodes": 10, "max_steps": 1000}
        eval_ins = fl.common.EvaluateIns(parameters, config)
        
        return [(client, eval_ins) for client in clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}
        
        rewards = [-res.loss for _, res in results]
        avg_reward = np.mean(rewards)
        
        logger.info(f"Round {server_round}: Avg Reward = {avg_reward:.2f} ± {np.std(rewards):.2f}")
        
        return -avg_reward, {
            "avg_reward": avg_reward,
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }
    
    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation."""
        rewards = []
        for _ in range(10):
            _, reward = sample_trajectory(self.env, self.policy)
            rewards.append(reward)
        
        avg_reward = np.mean(rewards)
        return -avg_reward, {
            "server_avg_reward": avg_reward,
            "server_std_reward": np.std(rewards),
        }
