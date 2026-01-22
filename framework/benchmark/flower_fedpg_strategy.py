# pyright: reportGeneralTypeIssues=false
# type: ignore

"""
Flower Strategy Implementation for FedPG-BR
Custom strategy that implements Byzantine filtering and SCSG updates
"""


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

import torch
import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

from fedpg_br_old import (
    PolicyNetwork,
    ByzantineFilter,
    sample_trajectory,
    compute_policy_gradient,
    compute_log_probs_fixed_actions,
    compute_returns  # **NEU: FEHLENDER IMPORT**
)


class FedPGStrategy(Strategy):
    """
    Custom Flower Strategy implementing FedPG-BR
    
    This strategy:
    1. Broadcasts server policy theta_t_0 to all clients
    2. Receives gradients (not parameters!) from clients
    3. Applies Byzantine filtering (Algorithm 1.1)
    4. Performs SCSG inner loop at server
    5. Updates server policy
    """
    
    def __init__(
        self,
        env_name: str,
        state_dim: int,
        action_dim: int,
        batch_size: int = 16,
        mini_batch_size: int = 4,
        step_size: float = 1e-3,  # Wird jetzt lr für Adam
        gamma: float = 0.99,
        variance_bound: float = 0.1,
        confidence_param: float = 0.6,
        byzantine_ratio: float = 0.0,
        num_agents: int = 10,
        min_available_clients: int = 10,
        evaluate_every: int = 10,
        batch_size_range: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.step_size = step_size  # Behalte für Kompatibilität
        self.gamma = gamma
        self.variance_bound = variance_bound
        self.confidence_param = confidence_param
        self.byzantine_ratio = byzantine_ratio
        self.num_agents = num_agents
        self.min_available_clients = min_available_clients
        self.evaluate_every = evaluate_every
        self.batch_size_range = batch_size_range
        
        # Initialize server policy
        self.server_policy = PolicyNetwork(state_dim, action_dim)
        
        # **NEU: Adam Optimizer wie Betreuer**
        self.optimizer = torch.optim.Adam(
            self.server_policy.parameters(), 
            lr=step_size
        )
        
        # Snapshots for random output selection
        self.policy_snapshots = []
        
        # Byzantine filter
        self.byzantine_filter = ByzantineFilter(
            sigma=variance_bound,
            delta=confidence_param,
            K=num_agents,
            alpha=byzantine_ratio
        )
        
        # Create server environment
        self.server_env = gym.make(env_name)
        
        # Tracking
        self.current_round = 0
        self.theta_t_0 = None
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """
        Initialize global model parameters
        Called once at the start
        """
        # Get initial parameters from server policy
        ndarrays = [val.cpu().numpy() for val in self.server_policy.state_dict().values()]
        return ndarrays_to_parameters(ndarrays)
    
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """
        Configure the next round of training
        
        Algorithm 1, line 3: Broadcast theta_t_0 to all agents
        """
        self.current_round = server_round
        
        # Line 3: theta_t_0 <- theta_tilde_{t-1}
        # Save snapshot of current policy
        self.theta_t_0 = self._get_policy_tensor().clone()
        
        # Adaptive batch size sampling (like in paper Table 2)
        if self.batch_size_range is not None:
            current_batch_size = np.random.randint(
                self.batch_size_range[0], 
                self.batch_size_range[1] + 1
            )
        else:
            current_batch_size = self.batch_size
        
        # Store for use in aggregate_fit
        self.current_batch_size = current_batch_size
        
        # Sample all clients (we need all K agents)
        sample_size = self.num_agents
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients
        )
        
        # Create fit instructions with batch size
        config = {
            "batch_size": current_batch_size,
            "server_round": server_round
        }
        
        fit_ins = fl.common.FitIns(parameters, config)
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate gradients from clients with Byzantine filtering
        Then perform SCSG inner loop
        
        Algorithm 1, lines 7-13
        """
        if not results:
            return None, {}
        
        # Extract gradients from results (clients sent gradients, not parameters)
        gradients_list = []
        client_metrics = []
        
        for client, fit_res in results:
            # Convert parameters (which are actually gradients) to tensors
            gradient_numpy = parameters_to_ndarrays(fit_res.parameters)
            gradient_tensor = self._numpy_list_to_tensor(gradient_numpy)
            gradients_list.append(gradient_tensor)
            client_metrics.append(fit_res.metrics)
        
        # Line 7: Byzantine filtering and aggregation
        # Use current_batch_size from configure_fit
        batch_size_used = getattr(self, 'current_batch_size', self.batch_size)
        mu_t, good_agents = self.byzantine_filter.aggregate(gradients_list, batch_size_used)
        
        # Line 8: Sample N_t from geometric distribution
        # NOTE: np.random.geometric(p) uses p as SUCCESS probability
        # We want p = b/(B+b) for success, so use 1 - B/(B+b)
        p_geom = 1.0 - batch_size_used / (batch_size_used + self.mini_batch_size)
        N_t = np.random.geometric(p_geom)
        
        # Lines 9-12: SCSG inner loop
        theta_t_n = self.theta_t_0.clone()
        actual_N_t = N_t  # Track actual steps taken (for early stopping)
        
        for n in range(N_t):
            # Line 11: Compute semi-stochastic gradient
            v_t_n, ratio_mean = self._compute_semi_stochastic_gradient(
                theta_t_n, self.theta_t_0, mu_t
            )
            
            # **KRITISCH: Early stopping check (Betreuer Zeilen 285-288)**
            if abs(ratio_mean) < 0.995 or abs(ratio_mean) > 1.005:
                actual_N_t = n
                print(f"  Round {server_round}, Step {n}/{N_t}: Early stop (ratio={ratio_mean:.4f})")
                break
            
            # **KORRIGIERT: Use optimizer.step() wie Betreuer (Zeilen 282-286)**
            # Set current policy parameters
            self._set_policy_from_tensor(theta_t_n)
            
            # Manually set gradients (v_t_n is the final gradient)
            self.optimizer.zero_grad()
            offset = 0
            for param in self.server_policy.parameters():
                numel = param.numel()
                param.grad = v_t_n[offset:offset + numel].view(param.shape).clone()
                offset += numel
            
            # Let Adam optimizer do the update
            self.optimizer.step()
            
            # Get updated parameters as tensor
            theta_t_n = self._get_policy_tensor()
        
        # Line 13: Update server policy with theta_t_N_t
        self._set_policy_from_tensor(theta_t_n)
        
        # Save snapshot for random selection at end
        self.policy_snapshots.append(theta_t_n.clone())
        
        # Convert updated policy to parameters
        updated_parameters = self._get_parameters()
        
        # Aggregate metrics
        num_good = len(good_agents)
        num_byzantine = len(results) - num_good
        
        metrics = {
            "num_good_agents": num_good,
            "num_byzantine_detected": num_byzantine,
            "scsg_steps": actual_N_t,  # Use actual steps taken
            "sampled_N_t": N_t,  # Original sampled value
            "gradient_norm": torch.norm(mu_t).item(),
            "batch_size_used": batch_size_used,
        }
        
        # Trajectory counting (from original agent.py line 306)
        # step += round((Batch_size * world_size + b * N_t) / (1 + world_size))
        total_trajectories = round(
            (batch_size_used * self.num_agents + self.mini_batch_size * actual_N_t) 
            / (1 + self.num_agents)
        )
        metrics["total_trajectories"] = total_trajectories
        
        # Add client metrics
        byzantine_detected = []
        for i, m in enumerate(client_metrics):
            if i not in good_agents:
                byzantine_detected.append(i)
        
        if byzantine_detected:
            metrics["byzantine_agents"] = str(byzantine_detected)
        
        return ndarrays_to_parameters(updated_parameters), metrics
    
    def configure_evaluate(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """
        Configure evaluation round
        Only evaluate every N rounds to save computation
        """
        if server_round % self.evaluate_every != 0:
            return []
        
        # Sample subset of clients for evaluation
        sample_size = min(5, self.num_agents)
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min(3, sample_size)
        )
        
        config = {"num_eval_episodes": 10}
        evaluate_ins = fl.common.EvaluateIns(parameters, config)
        
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients
        """
        if not results:
            return None, {}
        
        # Aggregate rewards (loss is negative reward)
        rewards = [-res.loss for _, res in results]
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        metrics = {
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards)
        }
        
        print(f"Round {server_round}: Avg Reward = {avg_reward:.2f} ± {std_reward:.2f}")
        
        # Return negative reward as loss
        return -avg_reward, metrics
    
    def evaluate(
        self, 
        server_round: int, 
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Server-side evaluation of current policy
        """
        # Evaluate current server policy
        total_rewards = []
        for _ in range(10):
            _, reward = sample_trajectory(self.server_env, self.server_policy)
            total_rewards.append(reward)
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        metrics = {
            "server_avg_reward": avg_reward,
            "server_std_reward": std_reward
        }
        
        return -avg_reward, metrics
    
    def _compute_semi_stochastic_gradient(
        self, 
        theta_t_n: torch.Tensor, 
        theta_t_0: torch.Tensor,
        mu_t: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute semi-stochastic gradient (Algorithm 1, line 11)
        
        v_t_n = (1/b_t) * sum [g(tau|theta_n) - omega(tau|theta_n, theta_0) * g(tau|theta_0)] + mu_t
        
        Returns: (v_t_n, ratio_mean)
        """
        # Create policies for theta_n and theta_0
        policy_n = PolicyNetwork(self.state_dim, self.action_dim)
        policy_n.set_parameters(theta_t_n)
        
        policy_0 = PolicyNetwork(self.state_dim, self.action_dim)
        policy_0.set_parameters(theta_t_0)
        
        all_grad_new = []
        all_grad_old = []
        all_ratios = []
        
        for step_idx in range(self.mini_batch_size):
            # Sample trajectory using policy_n (theta_n)
            trajectory, _ = sample_trajectory(self.server_env, policy_n)
            
            # Compute returns for weighting
            returns = compute_returns(trajectory, self.gamma)
            
            # Step 1: Compute gradient for policy_n
            grad_new, log_probs_n = compute_policy_gradient(
                trajectory, policy_n, self.gamma, returns
            )
            
            # Step 2: Compute log_probs for policy_0 with FIXED actions from trajectory
            # This is the KEY for importance sampling!
            log_probs_0 = compute_log_probs_fixed_actions(trajectory, policy_0)
            
            # Step 3: Compute importance ratios
            # ratios = exp(log_prob_0 - log_prob_n) per timestep
            ratios = torch.exp(log_probs_0.detach() - log_probs_n.detach())
            ratio_mean = ratios.mean().item()
            all_ratios.append(ratio_mean)
            
            # Step 4: Compute weighted loss for policy_0
            # Betreuer line 278: loss_old = -(old_logp * weights * ratios).mean()
            loss_old = -(log_probs_0 * returns * ratios).mean()
            
            # Step 5: Backpropagate to get grad_old
            policy_0.zero_grad()
            loss_old.backward()
            grad_old = torch.cat([p.grad.flatten().clone() for p in policy_0.parameters()])
            
            # Store gradients
            all_grad_new.append(grad_new)
            all_grad_old.append(grad_old)
        
        # Average gradients across mini-batch
        avg_grad_new = torch.mean(torch.stack(all_grad_new), dim=0)
        avg_grad_old = torch.mean(torch.stack(all_grad_old), dim=0)
        avg_ratio = np.mean(all_ratios)
        
        # SCSG correction: v_t_n = grad_new - grad_old + mu_t
        # Betreuer line 282: item.grad = item.grad - grad_old[idx] + mu[idx]
        v_t_n = avg_grad_new - avg_grad_old + mu_t
        
        return v_t_n, avg_ratio
    
    def _get_policy_tensor(self) -> torch.Tensor:
        """Get flattened policy parameters as tensor"""
        return torch.cat([p.flatten() for p in self.server_policy.parameters()])
    
    def _set_policy_from_tensor(self, params: torch.Tensor):
        """Set policy parameters from flattened tensor"""
        self.server_policy.set_parameters(params)
    
    def _get_parameters(self) -> List[np.ndarray]:
        """Get policy parameters as list of numpy arrays"""
        return [val.cpu().numpy() for val in self.server_policy.state_dict().values()]
    
    def _numpy_list_to_tensor(self, numpy_list: List[np.ndarray]) -> torch.Tensor:
        """Convert list of numpy arrays to flattened tensor"""
        tensors = [torch.from_numpy(arr).flatten() for arr in numpy_list]
        return torch.cat(tensors).float()
