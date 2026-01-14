# pyright: reportGeneralTypeIssues=false
# type: ignore
"""
Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee
Implementation of FedPG-BR (Federated Policy Gradient with Byzantine Resilience)

Based on the paper by Flint Xiaofeng Fan et al. (NeurIPS 2021)
Algorithm 1 and Algorithm 1.1 from the paper
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import gymnasium as gym


@dataclass
class FedPGConfig:
    """Configuration for FedPG-BR algorithm"""
    num_rounds: int = 100  # T in paper
    batch_size: int = 32  # B_t in paper
    mini_batch_size: int = 4  # b_t in paper
    step_size: float = 1e-3  # eta_t in paper
    discount_factor: float = 0.99  # gamma in paper
    variance_bound: float = 0.1  # sigma in paper
    confidence_param: float = 0.6  # delta in paper
    byzantine_ratio: float = 0.0  # alpha in paper (fraction of Byzantine agents)
    num_agents: int = 10  # K in paper
    
    # Computed parameters
    def get_filtering_threshold(self, batch_size: int) -> float:
        """T_mu = 2*sigma*sqrt(V/B_t) where V = 2*log(2K/delta)"""
        V = 2 * np.log(2 * self.num_agents / self.confidence_param)
        return 2 * self.variance_bound * np.sqrt(V / batch_size)


class PolicyNetwork(nn.Module):
    """Neural network policy for RL agents"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.network(state)
    
    def get_parameters(self) -> torch.Tensor:
        """Get flattened parameters"""
        return torch.cat([p.flatten() for p in self.parameters()])
    
    def set_parameters(self, params: torch.Tensor):
        """Set parameters from flattened tensor"""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = params[offset:offset + numel].view(p.shape)
            offset += numel


class TrajectoryBuffer:
    """Buffer for storing trajectories"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
    def add(self, state, action, reward, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
    def __len__(self):
        return len(self.rewards)


def sample_trajectory(env: gym.Env, policy: PolicyNetwork, max_steps: int = 1000) -> Tuple[TrajectoryBuffer, float]:
    """
    Sample a trajectory using the given policy
    Returns: (trajectory_buffer, cumulative_reward)
    """
    buffer = TrajectoryBuffer()
    
    # Handle both old and new gym API
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state, _ = reset_result  # New gym API returns (obs, info)
    else:
        state = reset_result  # Old gym API returns obs only
    
    total_reward = 0.0
    
    for _ in range(max_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = policy(state_tensor)
        
        # Sample action from policy
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        # Handle both old and new gym API
        step_result = env.step(action.item())
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result  # New API
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_result  # Old API
        
        buffer.add(state, action.item(), reward, log_prob.item())
        total_reward += reward
        state = next_state
        
        if done:
            break
            
    return buffer, total_reward


def compute_gpomdp_gradient(trajectory: TrajectoryBuffer, policy: PolicyNetwork, 
                           gamma: float, baseline: float = 0.0) -> torch.Tensor:
    """
    Compute GPOMDP gradient estimator g(tau|theta)
    
    GPOMDP from paper:
    g(tau|theta) = sum_{h=0}^{H-1} [sum_{t=0}^h grad_theta log pi_theta(a_t|s_t)] * (gamma^h * r(s_h, a_h) - C_b_h)
    """
    policy.zero_grad()
    gradient = None
    
    H = len(trajectory)
    cumulative_log_prob_grad = None
    
    for h in range(H):
        # Compute gradient of log pi_theta(a_h | s_h)
        state = torch.FloatTensor(trajectory.states[h]).unsqueeze(0)
        action_probs = policy(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = torch.tensor([trajectory.actions[h]])
        log_prob = action_dist.log_prob(action)
        
        # Accumulate log probability gradients
        log_prob.backward(retain_graph=True)
        if cumulative_log_prob_grad is None:
            cumulative_log_prob_grad = torch.cat([p.grad.flatten().clone() for p in policy.parameters()])
        else:
            current_grad = torch.cat([p.grad.flatten() for p in policy.parameters()])
            cumulative_log_prob_grad = cumulative_log_prob_grad + current_grad
        policy.zero_grad()
        
        # Compute advantage: gamma^h * r(s_h, a_h) - baseline
        advantage = (gamma ** h) * trajectory.rewards[h] - baseline
        
        # Accumulate gradient
        if gradient is None:
            gradient = advantage * cumulative_log_prob_grad.clone()
        else:
            gradient += advantage * cumulative_log_prob_grad.clone()
    
    return gradient


def compute_importance_weight(trajectory: TrajectoryBuffer, policy_n: PolicyNetwork, 
                              policy_0: PolicyNetwork) -> float:
    """
    Compute importance weight omega(tau | theta_n, theta_0) = p(tau|theta_0) / p(tau|theta_n)
    """
    log_prob_0 = 0.0
    log_prob_n = 0.0
    
    for state, action in zip(trajectory.states, trajectory.actions):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            probs_0 = policy_0(state_tensor)
            probs_n = policy_n(state_tensor)
            
        log_prob_0 += torch.log(probs_0[0, action] + 1e-8).item()
        log_prob_n += torch.log(probs_n[0, action] + 1e-8).item()
    
    # omega = p(tau|theta_0) / p(tau|theta_n) = exp(log_prob_0 - log_prob_n)
    omega = np.exp(log_prob_0 - log_prob_n)
    return omega


class ByzantineFilter:
    """
    Implementation of Algorithm 1.1: FedPG-Aggregate
    Byzantine filtering with two rules R1 and R2
    """
    def __init__(self, sigma: float, delta: float, K: int, alpha: float):
        self.sigma = sigma
        self.delta = delta
        self.K = K
        self.alpha = alpha
        
    def aggregate(self, gradients: List[torch.Tensor], batch_size: int) -> Tuple[torch.Tensor, List[int]]:
        """
        Aggregate gradients with Byzantine filtering
        
        Args:
            gradients: List of gradient tensors from K agents
            batch_size: B_t used for this round
            
        Returns:
            (aggregated_gradient, list_of_good_agent_indices)
        """
        K = len(gradients)
        V = 2 * np.log(2 * K / self.delta)
        T_mu = 2 * self.sigma * np.sqrt(V / batch_size)
        
        # Try R1 first (with tighter threshold)
        G_t = self._apply_filtering_rule(gradients, T_mu)
        
        # If R1 fails (doesn't include enough agents), fall back to R2
        if len(G_t) < (1 - self.alpha) * K:
            # R2 uses 2*sigma as threshold
            G_t = self._apply_filtering_rule(gradients, 2 * self.sigma)
        
        # Aggregate gradients from good agents
        good_gradients = [gradients[i] for i in G_t]
        aggregated = torch.mean(torch.stack(good_gradients), dim=0)
        
        return aggregated, G_t
    
    def _apply_filtering_rule(self, gradients: List[torch.Tensor], threshold: float) -> List[int]:
        """
        Apply filtering rule to find good agents
        
        Algorithm 1.1 lines 2-4 (R1) or lines 6-8 (R2):
        1. Find S: set of gradients where each has >K/2 neighbors within threshold
        2. Find mu_mom: mean of median (gradient in S closest to mean of S)
        3. Find G_t: agents within threshold of mu_mom
        """
        K = len(gradients)
        
        # Step 1: Find S (vector medians)
        S_indices = []
        for k in range(K):
            # Count neighbors within threshold
            neighbors = sum(1 for j in range(K) 
                          if torch.norm(gradients[k] - gradients[j]) <= threshold)
            if neighbors > K / 2:
                S_indices.append(k)
        
        if len(S_indices) == 0:
            # Fallback: return all agents
            return list(range(K))
        
        # Step 2: Find mu_mom (mean of median)
        S_gradients = [gradients[i] for i in S_indices]
        mean_S = torch.mean(torch.stack(S_gradients), dim=0)
        
        # Find gradient in S closest to mean
        min_dist = float('inf')
        mu_mom_idx = S_indices[0]
        for idx in S_indices:
            dist = torch.norm(gradients[idx] - mean_S).item()
            if dist < min_dist:
                min_dist = dist
                mu_mom_idx = idx
        
        mu_mom = gradients[mu_mom_idx]
        
        # Step 3: Find G_t (agents within threshold of mu_mom)
        G_t = []
        for k in range(K):
            if torch.norm(gradients[k] - mu_mom) <= threshold:
                G_t.append(k)
        
        return G_t


class FedPGBR:
    """
    Main implementation of Algorithm 1: FedPG-BR
    Federated Policy Gradient with Byzantine Resilience
    """
    def __init__(self, env_fn: Callable, config: FedPGConfig, state_dim: int, action_dim: int):
        self.config = config
        self.env_fn = env_fn
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize server policy
        self.server_policy = PolicyNetwork(state_dim, action_dim)
        self.theta_tilde = [self.server_policy.get_parameters().clone()]  # List of snapshots
        
        # Initialize Byzantine filter
        self.byzantine_filter = ByzantineFilter(
            sigma=config.variance_bound,
            delta=config.confidence_param,
            K=config.num_agents,
            alpha=config.byzantine_ratio
        )
        
        # Statistics tracking
        self.stats = defaultdict(list)
        
    def train(self) -> Dict:
        """
        Main training loop - Algorithm 1 lines 2-13
        """
        T = self.config.num_rounds
        B_t = self.config.batch_size
        b_t = self.config.mini_batch_size
        eta_t = self.config.step_size
        gamma = self.config.discount_factor
        
        print(f"Starting FedPG-BR training: {T} rounds, {self.config.num_agents} agents")
        print(f"Byzantine ratio: {self.config.byzantine_ratio:.2%}")
        
        for t in range(1, T + 1):
            # Line 3: theta_t_0 <- theta_tilde_{t-1}
            theta_t_0 = self.theta_tilde[-1].clone()
            self.server_policy.set_parameters(theta_t_0)
            
            # Lines 4-6: Each agent samples trajectories and computes gradient
            agent_gradients = []
            for k in range(self.config.num_agents):
                mu_k_t = self._agent_compute_gradient(k, B_t, theta_t_0, gamma)
                agent_gradients.append(mu_k_t)
            
            # Line 7: Aggregate with Byzantine filtering
            mu_t, good_agents = self.byzantine_filter.aggregate(agent_gradients, B_t)
            
            # Line 8: Sample number of steps N_t from geometric distribution
            p_geom = B_t / (B_t + b_t)
            N_t = np.random.geometric(p_geom)
            
            # Lines 9-12: SCSG inner loop
            theta_t_n = theta_t_0.clone()
            for n in range(N_t):
                # Line 10: Sample mini-batch trajectories at server
                v_t_n = self._compute_semi_stochastic_gradient(
                    theta_t_n, theta_t_0, mu_t, b_t, gamma
                )
                
                # Line 12: Update policy
                theta_t_n = theta_t_n + eta_t * v_t_n
            
            # Line 13: Update server snapshot
            self.theta_tilde.append(theta_t_n)
            self.server_policy.set_parameters(theta_t_n)
            
            # Evaluate and log
            if t % 10 == 0 or t == 1:
                avg_reward = self._evaluate_policy()
                self.stats['round'].append(t)
                self.stats['avg_reward'].append(avg_reward)
                self.stats['num_good_agents'].append(len(good_agents))
                print(f"Round {t}/{T}: Avg Reward = {avg_reward:.2f}, "
                      f"Good Agents = {len(good_agents)}/{self.config.num_agents}")
        
        # Line 14: Return uniformly random snapshot
        idx = np.random.randint(1, len(self.theta_tilde))
        self.server_policy.set_parameters(self.theta_tilde[idx])
        
        return dict(self.stats)
    
    def _agent_compute_gradient(self, agent_id: int, B_t: int, theta_t_0: torch.Tensor, 
                               gamma: float) -> torch.Tensor:
        """
        Simulate agent k computing gradient (Lines 5-6 of Algorithm 1)
        
        For Byzantine agents, return adversarial gradient based on attack type
        """
        # Determine if this agent is Byzantine
        num_byzantine = int(self.config.byzantine_ratio * self.config.num_agents)
        is_byzantine = agent_id < num_byzantine
        
        if is_byzantine:
            return self._generate_byzantine_gradient(theta_t_0)
        
        # Good agent: compute honest gradient
        self.server_policy.set_parameters(theta_t_0)
        
        gradients = []
        for _ in range(B_t):
            env = self.env_fn()
            trajectory, _ = sample_trajectory(env, self.server_policy)
            grad = compute_gpomdp_gradient(trajectory, self.server_policy, gamma)
            gradients.append(grad)
            env.close()
        
        # mu_k_t = (1/B_t) * sum of gradients
        mu_k_t = torch.mean(torch.stack(gradients), dim=0)
        return mu_k_t
    
    def _generate_byzantine_gradient(self, theta_t_0: torch.Tensor) -> torch.Tensor:
        """
        Generate Byzantine gradient (for testing fault tolerance)
        Different attack types: random noise, sign flipping, etc.
        """
        # Simple random noise attack
        param_dim = len(theta_t_0)
        return torch.randn(param_dim) * self.config.variance_bound * 10
    
    def _compute_semi_stochastic_gradient(self, theta_t_n: torch.Tensor, theta_t_0: torch.Tensor,
                                         mu_t: torch.Tensor, b_t: int, gamma: float) -> torch.Tensor:
        """
        Compute semi-stochastic gradient (Line 11 of Algorithm 1)
        
        v_t_n = (1/b_t) * sum_j [g(tau_j|theta_n) - omega(tau_j|theta_n, theta_0) * g(tau_j|theta_0)] + mu_t
        """
        # Create policies for theta_n and theta_0
        policy_n = PolicyNetwork(self.state_dim, self.action_dim)
        policy_n.set_parameters(theta_t_n)
        
        policy_0 = PolicyNetwork(self.state_dim, self.action_dim)
        policy_0.set_parameters(theta_t_0)
        
        corrections = []
        for _ in range(b_t):
            env = self.env_fn()
            
            # Sample trajectory using theta_n
            trajectory, _ = sample_trajectory(env, policy_n)
            
            # Compute g(tau|theta_n)
            g_theta_n = compute_gpomdp_gradient(trajectory, policy_n, gamma)
            
            # Compute importance weight omega(tau|theta_n, theta_0)
            omega = compute_importance_weight(trajectory, policy_n, policy_0)
            
            # Compute g(tau|theta_0)
            g_theta_0 = compute_gpomdp_gradient(trajectory, policy_0, gamma)
            
            # Correction term: g(tau|theta_n) - omega * g(tau|theta_0)
            correction = g_theta_n - omega * g_theta_0
            corrections.append(correction)
            
            env.close()
        
        # Average corrections and add mu_t
        avg_correction = torch.mean(torch.stack(corrections), dim=0)
        v_t_n = avg_correction + mu_t
        
        return v_t_n
    
    def _evaluate_policy(self, num_episodes: int = 10) -> float:
        """Evaluate current policy over multiple episodes"""
        total_rewards = []
        for _ in range(num_episodes):
            env = self.env_fn()
            _, reward = sample_trajectory(env, self.server_policy)
            total_rewards.append(reward)
            env.close()
        return np.mean(total_rewards)


def main():
    """Example usage"""
    # Create environment factory
    def make_env():
        return gym.make('CartPole-v1')
    
    # Get dimensions from environment
    env = make_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    # Configure FedPG-BR
    config = FedPGConfig(
        num_rounds=50,
        batch_size=16,
        mini_batch_size=4,
        step_size=1e-3,
        num_agents=10,
        byzantine_ratio=0.3,  # 30% Byzantine agents
        variance_bound=0.06,
        confidence_param=0.6
    )
    
    # Train
    fedpg = FedPGBR(make_env, config, state_dim, action_dim)
    stats = fedpg.train()
    
    print("\nTraining completed!")
    print(f"Final average reward: {stats['avg_reward'][-1]:.2f}")


if __name__ == "__main__":
    main()
