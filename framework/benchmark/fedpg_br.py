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


def compute_returns(trajectory: TrajectoryBuffer, gamma: float) -> torch.Tensor:
    """
    Compute discounted returns with reward-to-go
    Returns: tensor of shape [trajectory_length] with normalized advantages
    """
    rewards = trajectory.rewards
    returns = []
    R = 0.0
    
    # Compute returns in reverse order
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.FloatTensor(returns)
    
    # Normalize returns (advantage estimation)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns


def compute_policy_gradient(trajectory: TrajectoryBuffer, policy: PolicyNetwork, 
                           gamma: float, returns: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute policy gradient using the policy gradient theorem
    
    Returns: (gradient_vector, log_probs_tensor)
    
    This matches Betreuer's approach:
    - Compute log_probs for all timesteps
    - Weight by returns/advantages
    - Backpropagate to get gradients
    """
    if returns is None:
        returns = compute_returns(trajectory, gamma)
    
    # Compute log probabilities for the trajectory
    log_probs = []
    for state, action in zip(trajectory.states, trajectory.actions):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action_tensor = torch.tensor([action])
        log_prob = action_dist.log_prob(action_tensor)
        log_probs.append(log_prob)
    
    log_probs = torch.stack(log_probs)
    
    # Compute weighted loss
    loss = -(log_probs * returns).mean()
    
    # Backpropagate to get gradients
    policy.zero_grad()
    loss.backward()
    
    # Extract gradient as flattened vector
    gradient = torch.cat([p.grad.flatten().clone() for p in policy.parameters()])
    
    return gradient, log_probs


def compute_log_probs_fixed_actions(trajectory: TrajectoryBuffer, policy: PolicyNetwork) -> torch.Tensor:
    """
    Compute log probabilities for a trajectory with FIXED actions
    
    This is the KEY function for importance sampling!
    We evaluate policy_0 on the SAME actions that were taken by policy_n
    """
    log_probs = []
    for state, action in zip(trajectory.states, trajectory.actions):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action_tensor = torch.tensor([action])
        log_prob = action_dist.log_prob(action_tensor)
        log_probs.append(log_prob)
    
    return torch.stack(log_probs)


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
        if len(G_t) > 0:
            good_gradients = [gradients[i] for i in G_t]
            aggregated = torch.mean(torch.stack(good_gradients), dim=0)
        else:
            # Fallback: use mean of all gradients
            aggregated = torch.mean(torch.stack(gradients), dim=0)
        
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
    CORRECTED: Main implementation of Algorithm 1: FedPG-BR
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
        print(f"Batch size: B={B_t}, mini-batch: b={b_t}")
        
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
            # Note: np.random.geometric expects p = probability of success
            # Paper uses: N_t ~ Geom(B/(B+b))
            # Betreuer uses: p = 1 - B/(B+b) which is correct for numpy
            p_geom = 1 - B_t / (B_t + b_t)
            N_t = np.random.geometric(p_geom)
            
            # Lines 9-12: SCSG inner loop with CORRECTED importance sampling
            theta_t_n = theta_t_0.clone()
            
            # Store theta_0 policy for importance sampling
            policy_0 = PolicyNetwork(self.state_dim, self.action_dim)
            policy_0.set_parameters(theta_t_0)
            
            actual_steps = 0
            for n in range(N_t):
                # Line 10-11: Compute semi-stochastic gradient
                v_t_n, ratio_mean = self._compute_semi_stochastic_gradient_corrected(
                    theta_t_n, policy_0, mu_t, b_t, gamma
                )
                
                # CRITICAL: Early stopping if importance ratios diverge
                # Matches Betreuer's code (lines ~295-297)
                if abs(ratio_mean) < 0.995 or abs(ratio_mean) > 1.005:
                    print(f"  Round {t}, Step {n}: Early stop (ratio={ratio_mean:.4f})")
                    actual_steps = n
                    break
                
                # Line 12: Update policy
                theta_t_n = theta_t_n + eta_t * v_t_n
                actual_steps = n + 1
            
            # Line 13: Update server snapshot
            self.theta_tilde.append(theta_t_n)
            self.server_policy.set_parameters(theta_t_n)
            
            # Evaluate and log
            if t % 10 == 0 or t == 1:
                avg_reward = self._evaluate_policy()
                self.stats['round'].append(t)
                self.stats['avg_reward'].append(avg_reward)
                self.stats['num_good_agents'].append(len(good_agents))
                self.stats['actual_N_t'].append(actual_steps)
                print(f"Round {t}/{T}: Avg Reward = {avg_reward:.2f}, "
                      f"Good Agents = {len(good_agents)}/{self.config.num_agents}, "
                      f"N_t = {actual_steps}/{N_t}")
        
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
            grad, _ = compute_policy_gradient(trajectory, self.server_policy, gamma)
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
    
    def _compute_semi_stochastic_gradient_corrected(
        self, 
        theta_t_n: torch.Tensor, 
        policy_0: PolicyNetwork,
        mu_t: torch.Tensor, 
        b_t: int, 
        gamma: float
    ) -> Tuple[torch.Tensor, float]:
        """
        CORRECTED: Compute semi-stochastic gradient (Line 11 of Algorithm 1)
        
        This matches Betreuer's approach:
        v_t_n = grad_new - grad_old + mu_t
        
        where:
        - grad_new: gradient computed with policy_n on trajectories from policy_n
        - grad_old: gradient computed with policy_0 on SAME trajectories (fixed actions!)
                    weighted by importance ratios
        
        Returns: (v_t_n, mean_ratio) for early stopping check
        """
        # Create policy for theta_n
        policy_n = PolicyNetwork(self.state_dim, self.action_dim)
        policy_n.set_parameters(theta_t_n)
        
        all_grad_new = []
        all_grad_old = []
        all_ratios = []
        
        for _ in range(b_t):
            env = self.env_fn()
            
            # Step 1: Sample trajectory using policy_n
            trajectory, _ = sample_trajectory(env, policy_n)
            
            # Step 2: Compute returns for this trajectory
            returns = compute_returns(trajectory, gamma)
            
            # Step 3: Compute gradient for policy_n (grad_new)
            grad_new, log_probs_n = compute_policy_gradient(
                trajectory, policy_n, gamma, returns
            )
            all_grad_new.append(grad_new)
            
            # Step 4: Compute log_probs for policy_0 on SAME actions
            log_probs_0 = compute_log_probs_fixed_actions(trajectory, policy_0)
            
            # Step 5: Compute importance ratios (element-wise)
            # ratios = exp(log_prob_0 - log_prob_n)
            ratios = torch.exp(log_probs_0.detach() - log_probs_n.detach())
            all_ratios.append(ratios.mean().item())  # Track mean ratio for early stopping
            
            # Step 6: Compute weighted loss for policy_0
            # loss_old = -(log_probs_0 * returns * ratios).mean()
            loss_old = -(log_probs_0 * returns * ratios).mean()
            
            # Step 7: Backpropagate to get grad_old
            policy_0.zero_grad()
            loss_old.backward()
            grad_old = torch.cat([p.grad.flatten().clone() for p in policy_0.parameters()])
            all_grad_old.append(grad_old)
            
            env.close()
        
        # Average gradients
        avg_grad_new = torch.mean(torch.stack(all_grad_new), dim=0)
        avg_grad_old = torch.mean(torch.stack(all_grad_old), dim=0)
        mean_ratio = np.mean(all_ratios)
        
        # SCSG update: v_t_n = grad_new - grad_old + mu_t
        v_t_n = avg_grad_new - avg_grad_old + mu_t
        
        return v_t_n, mean_ratio
    
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
    print("\n" + "="*80)
    print("CORRECTED FedPG-BR Implementation")
    print("="*80)
    fedpg = FedPGBR(make_env, config, state_dim, action_dim)
    stats = fedpg.train()
    
    print("\nTraining completed!")
    print(f"Final average reward: {stats['avg_reward'][-1]:.2f}")


if __name__ == "__main__":
    main()


