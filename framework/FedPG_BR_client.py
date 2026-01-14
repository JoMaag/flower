# client.py - FINAL FIX

from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from typing import cast
import numpy as np
import torch


class FedPGClient(NumPyClient):
    """
    FedPG Client - FINAL FIXED VERSION
    
    KEY FIX: Sample trajectories WITH theta_t0, not current policy!
    """
    
    def __init__(self, env, policy_network, cid: str):
        self.env = env
        self.policy = policy_network
        self.cid = cid
        self.is_torch = hasattr(policy_network, 'parameters')
    
    def fit(self, parameters, config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Client training.
        
        CRITICAL: Sample trajectories WITH theta_t0!
        """
        theta_t0 = parameters[0]
        Bt = int(config["Bt"])
        
        # ===== FIX: Set weights BEFORE sampling! =====
        self.policy.set_weights(theta_t0)
        
        # Sample trajectories with theta_t0
        trajectories = [
            self._sample_trajectory(theta_t0)  # ← Pass theta!
            for _ in range(Bt)
        ]
        
        # Compute gradient at theta_t0
        mu_k = self._compute_batch_gradient(trajectories, theta_t0)
        
        # Metrics
        avg_length = np.mean([len(t['states']) for t in trajectories])
        avg_reward = np.mean([sum(t['rewards']) for t in trajectories])
        grad_norm = np.linalg.norm(mu_k)
        
        return [mu_k], len(trajectories), {
            "avg_trajectory_length": float(avg_length),
            "avg_trajectory_reward": float(avg_reward),
            "gradient_norm": float(grad_norm),
        }
    
    def _sample_trajectory(self, theta: np.ndarray) -> dict:
        """
        Sample trajectory WITH GIVEN theta.
        
        CRITICAL FIX: theta parameter!
        """
        # ===== FIX: Ensure policy uses theta =====
        self.policy.set_weights(theta)
        
        if self.is_torch:
            self.policy.eval()
        
        states, actions, rewards = [], [], []
        
        reset_result = self.env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        steps = 0
        max_steps = 500
        
        with torch.no_grad() if self.is_torch else torch.enable_grad():
            while not done and steps < max_steps:
                action = self.policy.sample_action(state)
                
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
                steps += 1
        
        if self.is_torch:
            self.policy.train()
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards
        }
    
    def _compute_batch_gradient(
        self, 
        trajectories: list[dict], 
        theta: np.ndarray
    ) -> np.ndarray:
        """Compute batch gradient."""
        if self.is_torch:
            return self._compute_batch_gradient_torch(trajectories, theta)
        else:
            return self._compute_batch_gradient_numpy(trajectories, theta)
    
    def _compute_batch_gradient_torch(
        self,
        trajectories: list[dict],
        theta: np.ndarray
    ) -> np.ndarray:
        """Compute batch gradient with PyTorch."""
        self.policy.set_weights(theta)
        self.policy.train()
        
        grads = []
        invalid_count = 0
        
        for tau in trajectories:
            grad = self._compute_gradient_torch(tau)
            
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                invalid_count += 1
                continue
            
            grads.append(grad)
        
        if not grads:
            return np.zeros_like(theta)
        
        if invalid_count > 0:
            print(f"[Client {self.cid}] Filtered {invalid_count} invalid gradients")
        
        avg_grad = torch.stack(grads).mean(0)
        return avg_grad.detach().numpy()
    
    def _compute_gradient_torch(self, trajectory: dict) -> torch.Tensor:
        """REINFORCE gradient - gamma = 0.999"""
        states = trajectory['states']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        
        if not states:
            return torch.zeros(self.policy.param_size)
        
        gamma = 0.999
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        baseline = returns.mean()
        advantages = returns - baseline
        
        # Compute policy gradient
        self.policy.zero_grad()
        
        log_probs = []
        for s, a in zip(states, actions):
            _, log_prob = self.policy(s, fixed_action=a)
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs)
        
        # REINFORCE loss
        loss = -(log_probs * advantages).mean()
        loss.backward()
        
        # Extract gradient
        grad = torch.cat([
            p.grad.flatten() if p.grad is not None 
            else torch.zeros_like(p.flatten())
            for p in self.policy.parameters()
        ])
        
        return grad
    
    def _compute_batch_gradient_numpy(
        self,
        trajectories: list[dict],
        theta: np.ndarray
    ) -> np.ndarray:
        """Fallback numpy version."""
        grads = []
        
        for tau in trajectories:
            grad = self._compute_gradient_numpy(tau, theta)
            if not (np.isnan(grad).any() or np.isinf(grad).any()):
                grads.append(grad)
        
        if not grads:
            return np.zeros_like(theta)
        
        return np.mean(grads, axis=0)
    
    def _compute_gradient_numpy(
        self, 
        trajectory: dict, 
        theta: np.ndarray
    ) -> np.ndarray:
        """REINFORCE gradient numpy - gamma = 0.999"""
        states = trajectory['states']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        
        if not states:
            return np.zeros_like(theta)
        
        gamma = 0.999
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        baseline = np.mean(returns)
        
        grad = np.zeros_like(theta)
        for s, a, G in zip(states, actions, returns):
            advantage = G - baseline
            grad += self.policy.log_prob_gradient(s, a, theta) * advantage
        
        return grad / len(states)
    
    def evaluate(
        self, 
        parameters, 
        config
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate policy."""
        theta = parameters[0]
        self.policy.set_weights(theta)
        
        if self.is_torch:
            self.policy.eval()
        
        num_eval = 10
        total_reward = 0
        
        with torch.no_grad() if self.is_torch else torch.enable_grad():
            for _ in range(num_eval):
                reset_result = self.env.reset()
                state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                done = False
                ep_reward = 0
                steps = 0
                
                while not done and steps < 500:
                    action = self.policy.sample_action(state)
                    step_result = self.env.step(action)
                    
                    if len(step_result) == 5:
                        state, reward, term, trunc, _ = step_result
                        done = term or trunc
                    else:
                        state, reward, done, _ = step_result
                    
                    ep_reward += reward
                    steps += 1
                
                total_reward += ep_reward
        
        avg = total_reward / num_eval
        
        if self.is_torch:
            self.policy.train()
        
        return float(-avg), len(theta), cast(dict[str, Scalar], {"avg_reward": float(avg)})