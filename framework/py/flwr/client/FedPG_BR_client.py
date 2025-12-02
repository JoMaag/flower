from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from typing import cast
import numpy as np

class FedPGClient(NumPyClient):
    def __init__(self, env, policy_network, cid: str):
        self.env = env
        self.policy = policy_network
        self.cid = cid
    def fit(self, parameters, config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Client does Algorithm 1 lines 5-6."""
        theta_t0 = parameters[0]
        Bt = int(config["Bt"])
        
        print(f"[Client {self.cid}] Sampling {Bt} trajectories...")
        
        # Line 5: Sample trajectories
        trajectories = [
            self._sample_trajectory(theta_t0) 
            for _ in range(Bt)
        ]
        
        # Line 6: Compute gradient
        mu_k = self._compute_batch_gradient(trajectories, theta_t0)
        
        # Metrics
        avg_length = np.mean([len(t['states']) for t in trajectories])
        avg_reward = np.mean([sum(t['rewards']) for t in trajectories])
        grad_norm = np.linalg.norm(mu_k)
        
        print(f"[Client {self.cid}] Avg length={avg_length:.1f}, reward={avg_reward:.2f}, grad_norm={grad_norm:.4f}")
        
        return [mu_k], len(trajectories), {
            "avg_trajectory_length": float(avg_length),
            "avg_trajectory_reward": float(avg_reward),
            "gradient_norm": float(grad_norm),
        }
    
    def _sample_trajectory(self, theta):
        """Sample trajectory from client's MDP."""
        self.policy.set_weights(theta)
        
        states, actions, rewards = [], [], []
        
        # Handle both Gym APIs
        reset_result = self.env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        
        max_steps = 1000
        steps = 0
        
        while not done and steps < max_steps:
            action = self.policy.sample_action(state)
            
            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            steps += 1
        
        return {'states': states, 'actions': actions, 'rewards': rewards}
    
    def _compute_batch_gradient(self, trajectories, theta):
        """μ = (1/B_t) Σ g(τ_i|θ)."""
        grads = []
        invalid_count = 0
        
        for tau in trajectories:
            grad = self._compute_gradient(tau, theta)
            
            if np.isnan(grad).any() or np.isinf(grad).any():
                invalid_count += 1
                continue
            
            grads.append(grad)
        
        if not grads:
            print(f"[Client {self.cid}] ERROR: All gradients invalid!")
            return np.zeros_like(theta)
        
        if invalid_count > 0:
            print(f"[Client {self.cid}] Filtered {invalid_count} invalid gradients")
        
        return np.mean(grads, axis=0)
    
    def _compute_gradient(self, trajectory, theta):
        """REINFORCE gradient with baseline."""
        states = trajectory['states']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        
        if len(states) == 0:
            return np.zeros_like(theta)
        
        # Returns
        gamma = 0.99
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        # Baseline
        baseline = np.mean(returns)
        
        # Gradient
        grad = np.zeros_like(theta)
        for s, a, G in zip(states, actions, returns):
            advantage = G - baseline
            grad += self.policy.log_prob_gradient(s, a, theta) * advantage
        
        return grad / len(states)
    def evaluate(self, parameters, config) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate policy."""
        theta = parameters[0]
        self.policy.set_weights(theta)
        
        num_eval = 10
        total_reward = 0
        
        for _ in range(num_eval):
            reset_result = self.env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            done = False
            ep_reward = 0
            steps = 0
            
            while not done and steps < 1000:
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
        return float(-avg), len(theta), cast(dict[str, Scalar], {"avg_reward": float(avg)})