"""
FedPG-BR Server Strategy - FIXED TO MATCH PAPER CODE
====================================================
All bugs fixed based on comparison with working paper implementation.
"""

from flwr.server.strategy import Strategy
from flwr.common import (
    Parameters, 
    FitIns, 
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import Optional, Union, List, Dict, Tuple
import numpy as np
import torch
import torch.optim as optim


class FedPGBR(Strategy):
    """
    FedPG-BR: Byzantine-Resilient Federated Policy Gradient
    
    FIXED VERSION - Matches paper code exactly.
    """
    
    def __init__(
        self,
        initial_parameters: np.ndarray,
        server_environment,
        policy_network,
        batch_size: int = 16,
        minibatch_size: int = 4,
        learning_rate: float = 0.001,
        variance_bound: float = 0.06,
        byzantine_fraction: float = 0.0,
        confidence_level: float = 0.6
    ):
        # Store numpy version for Flower
        self.theta_tilde_np = initial_parameters.copy()
        
        # PyTorch policy network
        self.policy = policy_network
        self.policy.set_weights(initial_parameters)
        
        # Server environment
        self.env = server_environment
        
        # Hyperparameters (EXACT paper values)
        self.Bt = batch_size
        self.bt = minibatch_size
        self.eta_t = learning_rate
        self.sigma = variance_bound
        self.alpha = byzantine_fraction
        self.delta = confidence_level
        
        # PyTorch Adam Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.eta_t
        )
        
        # Aggregated gradient
        self.mu_t = None
        self.current_round = 0
        
        print("="*60)
        print("FedPG-BR Server Strategy (FIXED)")
        print("="*60)
        print(f"Policy parameters: {self.policy.param_size}")
        print(f"Batch size (B_t): {self.Bt}")
        print(f"Mini-batch size (b_t): {self.bt}")
        print(f"Learning rate (η_t): {self.eta_t}")
        print(f"Optimizer: Adam")
        print(f"Byzantine tolerance (α): {self.alpha}")
        print("="*60)
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        print("\n[Step 1] Initializing global parameters θ̃₀")
        return ndarrays_to_parameters([self.theta_tilde_np])
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self.current_round = server_round
        
        print(f"\n{'='*60}")
        print(f"Round {server_round}: Configure Fit")
        print(f"{'='*60}")
        
        clients = list(client_manager.all().values())
        
        config: Dict[str, Scalar] = {
            "server_round": server_round,
            "Bt": self.Bt,
        }
        
        fit_ins = FitIns(parameters=parameters, config=config)
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}
        
        # Collect gradients
        print(f"\n[Step 3] Collecting gradients from {len(results)} clients...")
        gradients = []
        
        for client_proxy, fit_res in results:
            gradient = parameters_to_ndarrays(fit_res.parameters)[0]
            gradients.append(gradient)
            
            if fit_res.metrics:
                grad_norm = fit_res.metrics.get("gradient_norm", 0)
                avg_reward = fit_res.metrics.get("avg_trajectory_reward", 0)
                print(f"  Client {client_proxy.cid}: "
                      f"grad_norm={grad_norm:.4f}, avg_reward={avg_reward:.2f}")
        
        # Byzantine filter
        print("\n[Step 4] Byzantine-resilient aggregation...")
        self.mu_t = self._fedpg_aggregate(gradients)
        print(f"Aggregated gradient norm: {np.linalg.norm(self.mu_t):.4f}")
        
        # SCSG Inner Loop
        print("\n[Step 5] SCSG Inner Loop (PyTorch Adam)...")
        self._inner_loop_torch()
        
        # Update numpy version
        self.theta_tilde_np = self.policy.get_weights()
        print(f"Updated policy norm: {np.linalg.norm(self.theta_tilde_np):.4f}")
        
        metrics = {}
        
        print(f"\n{'='*60}")
        print(f"Round {server_round} Complete")
        print(f"{'='*60}\n")
        
        return ndarrays_to_parameters([self.theta_tilde_np]), metrics
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Optional: Configure evaluation round."""
        # We don't do evaluation in this implementation
        return []
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Optional: Aggregate evaluation results."""
        return None, {}
    
    def evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Optional: Server-side evaluation."""
        return None
    
    # =========================================================================
    # BYZANTINE FILTER
    # =========================================================================
    
    def _fedpg_aggregate(self, gradients: List[np.ndarray]) -> np.ndarray:
        """Byzantine-resilient aggregation."""
        K = len(gradients)
        V = 2 * np.log(2 * K / self.delta)
        T_mu = 2 * self.sigma * np.sqrt(V / self.Bt)
        
        print(f"  Filter threshold T_μ = {T_mu:.4f}")
        
        S1 = self._build_vector_median_set(gradients, T_mu)
        
        if S1:
            mu_mom = self._find_mean_of_median(S1)
            Gt = [g for g in gradients if np.linalg.norm(g - mu_mom) <= T_mu]
        else:
            Gt = []
        
        if len(Gt) < (1 - self.alpha) * K:
            S2 = self._build_vector_median_set(gradients, 2 * self.sigma)
            if S2:
                mu_mom = self._find_mean_of_median(S2)
                Gt = [g for g in gradients if np.linalg.norm(g - mu_mom) <= 2 * self.sigma]
            else:
                Gt = gradients
        
        if not Gt:
            Gt = gradients
        
        return np.mean(Gt, axis=0)
    
    def _build_vector_median_set(self, gradients, threshold):
        K = len(gradients)
        S = []
        for g in gradients:
            neighbors = sum(1 for g2 in gradients if np.linalg.norm(g - g2) <= threshold)
            if neighbors > K / 2:
                S.append(g)
        return S
    
    def _find_mean_of_median(self, S):
        if not S:
            raise ValueError("S is empty!")
        mean_S = np.mean(S, axis=0)
        return min(S, key=lambda g: np.linalg.norm(g - mean_S))
    
    # =========================================================================
    # SCSG INNER LOOP - FIXED VERSION
    # =========================================================================
    
    def _inner_loop_torch(self):
        """
        SCSG Inner Loop - FIXED TO MATCH PAPER CODE
        
        KEY FIX: p = 1 - B/(B+b) [PAPER CODE VERSION]
        """
        # ===== FIX 1: Correct geometric probability =====
        # Paper uses: p = B/(B+b) but calls it with (1-p)!
        # So effectively: p_effective = 1 - B/(B+b) = b/(B+b)
        p = self.bt / (self.Bt + self.bt)  # = 4/20 = 0.2
        N_t = np.random.geometric(p)
        
        print(f"  Inner loop: {N_t} iterations (p={p:.3f})")
        
        # Paper: if N_t == 0, return unchanged policy
        if N_t == 0:
            print("  Skipping inner loop (N_t=0)")
            return
        
        # Save reference policy θ_0
        theta_0_np = self.policy.get_weights().copy()
        
        for n in range(N_t):
            # Sample b_t trajectories
            trajectories = [
                self._sample_trajectory_torch() 
                for _ in range(self.bt)
            ]
            
            # Compute variance-reduced gradient
            v_tn = self._compute_vr_gradient_torch(trajectories, theta_0_np)
            
            # Set gradients manually
            self.optimizer.zero_grad()
            idx = 0
            for p in self.policy.parameters():
                numel = p.numel()
                p.grad = v_tn[idx:idx+numel].view(p.shape)
                idx += numel
            
            # PyTorch Adam step
            self.optimizer.step()
    
    # =========================================================================
    # TRAJECTORY SAMPLING
    # =========================================================================
    
    def _sample_trajectory_torch(self) -> dict:
        """Sample trajectory - FIXED gamma and max_steps."""
        self.policy.eval()
        
        states, actions, rewards = [], [], []
        
        reset_result = self.env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        steps = 0
        
        # ===== FIX 2: max_steps = 500 (paper setting) =====
        max_steps = 500
        
        with torch.no_grad():
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
        
        self.policy.train()
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards
        }
    
    # =========================================================================
    # VARIANCE-REDUCED GRADIENT - FIXED VERSION
    # =========================================================================
    
    def _compute_vr_gradient_torch(
        self,
        trajectories: List[dict],
        theta_0: np.ndarray
    ) -> torch.Tensor:
        """
        Variance-reduced gradient - FIXED importance weights.
        
        v_t_n = (1/b_t) Σ [g(τ|θ_n) - ω(τ) g(τ|θ_0)] + μ_t
        """
        corrections = []
        
        for tau in trajectories:
            # g(τ | θ_n) - gradient at current policy
            g_current = self._policy_gradient_torch(tau)
            
            # ===== FIX 3: Importance weight calculation =====
            # Save current weights
            current_weights = self.policy.get_weights()
            
            # Compute importance weight ω(τ | θ_current, θ_0)
            omega = self._importance_weight_torch(tau, current_weights, theta_0)
            
            # g(τ | θ_0) - gradient at reference policy
            self.policy.set_weights(theta_0)
            g_reference = self._policy_gradient_torch(tau)
            
            # Restore current policy
            self.policy.set_weights(current_weights)
            
            # Correction: g_current - ω * g_reference
            corrections.append(g_current - omega * g_reference)
        
        # Average corrections
        correction = torch.stack(corrections).mean(0)
        
        # Add aggregated gradient μ_t
        mu_t_torch = torch.from_numpy(self.mu_t).float() if self.mu_t is not None else torch.zeros_like(correction)
        
        return correction + mu_t_torch
    
    def _policy_gradient_torch(self, trajectory: dict) -> torch.Tensor:
        """
        REINFORCE gradient - FIXED gamma = 0.999
        """
        states = trajectory['states']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        
        if not states:
            return torch.zeros(self.policy.param_size)
        
        # ===== FIX 4: gamma = 0.999 (paper setting) =====
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
    
    def _importance_weight_torch(
        self,
        trajectory: dict,
        theta_current: np.ndarray,
        theta_0: np.ndarray
    ) -> float:
        """
        Importance weight - FIXED clipping.
        
        ω = p(τ|θ_0) / p(τ|θ_current)
        """
        states = trajectory['states']
        actions = trajectory['actions']
        
        if not states:
            return 1.0
        
        # Log prob under current policy
        self.policy.set_weights(theta_current)
        log_p_current = sum(
            self.policy.log_prob(s, a).item()
            for s, a in zip(states, actions)
        )
        
        # Log prob under reference policy
        self.policy.set_weights(theta_0)
        log_p_ref = sum(
            self.policy.log_prob(s, a).item()
            for s, a in zip(states, actions)
        )
        
        # Restore current policy
        self.policy.set_weights(theta_current)
        
        # ===== FIX 5: Proper clipping for stability =====
        log_omega = log_p_ref - log_p_current
        log_omega = np.clip(log_omega, -10, 10)
        
        return np.exp(log_omega)