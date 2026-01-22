"""
Byzantine Filtering for FedPG-BR (Algorithm 1.1: FedPG-Aggregate)

Implements two-rule filtering:
- R1: Tighter threshold based on concentration (probabilistic)
- R2: Fallback with 2σ threshold (deterministic guarantee)
"""

import logging
import numpy as np
import torch
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ByzantineFilter:
    """
    Byzantine-resilient gradient aggregation.
    
    Filters out Byzantine gradients using vector median approach
    and aggregates remaining gradients.
    """
    
    def __init__(
        self,
        sigma: float,
        delta: float,
        num_agents: int,
        alpha: float,
    ):
        """
        Args:
            sigma: Variance bound on gradient estimator
            delta: Confidence parameter for R1 threshold
            num_agents: Total number of agents (K)
            alpha: Byzantine ratio (must be < 0.5)
        """
        if alpha >= 0.5:
            raise ValueError(f"alpha must be < 0.5, got {alpha}")
        
        self.sigma = sigma
        self.delta = delta
        self.num_agents = num_agents
        self.alpha = alpha
    
    def aggregate(
        self,
        gradients: List[torch.Tensor],
        batch_size: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Aggregate gradients with Byzantine filtering.
        
        Algorithm 1.1 from paper:
        1. Try R1 (tight threshold based on concentration)
        2. If R1 fails, fall back to R2 (2σ threshold)
        3. Aggregate gradients from filtered set G_t
        
        Args:
            gradients: List of gradient tensors from agents
            batch_size: B_t used this round (affects R1 threshold)
            
        Returns:
            (aggregated_gradient, list_of_good_agent_indices)
        """
        K = len(gradients)
        
        # Compute thresholds
        V = 2 * np.log(2 * K / self.delta)
        T_mu = 2 * self.sigma * np.sqrt(V / batch_size)  # R1 threshold
        threshold_r2 = 2 * self.sigma                     # R2 threshold
        
        # Try R1 first (tighter threshold)
        good_agents = self._filter(gradients, T_mu)
        
        # Fall back to R2 if R1 doesn't include enough agents
        min_good = (1 - self.alpha) * K
        if len(good_agents) < min_good:
            logger.debug(f"R1 failed ({len(good_agents)} < {min_good:.0f}), using R2")
            good_agents = self._filter(gradients, threshold_r2)
        
        # Aggregate
        if good_agents:
            aggregated = torch.mean(
                torch.stack([gradients[i] for i in good_agents]), 
                dim=0
            )
        else:
            logger.warning("No agents passed filter, using mean of all")
            aggregated = torch.mean(torch.stack(gradients), dim=0)
        
        return aggregated, good_agents
    
    def _filter(
        self,
        gradients: List[torch.Tensor],
        threshold: float,
    ) -> List[int]:
        """
        Apply filtering rule (R1 or R2).
        
        Algorithm 1.1 lines 2-4 (R1) or 6-8 (R2):
        1. Find S: gradients with >K/2 neighbors within threshold
        2. Find μ_mom: gradient in S closest to mean(S)
        3. Return G_t: agents within threshold of μ_mom
        
        Args:
            gradients: List of gradient tensors
            threshold: Distance threshold (T_μ for R1, 2σ for R2)
            
        Returns:
            List of good agent indices
        """
        K = len(gradients)
        
        # Step 1: Find S (vector medians)
        S_indices = []
        for i in range(K):
            neighbors = sum(
                1 for j in range(K)
                if torch.norm(gradients[i] - gradients[j]).item() <= threshold
            )
            if neighbors > K / 2:
                S_indices.append(i)
        
        if not S_indices:
            return list(range(K))  # Fallback: all agents
        
        # Step 2: Find μ_mom (mean of median)
        S_grads = [gradients[i] for i in S_indices]
        mean_S = torch.mean(torch.stack(S_grads), dim=0)
        
        mu_mom_idx = min(
            S_indices,
            key=lambda i: torch.norm(gradients[i] - mean_S).item()
        )
        mu_mom = gradients[mu_mom_idx]
        
        # Step 3: Find G_t (agents within threshold of μ_mom)
        good_agents = [
            i for i in range(K)
            if torch.norm(gradients[i] - mu_mom).item() <= threshold
        ]
        
        return good_agents
