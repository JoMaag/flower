"""Byzantine Attack Types from FedPG-BR Paper (Section 5).

Attack types implemented:
1. Random Noise (RN): Byzantine agent sends a random vector
2. Random Action (RA): Byzantine agent takes random actions (simulates hardware failure)
3. Sign Flipping (SF): Byzantine agent sends -2.5 * gradient
4. FedPG Attack: Sophisticated attack that tries to evade the Byzantine filter
5. Variance Attack (VA): Exploits high variance in gradient estimation (Appendix G.2)
"""

import numpy as np
import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class AttackConfig:
    """Configuration for Byzantine attacks."""
    # Sign Flipping
    sign_flip_scale: float = -2.5
    
    # Random Noise
    noise_scale: float = 3.0
    
    # Variance Attack (from Baruch et al.)
    va_z_max: float = 0.18
    
    # FedPG Attack
    fedpg_attack_scale: float = 3.0  # 3σ as mentioned in paper


# Global storage for coordinated attacks (FedPG attack requires coordination)
_byzantine_gradients: Dict[int, List[torch.Tensor]] = {}
_attack_round: int = 0


def reset_attack_state():
    """Reset global attack state for new round."""
    global _byzantine_gradients, _attack_round
    _byzantine_gradients = {}
    _attack_round += 1


def register_byzantine_gradient(worker_id: int, gradients: List[torch.Tensor]):
    """Register gradient from a Byzantine worker for coordinated attacks."""
    _byzantine_gradients[worker_id] = [g.clone() for g in gradients]


def get_byzantine_gradients() -> Dict[int, List[torch.Tensor]]:
    """Get all registered Byzantine gradients."""
    return _byzantine_gradients


class ByzantineAttack:
    """Base class for Byzantine attacks."""
    
    def __init__(self, config: Optional[AttackConfig] = None):
        self.config = config or AttackConfig()
    
    def apply(self, gradients: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        """Apply attack to gradients. Override in subclasses."""
        raise NotImplementedError


class RandomNoiseAttack(ByzantineAttack):
    """Random Noise (RN) Attack.
    
    From paper Section 5:
    "each Byzantine agent sends a random vector to the server"
    
    This completely replaces the gradient with random noise.
    """
    
    def apply(self, gradients: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        attacked = []
        for grad in gradients:
            # Generate random vector with similar magnitude to gradients
            magnitude = grad.abs().mean().item() * self.config.noise_scale
            if magnitude < 1e-6:
                magnitude = 1.0
            random_grad = (torch.rand_like(grad) * 2 - 1) * magnitude
            attacked.append(random_grad)
        return attacked


class RandomActionAttack(ByzantineAttack):
    """Random Action (RA) Attack.
    
    From paper Section 5:
    "every Byzantine agent ignores the policy from the server and takes 
    actions randomly, which is used to simulate random system failures 
    (e.g., hardware failures) and results in false gradient computations 
    since the trajectories are no longer sampled according to the policy"
    
    Note: This attack is applied during trajectory sampling, not gradient manipulation.
    The gradient returned is the "honest" gradient from random trajectories.
    """
    
    def apply(self, gradients: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        # For RA attack, gradients are already computed from random actions
        # during trajectory sampling, so we just return them
        return gradients
    
    @staticmethod
    def get_random_action(action_space) -> Any:
        """Get a random action from the action space."""
        return action_space.sample()


class SignFlippingAttack(ByzantineAttack):
    """Sign Flipping (SF) Attack.
    
    From paper Section 5:
    "each Byzantine agent computes the correct gradient but sends the 
    scaled negative gradient (multiplied by −2.5), which is used to 
    simulate adversarial attacks aiming to manipulate the direction 
    of policy update at the server"
    """
    
    def apply(self, gradients: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        return [self.config.sign_flip_scale * grad for grad in gradients]


class FedPGAttack(ByzantineAttack):
    """FedPG Attack - Sophisticated attack designed to evade Byzantine filter.
    
    From paper Section 5:
    "FedPG attackers firstly estimate ∇J(θ_0^t) using the mean of their 
    gradients μ̄_t, and estimate σ by calculating the maximum Euclidean 
    distance between the gradients of any two FedPG attackers as 2σ̄. 
    Next, all FedPG attackers send the vector μ̄_t + 3σ̄ to the server."
    
    This attack requires coordination between Byzantine agents.
    """
    
    def apply(self, gradients: List[torch.Tensor], worker_id: int = 0, **kwargs) -> List[torch.Tensor]:
        # Register this worker's gradient
        register_byzantine_gradient(worker_id, gradients)
        
        # Get all Byzantine gradients
        all_byz_grads = get_byzantine_gradients()
        
        if len(all_byz_grads) < 2:
            # Not enough Byzantine agents to coordinate, fall back to sign flip
            return [self.config.sign_flip_scale * grad for grad in gradients]
        
        # Compute mean gradient across all Byzantine agents
        num_params = len(gradients)
        mean_grads = []
        
        for param_idx in range(num_params):
            param_grads = [all_byz_grads[wid][param_idx] for wid in all_byz_grads]
            mean_grad = torch.mean(torch.stack(param_grads), dim=0)
            mean_grads.append(mean_grad)
        
        # Estimate σ as max distance / 2 between any two Byzantine gradients
        max_distance = 0.0
        worker_ids = list(all_byz_grads.keys())
        
        for i, wid1 in enumerate(worker_ids):
            for wid2 in worker_ids[i+1:]:
                # Compute distance for flattened gradients
                flat1 = torch.cat([g.flatten() for g in all_byz_grads[wid1]])
                flat2 = torch.cat([g.flatten() for g in all_byz_grads[wid2]])
                dist = torch.norm(flat1 - flat2).item()
                max_distance = max(max_distance, dist)
        
        sigma_estimate = max_distance / 2.0
        
        # Create attack vector: μ̄_t + 3σ̄ in the direction away from mean
        attacked = []
        for param_idx, mean_grad in enumerate(mean_grads):
            # Add 3σ in a consistent direction (using gradient direction)
            direction = mean_grad / (torch.norm(mean_grad) + 1e-8)
            attack_grad = mean_grad + self.config.fedpg_attack_scale * sigma_estimate * direction
            attacked.append(attack_grad)
        
        return attacked


class VarianceAttack(ByzantineAttack):
    """Variance Attack (VA) from Baruch et al.
    
    From paper Appendix G.2:
    "The VA attackers collude together to estimate the population mean 
    and the standard-deviation of gradients at each round, and move the 
    mean by the largest value such that their values are still within 
    the population variance."
    
    This attack exploits high variance in gradient estimation.
    """
    
    def apply(self, gradients: List[torch.Tensor], worker_id: int = 0, **kwargs) -> List[torch.Tensor]:
        # Register this worker's gradient
        register_byzantine_gradient(worker_id, gradients)
        
        all_byz_grads = get_byzantine_gradients()
        
        if len(all_byz_grads) < 2:
            # Fall back to noise attack
            return RandomNoiseAttack(self.config).apply(gradients)
        
        # Estimate population statistics from Byzantine gradients
        num_params = len(gradients)
        attacked = []
        
        for param_idx in range(num_params):
            param_grads = torch.stack([all_byz_grads[wid][param_idx] for wid in all_byz_grads])
            
            # Estimate mean and std
            mean_grad = param_grads.mean(dim=0)
            std_grad = param_grads.std(dim=0) + 1e-8
            
            # Push gradient to edge of estimated variance
            # z_max controls how aggressive the attack is
            attack_grad = mean_grad + self.config.va_z_max * std_grad
            attacked.append(attack_grad)
        
        return attacked


class ZeroGradientAttack(ByzantineAttack):
    """Zero Gradient Attack - Simple attack that sends zero gradients."""
    
    def apply(self, gradients: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        return [torch.zeros_like(grad) for grad in gradients]


class RewardFlippingAttack(ByzantineAttack):
    """Reward Flipping Attack - Negates rewards during training.
    
    Note: This is applied during trajectory collection, not gradient manipulation.
    """
    
    def apply(self, gradients: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        # Gradients already computed with flipped rewards
        return gradients


# Attack type registry
ATTACK_REGISTRY = {
    "random-noise": RandomNoiseAttack,
    "random-action": RandomActionAttack,
    "sign-flip": SignFlippingAttack,
    "sign-flipping": SignFlippingAttack,  # Alias
    "fedpg-attack": FedPGAttack,
    "variance-attack": VarianceAttack,
    "zero-gradient": ZeroGradientAttack,
    "reward-flipping": RewardFlippingAttack,
}

# Paper attack types (from Section 5)
PAPER_ATTACK_TYPES = ["random-noise", "random-action", "sign-flip", "fedpg-attack"]


def get_attack(attack_type: str, config: Optional[AttackConfig] = None) -> ByzantineAttack:
    """Get attack instance by type name."""
    if attack_type not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack type: {attack_type}. "
                        f"Available: {list(ATTACK_REGISTRY.keys())}")
    return ATTACK_REGISTRY[attack_type](config)


def apply_attack(attack_type: str, gradients: List[torch.Tensor], 
                 worker_id: int = 0, config: Optional[AttackConfig] = None) -> List[torch.Tensor]:
    """Apply attack to gradients."""
    attack = get_attack(attack_type, config)
    return attack.apply(gradients, worker_id=worker_id)
