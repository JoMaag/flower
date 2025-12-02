import numpy as np

class CategoricalPolicy:
    """
    Simple linear policy for discrete action spaces (like CartPole).
    π_θ(a|s) = softmax(W @ s + b)
    
    Parameters:
        θ = [W.flatten(), b]
        W: (action_dim, state_dim)
        b: (action_dim,)
    """
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Parameter dimensions
        self.W_size = state_dim * action_dim
        self.b_size = action_dim
        self.param_size = self.W_size + self.b_size
        
        self.theta = None
    
    def set_weights(self, theta):
        """Set policy parameters θ."""
        assert len(theta) == self.param_size, \
            f"Expected {self.param_size} params, got {len(theta)}"
        self.theta = theta.copy()
    
    def _get_W_b(self, theta):
        """Extract W and b from flattened θ."""
        W = theta[:self.W_size].reshape(self.action_dim, self.state_dim)
        b = theta[self.W_size:]
        return W, b
    
    def _compute_logits(self, state, theta):
        """Compute action logits: W @ s + b."""
        W, b = self._get_W_b(theta)
        return W @ state + b
    
    def _softmax(self, logits):
        """Numerically stable softmax."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()
    
    def sample_action(self, state):
        """Sample action from π(·|state)."""
        logits = self._compute_logits(state, self.theta)
        probs = self._softmax(logits)
        return np.random.choice(self.action_dim, p=probs)
    
    def log_prob(self, state, action, theta):
        """Compute log π_θ(action|state)."""
        logits = self._compute_logits(state, theta)
        # Log-softmax: log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
        log_sum_exp = np.log(np.sum(np.exp(logits)))
        return logits[action] - log_sum_exp
    
    def log_prob_gradient(self, state, action, theta):
        """
        Compute ∇_θ log π_θ(action|state).
        
        For categorical policy with softmax:
        ∇ log π(a|s) = ∇ logits[a] - Σ_a' π(a'|s) ∇ logits[a']
        
        For linear policy (logits = Ws + b):
        ∇_W logits[a] = e_a ⊗ s  (outer product)
        ∇_b logits[a] = e_a
        
        Where e_a is one-hot vector for action a.
        """
        logits = self._compute_logits(state, theta)
        probs = self._softmax(logits)
        
        # Initialize gradient
        grad = np.zeros(self.param_size)
        
        # ∇_W log π(action|state)
        # For each action a', compute contribution to gradient
        W_grad = np.zeros((self.action_dim, self.state_dim))
        
        for a in range(self.action_dim):
            if a == action:
                # ∇_W logits[action] with weight (1 - π(action|s))
                W_grad[a] = (1 - probs[a]) * state
            else:
                # ∇_W logits[a] with weight (-π(a|s))
                W_grad[a] = -probs[a] * state
        
        # Flatten W gradient
        grad[:self.W_size] = W_grad.flatten()
        
        # ∇_b log π(action|state)
        # = e_action - probs
        b_grad = np.zeros(self.action_dim)
        b_grad[action] = 1.0
        b_grad -= probs
        
        grad[self.W_size:] = b_grad
        
        return grad


class GaussianPolicy:
    """
    Gaussian policy for continuous action spaces (like HalfCheetah).
    π_θ(a|s) = N(μ_θ(s), σ²)
    
    μ_θ(s) = W @ s + b
    σ is fixed
    
    Parameters:
        θ = [W.flatten(), b]
    """
    
    def __init__(self, state_dim, action_dim, log_std=-0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std = log_std
        self.std = np.exp(log_std)
        
        # Parameter dimensions
        self.W_size = state_dim * action_dim
        self.b_size = action_dim
        self.param_size = self.W_size + self.b_size
        
        self.theta = None
    
    def set_weights(self, theta):
        """Set policy parameters θ."""
        assert len(theta) == self.param_size
        self.theta = theta.copy()
    
    def _get_W_b(self, theta):
        """Extract W and b from flattened θ."""
        W = theta[:self.W_size].reshape(self.action_dim, self.state_dim)
        b = theta[self.W_size:]
        return W, b
    
    def _compute_mean(self, state, theta):
        """Compute mean action: μ = W @ s + b."""
        W, b = self._get_W_b(theta)
        return W @ state + b
    
    def sample_action(self, state):
        """Sample action from π(·|state) = N(μ(s), σ²)."""
        mean = self._compute_mean(state, self.theta)
        return mean + self.std * np.random.randn(self.action_dim)
    
    def log_prob(self, state, action, theta):
        """Compute log π_θ(action|state)."""
        mean = self._compute_mean(state, theta)
        # Log probability of Gaussian
        # log N(a|μ,σ²) = -0.5 * [(a-μ)²/σ² + log(2πσ²)]
        diff = action - mean
        log_prob = -0.5 * (
            np.sum((diff / self.std) ** 2) +
            self.action_dim * np.log(2 * np.pi * self.std ** 2)
        )
        return log_prob
    
    def log_prob_gradient(self, state, action, theta):
        """
        Compute ∇_θ log π_θ(action|state).
        
        ∇ log π(a|s) = ∇μ_θ(s) * (a - μ_θ(s)) / σ²
        
        For linear mean: μ = Ws + b
        ∇_W μ = I ⊗ s  (for each output dimension)
        ∇_b μ = I
        """
        mean = self._compute_mean(state, theta)
        diff = (action - mean) / (self.std ** 2)
        
        # Initialize gradient
        grad = np.zeros(self.param_size)
        
        # ∇_W log π(a|s)
        # For each action dimension i: ∇_W[i,:] = diff[i] * s
        W_grad = np.outer(diff, state)
        grad[:self.W_size] = W_grad.flatten()
        
        # ∇_b log π(a|s) = diff
        grad[self.W_size:] = diff
        
        return grad