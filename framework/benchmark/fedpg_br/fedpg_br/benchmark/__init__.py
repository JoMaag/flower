"""FedPG-BR Benchmark Framework.

A gym-like benchmark infrastructure for federated reinforcement learning
with Byzantine resilience testing.
"""

__version__ = "0.1.0"

# Import metrics_registry for easy access
from . import metrics_registry

__all__ = ["metrics_registry"]
