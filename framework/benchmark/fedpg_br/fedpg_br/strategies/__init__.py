"""Strategy plugin system for FedPG-BR.

Users can implement custom aggregation strategies by subclassing
AggregationStrategy and registering them with @register_strategy.

Example:
    from fedpg_br.strategies import AggregationStrategy, register_strategy

    @register_strategy("my-method")
    class MyStrategy(AggregationStrategy):
        def aggregate(self, gradients, batch_size, **kwargs):
            # Your custom aggregation logic
            return torch.mean(torch.stack(gradients), dim=0), list(range(len(gradients)))

        def server_update(self, policy, optimizer, theta_t_0, mu_t, config):
            # Your custom server update logic (single gradient step, SCSG, etc.)
            apply_gradient(policy, optimizer, mu_t)
            return 1  # number of steps taken

Then in your config TOML:
    method = "my-method"
"""

from fedpg_br.strategies.base import (
    AggregationStrategy,
    register_strategy,
    get_strategy,
    list_strategies,
)

__all__ = [
    "AggregationStrategy",
    "register_strategy",
    "get_strategy",
    "list_strategies",
]

# Import built-in strategies to trigger registration
import fedpg_br.strategies.gomdp
import fedpg_br.strategies.svrpg
import fedpg_br.strategies.fedpg_br_strategy
