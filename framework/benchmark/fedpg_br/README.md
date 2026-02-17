# FedPG-BR: Byzantine-Robust Federated Reinforcement Learning

A Flower-based implementation of [Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee](https://github.com/flint-xf-fan/Byzantine-Federated-RL) (Fan et al., NeurIPS 2021).

Extends the original paper with a plugin strategy system, web dashboard, Docker deployment, and adaptive communication.

## Installation

```bash
# Clone and install
pip install -e .

# For MuJoCo environments (HalfCheetah)
pip install gymnasium[mujoco]

# For Box2D environments (LunarLander)
pip install gymnasium[box2d]
```

## Quick Start

### Run an experiment

```bash
# CartPole with FedPG-BR (paper hyperparameters, ~2.5h)
flower-simulation --app . --num-supernodes 10 \
  --backend-config '{"client-resources": {"num-cpus": 1, "num-gpus": 0.0}}' \
  --run-config configs/paper_cartpole_final.toml

# Or shorter with flwr CLI
flwr run .
```

### Plot results (paper style)

```bash
# Single run
python plot_results.py output.txt --stats

# Multiple runs for confidence intervals
python plot_results.py run1.txt run2.txt run3.txt --output comparison.png
```

Plots use RBF interpolation with 90% confidence intervals, matching the original paper's Figure 2.

## Strategies

Three aggregation methods from the paper, selectable via config:

| Strategy | Config value | Description |
|----------|-------------|-------------|
| **FedPG-BR** | `method = "fedpg-br"` | Byzantine filtering + SCSG variance reduction (paper's main contribution) |
| **SVRPG** | `method = "svrpg"` | SCSG variance reduction only, no Byzantine protection |
| **GOMDP** | `method = "gomdp"` | Simple gradient averaging, single step (baseline) |

### Implement your own strategy

Create a file in `fedpg_br/strategies/`:

```python
from fedpg_br.strategies import AggregationStrategy, register_strategy

@register_strategy("coordinate-median")
class CoordinateMedian(AggregationStrategy):
    description = "Coordinate-wise median aggregation"

    def aggregate(self, gradients, batch_size, **kwargs):
        stacked = torch.stack(gradients)
        return torch.median(stacked, dim=0).values, list(range(len(gradients)))

    def server_update(self, policy, optimizer, theta_t_0, mu_t, config, **kwargs):
        from fedpg_br.strategies.base import apply_gradient
        apply_gradient(policy, optimizer, mu_t)
        return 1
```

Then use it: `method = "coordinate-median"` in your config TOML.

## Configuration

All experiments are configured via TOML files in `configs/`:

```toml
env = "CartPole-v1"           # Environment
num-server-rounds = 312       # Training rounds
num-workers = 10              # K agents (paper: K=10)
num-byzantine = 0             # B Byzantine agents (paper: B=3 for attack scenarios)
method = "fedpg-br"           # Strategy: fedpg-br, svrpg, gomdp
use-fedpg-br = true           # Enable adaptive batch sizing
attack-type = "random-noise"  # Attack: random-noise, sign-flip, random-action, fedpg-attack
use-adaptive-communication = false
```

### Paper configurations (included)

| Config | Env | Rounds | Expected reward | Runtime |
|--------|-----|--------|----------------|---------|
| `paper_cartpole_final.toml` | CartPole-v1 | 312 | ~500 | ~2.5h |
| `paper_lunarlander_final.toml` | LunarLander-v3 | 323 | ~200-250 | ~12h |
| `paper_halfcheetah_final.toml` | HalfCheetah-v5 | 208 | ~3000+ | ~4h |

Round counts are derived from the paper's trajectory budget using:
`rounds = max_trajectories / (K * batch_size + mini_batch_size) * (1 + world_size)`

## Environments

| Environment | Dependencies | Action space |
|-------------|-------------|--------------|
| CartPole-v1 | None | Discrete |
| MountainCar-v0 | None | Discrete |
| Acrobot-v1 | None | Discrete |
| LunarLander-v3 | `gymnasium[box2d]` | Discrete |
| HalfCheetah-v5 | `gymnasium[mujoco]` | Continuous |

## Byzantine Attacks

7 attack types for testing robustness:

| Attack | Config value | Description |
|--------|-------------|-------------|
| Random Noise | `random-noise` | Sends random gradient vectors |
| Sign Flip | `sign-flip` | Sends -2.5x the true gradient |
| Random Action | `random-action` | Takes random actions (hardware failure) |
| FedPG Attack | `fedpg-attack` | Sophisticated attack to evade the Byzantine filter |
| Variance Attack | `variance-attack` | Exploits gradient variance |
| Zero Gradient | `zero-gradient` | Sends zero vectors |
| Reward Flipping | `reward-flipping` | Negates rewards during training |

Example with 3 Byzantine agents using sign-flip:
```toml
num-workers = 10
num-byzantine = 3
attack-type = "sign-flip"
method = "fedpg-br"
```

## Web Dashboard

Monitor experiments in real-time:

```bash
pip install -e ".[dashboard]"
python -m fedpg_br.dashboard.app
# Open http://localhost:5000/experiment
```

Features:
- Configure experiments (environment, strategy, workers, attacks)
- Real-time learning curve
- Worker status visualization (good vs Byzantine)
- Strategy comparison overlay

## Docker Deployment

Run as distributed containers (server + workers on separate machines):

```bash
# Default: CartPole, FedPG-BR, 10 workers
docker-compose up --build

# Change environment and strategy
ENV=LunarLander-v3 METHOD=svrpg docker-compose up

# Byzantine attack scenario
BYZANTINE_RATIO=0.3 docker-compose up

# Dashboard at http://localhost:5000/experiment
```

### Distributed (multiple machines)

Server machine:
```bash
docker-compose -f docker-compose-server.yml up
```

Client machines:
```bash
docker-compose -f docker-compose-client.yml up
```

## Adaptive Communication

Reduce bandwidth by 30-65% -- clients skip rounds when their policy hasn't changed much:

```toml
use-adaptive-communication = true
divergence-threshold = 0.1
divergence-metric = "l2"   # l2, cosine, or max
```

## Project Structure

```
fedpg_br/
  config.py              # Environment hyperparameters (paper Table 1)
  server_app.py          # Flower server with strategy dispatch
  client_app.py          # Flower client (worker gradient computation)
  policy.py              # Neural network policies
  strategies/            # Plugin strategy system
    base.py              #   Abstract base + registry
    gomdp.py             #   Simple averaging (baseline)
    svrpg.py             #   SCSG variance reduction
    fedpg_br_strategy.py #   Full FedPG-BR (paper algorithm)
  core/
    byzantine.py         # Byzantine filtering (Algorithm 1.1)
    trajectory.py        # Trajectory sampling
    gradient.py          # Policy gradient computation
  dashboard/             # Web UI
  benchmark/             # Benchmark framework
  deploy_server.py       # Docker/distributed server entry
  deploy_client.py       # Docker/distributed client entry
configs/                 # Experiment TOML configs
plot_results.py          # Paper-style plotting (RBF + 90% CI)
```

## References

- Fan et al., "Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee", NeurIPS 2021
- [Original implementation](https://github.com/flint-xf-fan/Byzantine-Federated-RL)
- [Flower framework](https://flower.ai/)
