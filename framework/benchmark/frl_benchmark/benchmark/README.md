# Flower FRL Benchmark

> *"Those Chinese New Year celebration robots that dance in perfect sync — hundreds of them, each learning from its own sensors, sharing only what it discovered, none needing to hand its body over to a central controller."*
>
> That is the mental model for this project.

Each agent explores its own copy of the environment. After every round it shares **what it learned** (a gradient), not its raw experience. A coordinator aggregates those signals, updates the shared policy, and sends it back. Agents with corrupted or adversarial updates are detected and discarded by the Byzantine filter — the one damaged robot does not bring down the rest.

The result is a policy that emerges from collective experience, more sample-efficient than any single agent, and robust to a fraction of bad actors.

This is a [Flower](https://flower.ai/) federated reinforcement learning testbench, built around **FedPG-BR** (Fan et al., NeurIPS 2021). Extended with a plugin strategy system, a live web dashboard, and one-command Docker deployment.

---

## How It Works

```
Round t:
                        ┌─────────────────────────────────┐
                        │         Parameter Server         │
                        │  θ  ──► Byzantine Filter ──►  θ' │
                        └──────┬────────────────▲──────────┘
                               │ broadcast θ    │ gradients
              ┌────────────────┼────────────────┼────────────────┐
              ▼                ▼                ▼                ▼
         [Agent 0]        [Agent 1]        [Agent 2]  ...  [Agent K]
         rolls out        rolls out        rolls out        rolls out
         own episodes     own episodes     own episodes     own episodes
         ∇J(θ) ──►       ∇J(θ) ──►       ✗ Byzantine      ∇J(θ) ──►
                                          (filtered out)
```

**Three aggregation methods** (all interchangeable, selectable at runtime):

| Method | Filtering | Variance reduction | Use when |
|--------|-----------|-------------------|----------|
| **FedPG-BR** | Byzantine filter | SCSG (paper algorithm) | Untrusted agents present |
| **SVRPG** | None | SCSG | All agents trusted, noisy gradients |
| **GOMDP** | None | None | Baseline / ablation |

---

## Quick Start

### Docker (recommended — no local dependencies)

```bash
git clone <this-repo>
cd frl_benchmark

docker compose up --build
# → Dashboard at http://localhost:8050/experiment
```

The dashboard starts. Click **Start** to launch a training run with your chosen environment, strategy, number of workers, and Byzantine ratio.

### Local

```bash
pip install -e ".[dashboard]"
frl-dashboard
# → http://localhost:8050/experiment
```

---

## Strategies

Three strategies from the paper, selectable from the dashboard or config:

```toml
method = "fedpg-br"   # Byzantine filtering + SCSG variance reduction
method = "svrpg"      # SCSG only
method = "gomdp"      # Simple averaging (baseline)
```

### Implement your own

Drop a file into `frl_benchmark/strategies/` and it is automatically picked up:

```python
from frl_benchmark.strategies import AggregationStrategy, register_strategy

@register_strategy("coordinate-median")
class CoordinateMedian(AggregationStrategy):
    description = "Coordinate-wise median aggregation"

    def aggregate(self, gradients, batch_size, **kwargs):
        stacked = torch.stack(gradients)
        return torch.median(stacked, dim=0).values, list(range(len(gradients)))

    def server_update(self, policy, optimizer, theta_t_0, mu_t, config, **kwargs):
        from frl_benchmark.strategies.base import apply_gradient
        apply_gradient(policy, optimizer, mu_t)
        return 1
```

Select it with `method = "coordinate-median"` in your TOML or from the dashboard dropdown.

---

## Configuration

Experiments are configured via TOML files in `configs/`:

```toml
env = "CartPole-v1"           # Gymnasium environment
num-server-rounds = 312       # Training rounds
num-workers = 10              # K agents (paper: K=10)
num-byzantine = 3             # B Byzantine agents (must be < K/2 for guarantees)
method = "fedpg-br"           # Strategy
attack-type = "sign-flip"     # Attack type for Byzantine agents
```

### Included paper configs

| Config | Environment | Rounds | Target reward | Notes |
|--------|-------------|--------|---------------|-------|
| `paper_cartpole.toml` | CartPole-v1 | 312 | ~500 | ~2.5 h on CPU |
| `paper_lunarlander.toml` | LunarLander-v3 | 323 | ~200–250 | ~12 h |
| `paper_halfcheetah.toml` | HalfCheetah-v5 | 208 | ~3 000+ | ~4 h, needs MuJoCo |

Round counts follow the paper's trajectory budget:
`rounds = max_trajectories / (K × batch_size + mini_batch_size) × (1 + world_size)`

---

## Byzantine Attacks

Seven attack types for robustness evaluation:

| Attack | Key | What the agent sends |
|--------|-----|----------------------|
| Random Noise | `random-noise` | Gaussian random gradient |
| Sign Flip | `sign-flip` | −2.5 × true gradient |
| Random Action | `random-action` | Gradient from uniformly random policy |
| FedPG Attack | `fedpg-attack` | Optimised to evade Byzantine filter |
| Variance Attack | `variance-attack` | Exploits gradient variance estimates |
| Zero Gradient | `zero-gradient` | All zeros (free-rider) |
| Reward Flipping | `reward-flipping` | Negates rewards during rollout |

---

## Environments

| Environment | Extra dependency | Action space |
|-------------|-----------------|--------------|
| CartPole-v1 | — | Discrete |
| MountainCar-v0 | — | Discrete |
| Acrobot-v1 | — | Discrete |
| LunarLander-v3 | `pip install gymnasium[box2d]` | Discrete |
| HalfCheetah-v5 | `pip install gymnasium[mujoco]` | Continuous |

---

## Project Structure

```
frl_benchmark/
  server_app.py          # Flower ServerApp — strategy dispatch, Byzantine filter
  client_app.py          # Flower ClientApp — rollout, gradient computation
  policy.py              # MLP policies (discrete + continuous action spaces)
  config.py              # Per-environment hyperparameters (paper Table 1)
  strategies/            # Plugin aggregation strategies
    base.py              #   Abstract base + decorator registry
    gomdp.py             #   Simple averaging (baseline)
    svrpg.py             #   SCSG variance reduction
    fedpg_br_strategy.py #   Full FedPG-BR (paper algorithm)
    my_strategy.py       #   Empty template — start here for a custom strategy
    example_strategy.py  #   Worked example: trimmed-mean aggregation
  core/
    byzantine.py         # Byzantine filter (Algorithm 1.1)
    trajectory.py        # Episode sampling
    gradient.py          # Policy gradient + SCSG correction
  flower/
    worker.py            # Gym worker — env, policy, attack logic
  dashboard/             # Flask + SocketIO web UI
configs/                 # TOML experiment configs
plot_results.py          # Paper-style plots — RBF smoothing + 90% CI
```

---

## Installation Details

```bash
# Core (training only)
pip install -e .

# With dashboard
pip install -e ".[dashboard]"

# MuJoCo environments
pip install gymnasium[mujoco]

# Box2D environments
pip install gymnasium[box2d]
```

---

## Reference

Fan, X., Ma, Y., Dai, Z., Jing, W., Tan, C., & Low, B.K.H. (2021).
**Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee.**
*Advances in Neural Information Processing Systems (NeurIPS).*

- [Paper](https://arxiv.org/abs/2110.11164)
- [Original implementation](https://github.com/flint-xf-fan/Byzantine-Federated-RL)
- [Flower framework](https://flower.ai/)