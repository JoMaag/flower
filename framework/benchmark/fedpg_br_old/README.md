# FedPG-BR: Fault-Tolerant Federated Reinforcement Learning

Implementation of **FedPG-BR** (Federated Policy Gradient with Byzantine Resilience) based on the NeurIPS 2021 paper by Flint Xiaofeng Fan et al.

## Installation

```bash
pip install -e .
```

## Usage

### Basic Run (CartPole, 10 workers, 50 rounds)

```bash
flwr run .
```

### Custom Configuration

```bash
# Different environment
flwr run . --run-config "env='LunarLander-v2'"

# With Byzantine agents
flwr run . --run-config "num-byzantine=3 attack-type='sign-flipping'"

# FedPG-BR with adaptive batch size
flwr run . --run-config "use-fedpg-br=true"

# More rounds
flwr run . --run-config "num-server-rounds=100"

# Combine options
flwr run . --run-config "env='CartPole-v1' num-workers=5 num-byzantine=1 num-server-rounds=30"
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `env` | `CartPole-v1` | Environment (`CartPole-v1`, `LunarLander-v2`, `HalfCheetah-v2`) |
| `num-server-rounds` | `50` | Number of training rounds |
| `num-workers` | `10` | Number of federated workers (K) |
| `num-byzantine` | `0` | Number of Byzantine agents |
| `attack-type` | `random-noise` | Attack type (see below) |
| `use-fedpg-br` | `false` | Use adaptive batch size (FedPG-BR) |

### Attack Types

- `zero-gradient` - Send zero gradient
- `random-action` - Take random actions (hardware failure)
- `sign-flipping` - Send negative gradient (× -2.5)
- `reward-flipping` - Flip reward signs
- `random-reward` - Shuffle rewards randomly
- `random-noise` - Add random noise to gradient

## Algorithm

FedPG-BR implements:
1. **Algorithm 1**: Federated Policy Gradient with SCSG inner loop
2. **Algorithm 1.1**: Byzantine-resilient gradient aggregation with two filtering rules (R1, R2)

## Paper

```bibtex
@inproceedings{fan2021fault,
  title={Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee},
  author={Fan, Flint Xiaofeng and Ma, Yining and Dai, Zhongxiang and others},
  booktitle={NeurIPS},
  year={2021}
}
```
