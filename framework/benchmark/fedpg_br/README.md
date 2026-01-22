# FedPG-BR Configuration Files

This directory contains pre-configured experiment setups.

## Usage

```bash
# Run a specific config
flwr run . --run-config configs/cartpole_baseline.toml

# Or use the config from pyproject.toml
flwr run .
```

## Available Configurations

### Baseline Experiments
- `cartpole_baseline.toml` - 10 workers, no Byzantine agents
- `cartpole_single_worker.toml` - 1 worker (for comparison)

### Byzantine Attack Experiments (3 out of 10 workers are Byzantine)
- `cartpole_random_noise.toml` - Random Noise Attack
- `cartpole_sign_flipping.toml` - Sign Flipping Attack  
- `cartpole_random_action.toml` - Random Action Attack

## Creating Your Own Config

Create a new `.toml` file with these parameters:

```toml
env = "CartPole-v1"  # or "LunarLander-v3", "HalfCheetah-v5"
num-server-rounds = 150
num-workers = 10
num-byzantine = 3
attack-type = "random-noise"  # or "sign-flipping", "random-action", etc.
```

## Supported Environments

- **CartPole-v1** (works without extra dependencies)
- **LunarLander-v3** (requires: `pip install gymnasium-box2d`)
- **HalfCheetah-v5** (requires: `pip install gymnasium[mujoco]`)

## Attack Types

- `zero-gradient` - Send zero gradients
- `random-action` - Take random actions (simulates hardware failure)
- `sign-flipping` - Flip the sign of gradients
- `reward-flipping` - Flip reward signs
- `random-reward` - Use random rewards
- `random-noise` - Add random noise to gradients
