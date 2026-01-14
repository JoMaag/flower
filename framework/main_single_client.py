# pyright: reportGeneralTypeIssues=false
# type: ignore
"""
main_single_client.py
=====================
Test with K=1 (single client) to match paper.
Should reach ~500 reward in ~5000 trajectories.
"""
import logging
logging.getLogger("flwr").setLevel(logging.WARNING)  # Nur Warnings
import numpy as np
import gymnasium as gym
from flwr.server import ServerConfig
from flwr.simulation import start_simulation
from policy_network import CategoricalPolicy
from fedPG_BR import FedPGBR
from FedPG_BR_client import FedPGClient

def main():
    # Paper setup: 5000 trajectories
    num_clients = 1
    batch_size = 20
    num_rounds = 250  # 250 × 20 = 5000 trajectories
    
    print(f"K={num_clients}, Total trajectories={num_rounds * batch_size}")
    
    # Setup
    env_server = gym.make("CartPole-v1")
    policy_server = CategoricalPolicy(4, 2)
    init_theta = np.random.randn(policy_server.param_size) * 0.01
    
    strategy = FedPGBR(
        initial_parameters=init_theta,      # ← Richtig
        server_environment=env_server,      # ← Richtig
        policy_network=policy_server,       # ← Richtig
        batch_size=batch_size,
        minibatch_size=4,
        learning_rate=0.2,                 # ← Höher für K=1
        variance_bound=1.0,
        byzantine_fraction=0.0,             # ← α=0 (kein Byzantine)
        confidence_level=0.1
    )
    
    def client_fn(cid: str):
        return FedPGClient(
            gym.make("CartPole-v1"),
            CategoricalPolicy(4, 2),
            cid
        )
    
    # Train
    print("Training...")
    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    # Evaluate
    policy_server.set_weights(strategy.theta_tilde)
    rewards = []
    
    for _ in range(100):
        r = env_server.reset()
        state = r[0] if isinstance(r, tuple) else r
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < 1000:
            action = policy_server.sample_action(state)
            step = env_server.step(action)
            
            if len(step) == 5:
                state, reward, term, trunc, _ = step
                done = term or trunc
            else:
                state, reward, done, _ = step
            
            ep_reward += reward
            steps += 1
        
        rewards.append(ep_reward)
    
    avg = np.mean(rewards)
    print(f"\n{'='*60}")
    print(f"Trajectories: {num_rounds * batch_size}")
    print(f"Average reward: {avg:.1f} ± {np.std(rewards):.1f}")
    print(f"Target (paper): ~500")
    print(f"Solved threshold: 195")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()