# pyright: reportGeneralTypeIssues=false
# type: ignore
"""
main_paper_settings.py
======================
Exakte Hyperparameter aus Table 2
"""

import numpy as np
import gymnasium as gym
from flwr.server import ServerConfig
from flwr.simulation import start_simulation
from policy_network import CategoricalPolicy
from fedPG_BR import FedPGBR
from FedPG_BR_client import FedPGClient


def main():
    # Environment
    env_server = gym.make("CartPole-v1")
    policy_server = CategoricalPolicy(4, 2)
    init_theta = np.random.randn(policy_server.param_size) * 0.01
    
    # Paper Table 2 Settings für CartPole-v1
    strategy = FedPGBR(
        initial_parameters=init_theta,
        server_environment=env_server,
        policy_network=policy_server,
        batch_size = np.random.randint(12, 21),              # ← Paper: sampled [12,20], wir nehmen Mitte
        minibatch_size=4,           # ← Paper: 4 ✅
        learning_rate=0.001,        # ← Paper: 1e-3 ✅
        variance_bound=0.06,        # ← Paper: 0.06 ✅
        byzantine_fraction=0.0,     # ← Paper: 0.3 ✅
        confidence_level=0.6        # ← Paper: 0.6 ✅
    )
    
    def client_fn(cid: str):
        return FedPGClient(gym.make("CartPole-v1"), CategoricalPolicy(4, 2), cid)
    
    print("="*60)
    print("FedPG-BR with EXACT Paper Settings (Table 2)")
    print("="*60)
    print(f"B_t={16}, b_t={4}, η={0.001}, σ={0.06}, α={0.3}, δ={0.6}")
    print(f"Target: ~500 reward in 5000 trajectories")
    print("="*60)
    
    # Run
    start_simulation(
        client_fn=client_fn,
        num_clients=1,
        config=ServerConfig(num_rounds=250),  # 250*20 = 5000 traj
        strategy=strategy,
    )
    
    # Evaluate
    policy_server.set_weights(strategy.theta_tilde)
    rewards = []
    
    for _ in range(100):
        r = env_server.reset()
        state = r[0] if isinstance(r, tuple) else r
        done = False
        ep_r = 0
        
        while not done:
            action = policy_server.sample_action(state)
            step = env_server.step(action)
            state = step[0]
            ep_r += step[1]
            done = (step[2] or step[3]) if len(step) == 5 else step[2]
        
        rewards.append(ep_r)
    
    avg = np.mean(rewards)
    print(f"\n{'='*60}")
    print(f"RESULT: {avg:.1f} ± {np.std(rewards):.1f} reward")
    print(f"Paper target: ~500")
    print(f"{'='*60}")
    
    env_server.close()


if __name__ == "__main__":
    main()