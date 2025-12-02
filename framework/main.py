import numpy as np
import gymnasium as gym
from flwr.server import ServerConfig
from flwr.simulation import start_simulation

from fedpg_demo.server_strategy import fedPG_BR
from fedpg_demo.client import FedPGClient
from policy_network import CategoricalPolicy

def main():
    print("=" * 60)
    print("FedPG-BR Demo - CartPole")
    print("=" * 60)
    
    # Environment
    env_name = "CartPole-v1"
    state_dim = 4
    action_dim = 2
    
    env_server = gym.make(env_name)
    
    print(f"Environment: {env_name}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Policy
    policy_server = CategoricalPolicy(state_dim, action_dim)
    print(f"Policy params: {policy_server.param_size}")
    
    # Initialize
    init_theta = np.random.randn(policy_server.param_size) * 0.01
    
    # Server Strategy
    strategy = fedPG_BR(
        init_params=init_theta,
        env=env_server,
        policy_network=policy_server,
        batch_size=16,
        minibatch_size=4,
        step_size=0.01,
        sigma=1.0,
        alpha=0.3,
        delta=0.1
    )
    
    print("\nServer Configuration:")
    print(f"  Batch size (B_t): {strategy.Bt}")
    print(f"  Mini-batch size (b_t): {strategy.bt}")
    print(f"  Step size (η_t): {strategy.eta_t}")
    print(f"  Byzantine tolerance (α): {strategy.alpha}")
    
    # Client Factory
    def client_fn(cid: str):
        env_client = gym.make(env_name)
        policy_client = CategoricalPolicy(state_dim, action_dim)
        return FedPGClient(env_client, policy_client, cid)
    
    # Simulation
    num_clients = 10
    num_rounds = 50
    
    print(f"\nSimulation Configuration:")
    print(f"  Number of clients: {num_clients}")
    print(f"  Number of rounds: {num_rounds}")
    print("=" * 60)
    print()
    
    # Run
    history = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Print metrics
    try:
        if hasattr(history, "metrics_centralized"):
            print("\nFinal Metrics:")
            metrics = history.metrics_centralized  # type: ignore
            for key, values in metrics.items():  # type: ignore
                if values:
                    final_value = values[-1][1]  # type: ignore
                    print(f"  {key}: {final_value}")
    except Exception:
        pass
    
    # Evaluate
    print("\nEvaluating final policy...")
    policy_server.set_weights(strategy.theta_tilde)
    
    eval_episodes = 10
    total_reward = 0
    
    for ep in range(eval_episodes):
        reset_result = env_server.reset()  # type: ignore
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result  # type: ignore
        
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 1000:
            action = policy_server.sample_action(state)  # type: ignore
            step_result = env_server.step(action)  # type: ignore
            
            if len(step_result) == 5:  # type: ignore
                state, reward, terminated, truncated, _ = step_result  # type: ignore
                done = terminated or truncated
            else:
                state, reward, done, _ = step_result  # type: ignore
            
            episode_reward += reward  # type: ignore
            steps += 1
        
        total_reward += episode_reward
        print(f"  Episode {ep+1}: {episode_reward:.0f} reward")
    
    avg_reward = total_reward / eval_episodes
    print(f"\nAverage reward over {eval_episodes} episodes: {avg_reward:.2f}")
    
    env_server.close()
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()