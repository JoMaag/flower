# pyright: reportGeneralTypeIssues=false
# type: ignore
"""
FedPG-BR Main Script
====================
Runs the complete federated learning simulation.

Setup:
1. Create server with FedPGBR strategy
2. Create client factory
3. Run simulation
4. Evaluate final policy
"""

import numpy as np
import gymnasium as gym
from flwr.server import ServerConfig
from flwr.simulation import start_simulation

# Import our implementations
from fedPG_BR import FedPGBR
from FedPG_BR_client import FedPGClient  # Your existing client
from policy_network import CategoricalPolicy


def main():
    """Run FedPG-BR federated learning simulation."""
    
    print("=" * 70)
    print(" FedPG-BR: Byzantine-Resilient Federated Policy Gradient")
    print("=" * 70)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Environment
    env_name = "CartPole-v1"
    state_dim = 4
    action_dim = 2
    
    # Federated Learning
    num_clients = 2
    num_rounds = 100
    
    # Algorithm hyperparameters (from paper Table 2)
    batch_size = 16        # B_t - trajectories per client
    minibatch_size = 4     # b_t - trajectories for server inner loop
    learning_rate = 0.01   # η_t - step size
    variance_bound = 1.0   # σ - gradient variance bound
    byzantine_fraction = 0.3  # α - fraction of Byzantine clients tolerated
    
    print(f"\nConfiguration:")
    print(f"  Environment: {env_name}")
    print(f"  Clients: {num_clients}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Batch size (B_t): {batch_size}")
    print(f"  Learning rate (η_t): {learning_rate}")
    print(f"  Byzantine tolerance (α): {byzantine_fraction}")
    print("=" * 70)
    
    # =========================================================================
    # SETUP SERVER
    # =========================================================================
    
    print("\n[Setup] Creating server...")
    
    # Server environment
    env_server = gym.make(env_name)
    
    # Server policy
    policy_server = CategoricalPolicy(state_dim, action_dim)
    
    # Initialize policy parameters (small random values)
    init_theta = np.random.randn(policy_server.param_size) * 0.01
    print(f"  Initial policy parameters: {policy_server.param_size}")
    
    # Create server strategy
    strategy = FedPGBR(
        initial_parameters=init_theta,
        server_environment=env_server,
        policy_network=policy_server,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        learning_rate=learning_rate,
        variance_bound=variance_bound,
        byzantine_fraction=byzantine_fraction,
        confidence_level=0.1
    )
    
    # =========================================================================
    # SETUP CLIENTS
    # =========================================================================
    
    print("\n[Setup] Creating client factory...")
    
    def client_fn(cid: str) -> FedPGClient:
        """
        Create a client with its own environment and policy.
        
        Each client has:
        - Independent environment (own MDP)
        - Own policy network instance
        - Unique client ID
        """
        env_client = gym.make(env_name)
        policy_client = CategoricalPolicy(state_dim, action_dim)
        return FedPGClient(env_client, policy_client, cid)
    
    print(f"  Client factory ready for {num_clients} clients")
    
    # =========================================================================
    # RUN SIMULATION
    # =========================================================================
    
    print("\n" + "=" * 70)
    print(" STARTING FEDERATED LEARNING SIMULATION")
    print("=" * 70)
    
    history = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    
    # Print final metrics if available
    if hasattr(history, "metrics_centralized") and history.metrics_centralized:
        print("\nFinal Server Metrics:")
        try:
            metrics = history.metrics_centralized  # type: ignore
            for key, values in metrics.items():  # type: ignore
                if values:
                    final_value = values[-1][1]  # type: ignore
                    print(f"  {key}: {final_value}")
        except Exception as e:
            print(f"  Could not display metrics: {e}")
    
    # =========================================================================
    # EVALUATE FINAL POLICY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print(" EVALUATING FINAL POLICY")
    print("=" * 70)
    
    # Set policy to final learned parameters
    policy_server.set_weights(strategy.theta_tilde)
    
    # Run evaluation episodes
    num_eval_episodes = 20
    eval_rewards = []
    
    print(f"\nRunning {num_eval_episodes} evaluation episodes...")
    
    for ep in range(num_eval_episodes):
        # Reset environment
        reset_result = env_server.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        done = False
        episode_reward = 0
        steps = 0
        max_steps = 1000
        
        # Run episode
        while not done and steps < max_steps:
            action = policy_server.sample_action(state)
            step_result = env_server.step(action)
            
            # Handle both Gym APIs
            if len(step_result) == 5:
                state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                state, reward, done, _ = step_result
            
            episode_reward += reward
            steps += 1
        
        eval_rewards.append(episode_reward)
        
        if (ep + 1) % 5 == 0:
            print(f"  Episodes {ep-3}-{ep+1}: "
                  f"{np.mean(eval_rewards[-5:]):.1f} avg reward")
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    
    print("\n" + "=" * 70)
    print(" FINAL RESULTS")
    print("=" * 70)
    
    avg_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    min_reward = np.min(eval_rewards)
    max_reward = np.max(eval_rewards)
    
    print(f"\nEvaluation over {num_eval_episodes} episodes:")
    print(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Min reward: {min_reward:.0f}")
    print(f"  Max reward: {max_reward:.0f}")
    
    # CartPole is "solved" at 195+ average reward
    if avg_reward >= 195:
        print(f"\n🎉 SUCCESS! CartPole solved (≥195 reward)")
    elif avg_reward >= 100:
        print(f"\n📈 Good progress! Getting close to solving CartPole")
    else:
        print(f"\n⚠️  More training needed. Try:")
        print(f"     - Increase num_rounds to 200+")
        print(f"     - Increase learning_rate to 0.03")
        print(f"     - Increase batch_size to 32")
    
    print("\n" + "=" * 70)
    print(" DONE")
    print("=" * 70)
    
    # Clean up
    env_server.close()


# =============================================================================
# OPTIONAL: Test with Byzantine Clients
# =============================================================================

def main_with_byzantine():
    """
    Run simulation with Byzantine (malicious) clients.
    
    This tests if the Byzantine filter works correctly.
    """
    print("=" * 70)
    print(" FedPG-BR with Byzantine Clients (Attack Simulation)")
    print("=" * 70)
    
    # Same config as main()
    env_name = "CartPole-v1"
    state_dim = 4
    action_dim = 2
    num_clients = 10
    num_rounds = 100
    
    # Setup server (same as main)
    env_server = gym.make(env_name)
    policy_server = CategoricalPolicy(state_dim, action_dim)
    init_theta = np.random.randn(policy_server.param_size) * 0.01
    
    strategy = FedPGBR(
        initial_parameters=init_theta,
        server_environment=env_server,
        policy_network=policy_server,
        batch_size=16,
        minibatch_size=4,
        learning_rate=0.01,
        variance_bound=1.0,
        byzantine_fraction=0.3,
        confidence_level=0.1
    )
    
    # Client factory with Byzantine attackers
    def client_fn(cid: str) -> FedPGClient:
        env_client = gym.make(env_name)
        policy_client = CategoricalPolicy(state_dim, action_dim)
        client = FedPGClient(env_client, policy_client, cid)
        
        # Make first 3 clients Byzantine (random noise attack)
        if int(cid) < 3:
            original_fit = client.fit
            
            def byzantine_fit(params, config):
                # Call original fit to get structure
                result = original_fit(params, config)
                
                # Replace gradient with random noise
                gradient_shape = result[0][0].shape
                result[0][0] = np.random.randn(*gradient_shape) * 10
                
                print(f"[Client {cid}] 💀 BYZANTINE ATTACK: Sending random noise!")
                
                return result
            
            client.fit = byzantine_fit
        
        return client
    
    print(f"\n⚠️  Warning: {3} out of {num_clients} clients are Byzantine!")
    print("The Byzantine filter should detect and exclude them.\n")
    
    # Run simulation
    history = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    print("\n✅ Simulation complete with Byzantine clients!")
    print("Check the logs to see if Byzantine gradients were filtered.\n")
    
    env_server.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run normal simulation
    main()
    
    # Uncomment to test Byzantine robustness
    # print("\n\n")
    # main_with_byzantine()
