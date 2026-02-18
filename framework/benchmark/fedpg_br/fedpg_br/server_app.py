"""Flower ServerApp for FedPG-BR."""

import numpy as np
import torch
import gymnasium as gym
import fedpg_br.envs  # noqa: F401 - registers custom environments
from typing import Dict, List, Optional, Tuple, Union
from logging import INFO

from flwr.common import (
    FitRes, Parameters, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters, log,
)
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import Context
import flwr as fl

from fedpg_br.config import get_config, get_env_info
from fedpg_br.policy import create_policy
from fedpg_br.core.byzantine import ByzantineFilter
from fedpg_br.core.trajectory import sample_trajectory, compute_returns
from fedpg_br.core.gradient import compute_policy_gradient, compute_log_probs


class FedPGStrategy(Strategy):
    """FedPG-BR Strategy implementing Algorithm 1.

    Supports three aggregation methods from the paper:
      - 'fedpg-br': Full FedPG-BR (Byzantine filtering + SCSG)
      - 'svrpg': SCSG variance reduction only (no Byzantine filtering)
      - 'gomdp': Simple averaging (no SCSG, no filtering)
    """

    def __init__(self, env_name: str, num_agents: int, byzantine_ratio: float = 0.0,
                 use_adaptive_batch: bool = False, method: str = 'fedpg-br', config=None):
        super().__init__()

        self.config = config if config is not None else get_config(env_name)
        self.env_name = env_name
        self.method = method
        env_info = get_env_info(env_name)
        self.state_dim = env_info["state_dim"]
        self.action_dim = env_info["action_dim"]
        self.num_agents = num_agents
        self.use_adaptive_batch = use_adaptive_batch

        self.policy = create_policy(
            self.state_dim, self.action_dim, env_name,
            self.config.hidden_units, self.config.activation, self.config.output_activation
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr)

        self.byzantine_filter = ByzantineFilter(
            self.config.sigma, self.config.delta, num_agents, byzantine_ratio
        )

        self.env = gym.make(env_name)
        self.theta_t_0: torch.Tensor = self.policy.get_flat_params().clone()
        self._current_batch_size = self.config.batch_size

        log(INFO, f"Method: {self.method.upper()}")
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return ndarrays_to_parameters([p.cpu().detach().numpy() for p in self.policy.parameters()])
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        self.theta_t_0 = self.policy.get_flat_params().clone()
        
        if self.use_adaptive_batch:
            self._current_batch_size = np.random.randint(
                self.config.batch_size_min, self.config.batch_size_max + 1
            )
        else:
            self._current_batch_size = self.config.batch_size
        
        clients = client_manager.sample(num_clients=self.num_agents, min_num_clients=self.num_agents)
        fit_config: Dict[str, Scalar] = {"batch_size": int(self._current_batch_size), "round": server_round}
        fit_ins = fl.common.FitIns(parameters, fit_config)
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
                      failures) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Filter out clients that skipped this round (num_examples == 0)
        active_results = []
        skipped_count = 0
        total_divergence = 0.0

        for client_proxy, fit_res in results:
            if fit_res.num_examples > 0:
                active_results.append((client_proxy, fit_res))
            else:
                skipped_count += 1
                # Track divergence metrics from skipped clients
                if "divergence" in fit_res.metrics:
                    total_divergence += float(fit_res.metrics["divergence"])

        # If all clients skipped, return current parameters unchanged
        if not active_results:
            log(INFO, f"Round {server_round}: All clients skipped (divergence too small)")
            return ndarrays_to_parameters([p.cpu().detach().numpy() for p in self.policy.parameters()]), {
                "num_good_agents": 0,
                "scsg_steps": 0,
                "batch_size": self._current_batch_size,
                "skipped_clients": skipped_count,
                "active_clients": 0,
                "avg_divergence": total_divergence / len(results) if results else 0.0,
            }

        gradients = []
        for _, fit_res in active_results:
            grad_numpy = parameters_to_ndarrays(fit_res.parameters)
            grad_tensor = torch.cat([torch.from_numpy(g).flatten() for g in grad_numpy]).float()
            gradients.append(grad_tensor)

        if self.method == 'gomdp':
            # GOMDP: Simple averaging, single gradient step, no filtering
            mu_t = torch.mean(torch.stack(gradients), dim=0)
            good_agents = list(range(len(gradients)))

            self.policy.set_flat_params(self.theta_t_0)
            self.optimizer.zero_grad()
            offset = 0
            for param in self.policy.parameters():
                size = param.numel()
                param.grad = mu_t[offset:offset + size].view(param.shape).clone()
                offset += size
            self.optimizer.step()
            actual_steps = 1

        elif self.method == 'svrpg':
            # SVRPG: No Byzantine filtering, but uses SCSG variance reduction
            mu_t = torch.mean(torch.stack(gradients), dim=0)
            good_agents = list(range(len(gradients)))

            p_geom = self.config.mini_batch_size / (self._current_batch_size + self.config.mini_batch_size)
            N_t = np.random.geometric(p_geom)

            theta_t_n = self.theta_t_0.clone()
            actual_steps = 0

            for n in range(N_t):
                v_t_n, ratio = self._scsg_step(theta_t_n, self.theta_t_0, mu_t)
                if ratio < 0.995 or ratio > 1.005:
                    break
                self.policy.set_flat_params(theta_t_n)
                self.optimizer.zero_grad()
                offset = 0
                for param in self.policy.parameters():
                    size = param.numel()
                    param.grad = v_t_n[offset:offset + size].view(param.shape).clone()
                    offset += size
                self.optimizer.step()
                theta_t_n = self.policy.get_flat_params()
                actual_steps = n + 1

            self.policy.set_flat_params(theta_t_n)

        else:
            # FedPG-BR: Byzantine filtering + SCSG
            mu_t, good_agents = self.byzantine_filter.aggregate(gradients, self._current_batch_size)

            p_geom = self.config.mini_batch_size / (self._current_batch_size + self.config.mini_batch_size)
            N_t = np.random.geometric(p_geom)

            theta_t_n = self.theta_t_0.clone()
            actual_steps = 0

            for n in range(N_t):
                v_t_n, ratio = self._scsg_step(theta_t_n, self.theta_t_0, mu_t)
                if ratio < 0.995 or ratio > 1.005:
                    break
                self.policy.set_flat_params(theta_t_n)
                self.optimizer.zero_grad()
                offset = 0
                for param in self.policy.parameters():
                    size = param.numel()
                    param.grad = v_t_n[offset:offset + size].view(param.shape).clone()
                    offset += size
                self.optimizer.step()
                theta_t_n = self.policy.get_flat_params()
                actual_steps = n + 1

            self.policy.set_flat_params(theta_t_n)

        log(INFO, f"Round {server_round} [{self.method.upper()}]: good_agents={len(good_agents)}, "
                  f"scsg_steps={actual_steps}, active={len(active_results)}, skipped={skipped_count}")

        return ndarrays_to_parameters([p.cpu().detach().numpy() for p in self.policy.parameters()]), {
            "num_good_agents": len(good_agents),
            "scsg_steps": actual_steps,
            "batch_size": self._current_batch_size,
            "skipped_clients": skipped_count,
            "active_clients": len(active_results),
            "avg_divergence": total_divergence / len(results) if results else 0.0,
        }
    
    def _scsg_step(self, theta_n: torch.Tensor, theta_0: torch.Tensor, mu_t: torch.Tensor):
        policy_n = create_policy(self.state_dim, self.action_dim, self.env_name,
                                  self.config.hidden_units, self.config.activation, self.config.output_activation)
        policy_n.set_flat_params(theta_n)
        
        policy_0 = create_policy(self.state_dim, self.action_dim, self.env_name,
                                  self.config.hidden_units, self.config.activation, self.config.output_activation)
        policy_0.set_flat_params(theta_0)
        
        all_grad_new, all_grad_old, all_ratios = [], [], []
        
        for _ in range(self.config.mini_batch_size):
            trajectory, _ = sample_trajectory(self.env, policy_n)
            returns = compute_returns(trajectory, self.config.gamma)
            
            grad_new, log_probs_n = compute_policy_gradient(trajectory, policy_n, self.config.gamma, returns)
            log_probs_0 = compute_log_probs(trajectory, policy_0)
            
            ratios = torch.exp(log_probs_0.detach() - log_probs_n.detach())
            all_ratios.append(ratios.mean().item())
            
            loss_0 = -(log_probs_0 * returns * ratios).mean()
            policy_0.zero_grad()
            loss_0.backward()
            grad_old = torch.cat([p.grad.flatten().clone() for p in policy_0.parameters() if p.grad is not None])
            
            all_grad_new.append(grad_new)
            all_grad_old.append(grad_old)
        
        v_t_n = torch.mean(torch.stack(all_grad_new), dim=0) - torch.mean(torch.stack(all_grad_old), dim=0) + mu_t
        return v_t_n, float(np.mean(all_ratios))
    
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        if server_round % 10 != 0:
            return []
        clients = client_manager.sample(num_clients=min(3, self.num_agents), min_num_clients=1)
        return [(c, fl.common.EvaluateIns(parameters, {"num_episodes": 10})) for c in clients]
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        if not results:
            return None, {}
        rewards = [float(-res.loss) for _, res in results]
        avg = float(np.mean(rewards))
        log(INFO, f"Round {server_round}: Avg Reward = {avg:.2f}")
        return -avg, {"avg_reward": avg}
    
    def evaluate(self, server_round: int, parameters: Parameters):
        rewards = []
        for _ in range(10):
            _, reward = sample_trajectory(self.env, self.policy)
            rewards.append(reward)
        avg = float(np.mean(rewards))
        return -avg, {"server_avg_reward": avg}


_dashboard_started = False


def _start_dashboard_background():
    """Start the web dashboard in a background thread (auto-starts once)."""
    global _dashboard_started
    if _dashboard_started:
        return
    _dashboard_started = True

    try:
        import threading
        from fedpg_br.dashboard.app import app as flask_app, socketio

        def run():
            try:
                socketio.run(flask_app, host="127.0.0.1", port=8050,
                             debug=False, allow_unsafe_werkzeug=True, log_output=False)
            except Exception:
                pass  # Dashboard is optional, don't crash the server

        t = threading.Thread(target=run, daemon=True)
        t.start()
        log(INFO, "Dashboard started at http://127.0.0.1:8050/experiment")
    except ImportError:
        log(INFO, "Dashboard not available (install flask: pip install -e '.[dashboard]')")


def server_fn(context: Context):
    """Create FedPG-BR server."""
    run_config = context.run_config

    env_name = str(run_config.get("env", "CartPole-v1"))
    num_rounds = int(run_config.get("num-server-rounds", 50))
    num_workers = int(run_config.get("num-workers", 10))
    num_byzantine = int(run_config.get("num-byzantine", 0))
    use_fedpg_br = bool(run_config.get("use-fedpg-br", False))
    method = str(run_config.get("method", "fedpg-br" if use_fedpg_br else "gomdp"))
    # Override config with dashboard-provided advanced settings
    cfg = get_config(env_name)
    if int(run_config.get("batch-size", 0)) > 0:
        cfg.batch_size = int(run_config["batch-size"])
    if float(run_config.get("lr", 0)) > 0:
        cfg.lr = float(run_config["lr"])
    if float(run_config.get("sigma", 0)) > 0:
        cfg.sigma = float(run_config["sigma"])
    if float(run_config.get("gamma", 0)) > 0:
        cfg.gamma = float(run_config["gamma"])
    if int(run_config.get("mini-batch-size", 0)) > 0:
        cfg.mini_batch_size = int(run_config["mini-batch-size"])
    if float(run_config.get("delta", 0)) > 0:
        cfg.delta = float(run_config["delta"])
    if int(run_config.get("max-episode-len", 0)) > 0:
        cfg.max_episode_len = int(run_config["max-episode-len"])
    if str(run_config.get("hidden-units", "")):
        hu = str(run_config["hidden-units"])
        if hu and "," in hu:
            cfg.hidden_units = tuple(int(x.strip()) for x in hu.split(","))
    if str(run_config.get("activation", "")):
        act = str(run_config["activation"])
        if act in ("ReLU", "Tanh"):
            cfg.activation = act

    byzantine_ratio = num_byzantine / num_workers if num_workers > 0 else 0.0

    # Auto-start dashboard
    _start_dashboard_background()

    log(INFO, f"FedPG-BR Server: env={env_name}, workers={num_workers}, byzantine={num_byzantine}, method={method}")

    strategy = FedPGStrategy(
        env_name=env_name,
        num_agents=num_workers,
        byzantine_ratio=byzantine_ratio,
        use_adaptive_batch=use_fedpg_br,
        method=method,
        config=cfg,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return fl.server.ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
