"""Traffic light environment for federated reinforcement learning.

This environment simulates a single traffic light intersection where the agent
learns to optimize traffic flow by controlling the light phases.
"""

import random
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TrafficLightEnv(gym.Env):
    """Single traffic light intersection environment.

    State: [queue_north, queue_south, queue_east, queue_west, current_phase, time_in_phase]
    Action: 0 = keep current phase, 1 = switch to next phase
    Phases: 0 = North-South green, 1 = East-West green

    Reward: Negative total waiting time (minimize waiting)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, intersection_id: int = 0, traffic_intensity: float = 0.5):
        """Initialize traffic light environment.

        Args:
            intersection_id: Unique ID for this intersection
            traffic_intensity: Traffic arrival rate (0-1)
        """
        super().__init__()

        self.intersection_id = intersection_id
        self.traffic_intensity = traffic_intensity

        # State space: [queue_N, queue_S, queue_E, queue_W, phase, time_in_phase]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([50, 50, 50, 50, 1, 120]),
            dtype=np.float32,
        )

        # Action space: 0 = keep, 1 = switch
        self.action_space = spaces.Discrete(2)

        # Environment state
        self.queue_north = 0
        self.queue_south = 0
        self.queue_east = 0
        self.queue_west = 0
        self.current_phase = 0  # 0 = NS green, 1 = EW green
        self.time_in_phase = 0
        self.total_time = 0
        self.max_steps = 1000

        # Constants
        self.min_green_time = 10  # Minimum green time before switch
        self.yellow_time = 3
        self.car_service_rate = 2  # Cars per timestep when green

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Initialize random queues
        self.queue_north = random.randint(0, 5)
        self.queue_south = random.randint(0, 5)
        self.queue_east = random.randint(0, 5)
        self.queue_west = random.randint(0, 5)
        self.current_phase = 0
        self.time_in_phase = 0
        self.total_time = 0

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one timestep.

        Args:
            action: 0 = keep current phase, 1 = switch phase

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Handle action (switch or keep)
        if action == 1 and self.time_in_phase >= self.min_green_time:
            # Switch phase (with yellow time)
            self.current_phase = 1 - self.current_phase
            self.time_in_phase = 0
        else:
            self.time_in_phase += 1

        # Simulate traffic arrivals (Poisson-like)
        if random.random() < self.traffic_intensity:
            direction = random.choice(["north", "south", "east", "west"])
            if direction == "north":
                self.queue_north = min(self.queue_north + 1, 50)
            elif direction == "south":
                self.queue_south = min(self.queue_south + 1, 50)
            elif direction == "east":
                self.queue_east = min(self.queue_east + 1, 50)
            else:
                self.queue_west = min(self.queue_west + 1, 50)

        # Service cars based on current phase
        if self.current_phase == 0:  # North-South green
            self.queue_north = max(0, self.queue_north - self.car_service_rate)
            self.queue_south = max(0, self.queue_south - self.car_service_rate)
        else:  # East-West green
            self.queue_east = max(0, self.queue_east - self.car_service_rate)
            self.queue_west = max(0, self.queue_west - self.car_service_rate)

        # Calculate reward (negative total waiting time)
        total_queue = (
            self.queue_north + self.queue_south + self.queue_east + self.queue_west
        )
        reward = -total_queue  # Minimize total queue

        # Check termination
        self.total_time += 1
        terminated = False
        truncated = self.total_time >= self.max_steps

        info = {
            "total_queue": total_queue,
            "phase": self.current_phase,
            "time_in_phase": self.time_in_phase,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return np.array(
            [
                self.queue_north,
                self.queue_south,
                self.queue_east,
                self.queue_west,
                self.current_phase,
                self.time_in_phase,
            ],
            dtype=np.float32,
        )

    def render(self):
        """Render the environment (console output)."""
        if self.current_phase == 0:
            phase_str = "North-South GREEN"
        else:
            phase_str = "East-West GREEN"

        print(f"\n=== Traffic Light {self.intersection_id} ===")
        print(f"Phase: {phase_str} (time: {self.time_in_phase}s)")
        print(f"Queues: N={self.queue_north} S={self.queue_south} "
              f"E={self.queue_east} W={self.queue_west}")
        print(f"Total waiting: {self.queue_north + self.queue_south + self.queue_east + self.queue_west}")


# Register environment with Gymnasium
gym.register(
    id="TrafficLight-v1",
    entry_point="fedpg_br.envs.traffic_light_env:TrafficLightEnv",
    max_episode_steps=1000,
)
