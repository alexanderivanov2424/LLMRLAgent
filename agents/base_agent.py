from typing import Tuple

import gymnasium as gym


class BaseAgent:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def get_agent_ID(self) -> str:
        """Return the agent's identifier."""
        raise NotImplementedError("Subclasses must implement this method")

    def policy(self, observation):
        """Select an action based on the current observation."""
        raise NotImplementedError("Subclasses must implement this method")

    def update(self, observation, action, reward, terminated, truncated):
        """Update the agent's internal state after taking an action."""
        raise NotImplementedError("Subclasses must implement this method")

    def train(
        self,
        env: gym.Env,
        total_timesteps: int = 100000,
        seed: int = 0,
    ):
        """
        Train the agent on the environment.

        Args:
            env: The environment to train on
            total_timesteps: Total number of timesteps to train
            seed: Random seed

        Returns:
            The trained model
        """
        raise NotImplementedError("Subclasses must implement this method")

    def evaluate(
        self,
        model,
        env: gym.Env,
        n_eval_episodes: int = 10,
    ) -> Tuple[float, float]:
        """
        Evaluate the agent's performance.

        Args:
            model: The trained model
            env: The environment to evaluate on
            n_eval_episodes: Number of episodes to evaluate

        Returns:
            Tuple of (mean_reward, std_reward)
        """
        raise NotImplementedError("Subclasses must implement this method")
