import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# For older python versions
if sys.version_info < (3, 12):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Available environments
ENVIRONMENTS = {
    "Empty": "MiniGrid-Empty-5x5-v0",
    "DoorKey": "MiniGrid-DoorKey-5x5-v0",
    "GoToObj": "MiniGrid-GoToObject-6x6-N2-v0",
    "MemoryS7": "MiniGrid-MemoryS11-v0",
}

# Environment-specific hyperparameters
ENV_HYPERPARAMS = {
    "Empty": {
        "DQN": {
            "learning_starts": 10000,
            "buffer_size": 50000,
            "gamma": 0.99,
            "learning_rate": 0.0003,
            "batch_size": 128,
            "train_freq": 16,
            "gradient_steps": 8,
            "target_update_interval": 800,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
        },
        "PPO": {
            "learning_rate": 3e-5,
            "n_epochs": 20,
            "gae_lambda": 0.9,
        },
    },
    "DoorKey": {
        "DQN": {
            "learning_starts": 50000,
            "buffer_size": 100000,
            "gamma": 0.99,
            "learning_rate": 0.0003,
            "batch_size": 128,
            "train_freq": 16,
            "gradient_steps": 8,
            "target_update_interval": 800,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
        },
        "PPO": {
            "learning_rate": 3e-5,
            "n_epochs": 20,
            "gae_lambda": 0.9,
        },
    },
    "GoToObj": {
        "DQN": {
            "learning_starts": 10000,
            "buffer_size": 500000,
            "gamma": 0.99,
            "learning_rate": 0.0003,
            "batch_size": 128,
            "train_freq": 16,
            "gradient_steps": 8,
            "target_update_interval": 800,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
        },
        "PPO": {
            "learning_rate": 3e-5,
            "n_epochs": 20,
            "gae_lambda": 0.9,
        },
    },
    "MemoryS7": {
        "DQN": {
            "learning_starts": 50000,
            "buffer_size": 500000,
            "gamma": 0.99,
            "learning_rate": 0.0003,
            "batch_size": 128,
            "train_freq": 16,
            "gradient_steps": 8,
            "target_update_interval": 800,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
        },
        "PPO": {
            "learning_rate": 3e-5,
            "n_epochs": 20,
            "gae_lambda": 0.95,
        },
    },
    # Add more environment-specific hyperparameters as needed
}


def make_env(env_id: str) -> gym.Env:
    """Create a wrapped, monitored environment."""
    env = gym.make(env_id)
    env = FlatObsWrapper(env)  # Get full grid observation
    return env


class LearningCurveCallback:
    def __init__(self, eval_freq: int = 10000, n_eval_episodes: int = 5):
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.timesteps = []
        self.mean_rewards = []
        self.std_rewards = []

    def __call__(self, locals_: Dict, globals_: Dict) -> bool:
        if locals_["self"].num_timesteps % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                locals_["self"],
                locals_["self"].get_env(),
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )
            self.timesteps.append(locals_["self"].num_timesteps)
            self.mean_rewards.append(float(mean_reward))
            self.std_rewards.append(float(std_reward))
        return True


def train_and_evaluate(
    env_id: str,
    agent_type: str,
    hyperparams: Dict[str, Any],
    total_timesteps: int = 1000000,
    n_eval_episodes: int = 20,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """Train and evaluate an agent with given hyperparameters and capture learning curve."""
    vec_env = DummyVecEnv([lambda: make_env(env_id)])
    vec_env = VecMonitor(vec_env)

    agent_class = DQN if agent_type == "DQN" else PPO

    if agent_type == "DQN":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    # Initialize learning curve callback
    learning_curve_callback = LearningCurveCallback(eval_freq=100, n_eval_episodes=5)


    model = agent_class(
        "MlpPolicy",
        vec_env,
        device=device,
        **hyperparams,
    )

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10,
        progress_bar=True,
        callback=learning_curve_callback,
    )

    # Final evaluation
    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    # Convert learning curve data to dictionary
    learning_curve = {
        "timesteps": learning_curve_callback.timesteps,
        "mean_rewards": learning_curve_callback.mean_rewards,
        "std_rewards": learning_curve_callback.std_rewards,
    }

    return {
        "hyperparameters": hyperparams,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
    }, learning_curve


def save_results(
    results: Dict[str, Any],
    learning_curve: Dict[str, List[float]],
    env_name: str,
    agent_type: str,
):
    """Save results and learning curve to JSON files."""
    os.makedirs("./experiment_data/single_agent", exist_ok=True)

    # Save main results
    results_filename = (
        f"./experiment_data/single_agent/{agent_type.lower()}_{env_name}_results.json"
    )
    with open(results_filename, "w") as f:
        json.dump(results, f, indent=2)

    # Save learning curve
    curve_filename = f"./experiment_data/single_agent/{agent_type.lower()}_{env_name}_learning_curve.json"
    with open(curve_filename, "w") as f:
        json.dump(learning_curve, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a single agent on MiniGrid environments"
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=list(ENVIRONMENTS.keys()),
        nargs="+",  # Allow multiple values
        default=list(ENVIRONMENTS.keys()),
        help="Environments to run (can specify multiple)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["DQN", "PPO"],
        default="DQN",
        help="Agent type to use",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000000,
        help="Total number of training timesteps",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=20,
        help="Number of episodes for final evaluation",
    )
    args = parser.parse_args()

    # Process each environment
    for env_name in args.env:
        env_id = ENVIRONMENTS[env_name]
        hyperparams = ENV_HYPERPARAMS[env_name][args.agent]

        print(f"\nTraining {args.agent} on {env_name} environment...")
        print(f"Hyperparameters: {hyperparams}")

        results, learning_curve = train_and_evaluate(
            env_id,
            args.agent,
            hyperparams,
            args.timesteps,
            args.n_eval_episodes,
        )

        save_results(results, learning_curve, env_name, args.agent)

        print(f"\nResults for {env_name}:")
        print(
            f"Mean reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}"
        )
        print(f"Results and learning curve saved to ./experiment_data/single_agent/")


if __name__ == "__main__":
    main()


