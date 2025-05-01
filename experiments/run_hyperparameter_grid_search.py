import argparse
import itertools
import json
import os
import sys
from typing import Any, Dict, List

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# For older python versions
if sys.version_info < (3, 12):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from agents.llm_agent import LLMAgent
from agents.llm_context_agent import LLMContextAgent
from agents.random_agent import RandomAgent

# Available environments
ENVIRONMENTS = {
    "Empty": "MiniGrid-Empty-5x5-v0",
    "DoorKey": "MiniGrid-DoorKey-5x5-v0",
    "GoToObj": "MiniGrid-GoToObject-6x6-N2-v0",
    "MemoryS7": "MiniGrid-MemoryS11-v0",
    "KeyCorridor": "MiniGrid-KeyCorridorS6R3-v0",
    "UnlockPickup": "MiniGrid-Unlock-v0",
    "MultiRoom": "MiniGrid-MultiRoom-N4-S5-v0",
    "LavaGap": "MiniGrid-LavaGapS5-v0",
}

# Available agents
AGENTS = {
    "DQN": DQN,
    "PPO": PPO,
}

HYPERPARAM_SPACES = {
    "DQN": {
        # "learning_rate": [1e-6, 5e-6, 1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01],
        # "train_freq": [1, 2, 4, 16, 48],
        # "exploration_initial_eps": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        # "exploration_final_eps": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        # "batch_size": [16, 32, 64, 128, 256],
        # "gradient_steps": [1, 2, 4, 8, 16],
        # "learning_starts": [1000, 10000, 100000],
        # "tau": [0.001, 0.01, 0.05],
        # "buffer_size": [500, 5000, 50000, 500000],
        #
        #
        # First set of hyperparameters
        "learning_starts": [10000, 50000, 100000],
        "buffer_size": [50000, 100000, 500000],
        "gamma": [0.99],
        "learning_rate": [3e-4],
        "batch_size": [128],
        "train_freq": [16],
        "gradient_steps": [8],
        "target_update_interval": [800],
        "exploration_fraction": [0.1],
        "exploration_final_eps": [0.05],
        #
        # Second set of hyperparameters
        # "learning_starts": [10000, 50000, 100000],
        # "buffer_size": [500000, 1000000, 5000000],
        # "gamma": [0.99],
        # "learning_rate": [3e-4],
        # "batch_size": [128],
        # "train_freq": [16],
        # "gradient_steps": [8],
        # "target_update_interval": [800],
    },
    "PPO": {
        "learning_rate": [3e-3, 3e-4, 3e-5],
        "n_epochs": [5, 10, 20],
        "gae_lambda": [0.9, 0.95, 0.99],
    },
}


def make_env(env_id: str) -> gym.Env:
    """Create a wrapped, monitored environment."""
    env = gym.make(env_id)
    env = FlatObsWrapper(env)  # Get full grid observation
    return env


def train_and_evaluate(
    env_id: str,
    agent_type: str,
    hyperparams: Dict[str, Any],
    total_timesteps: int = 1000000,
    n_eval_episodes: int = 20,
) -> Dict[str, Any]:
    """Train and evaluate an agent with given hyperparameters."""
    vec_env = DummyVecEnv([lambda: make_env(env_id)])
    vec_env = VecMonitor(vec_env)

    agent_class = AGENTS[agent_type]

    if agent_type == "DQN":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

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
    )

    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    mean_reward = float(mean_reward)
    std_reward = float(std_reward)

    return {
        "hyperparameters": hyperparams,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    }


def generate_hyperparameter_combinations(agent_type: str) -> List[Dict[str, Any]]:
    """Generate all combinations of hyperparameters for grid search."""
    param_space = HYPERPARAM_SPACES[agent_type]
    if not param_space:
        return [{}]  # Return empty dict for agents with no hyperparameters

    # Generate all combinations
    keys = param_space.keys()
    values = param_space.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations


def save_results(results: List[Dict[str, Any]], env_name: str, agent_type: str):
    """Save grid search results to a JSON file."""
    os.makedirs("./experiment_data/grid_search", exist_ok=True)
    filename = f"./experiment_data/grid_search/{agent_type.lower()}_{env_name}_grid_search.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter grid search on MiniGrid environments"
    )
    parser.add_argument(
        "--env",
        action="append",
        choices=list(ENVIRONMENTS.keys()),
        default=None,
        help="Environments to run (can be specified multiple times)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=list(AGENTS.keys()),
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
        help="Number of episodes for evaluation",
    )
    args = parser.parse_args()

    # If no environments specified, use all environments
    if args.env is None:
        args.env = list(ENVIRONMENTS.keys())

    print(f"Running grid search for {args.agent} on {args.env} environments...\n\n")

    for env_name in args.env:
        env_id = ENVIRONMENTS[env_name]
        hyperparam_combinations = generate_hyperparameter_combinations(args.agent)

        print(f"Starting grid search for {args.agent} on {env_name} environment...")
        print(f"Total combinations to try: {len(hyperparam_combinations)}")

        results = []
        for i, hyperparams in enumerate(hyperparam_combinations, 1):
            print(f"\nTrying combination {i}/{len(hyperparam_combinations)}")
            print(f"Hyperparameters: {hyperparams}")

            result = train_and_evaluate(
                env_id,
                args.agent,
                hyperparams,
                args.timesteps,
                args.n_eval_episodes,
            )
            results.append(result)
            print(
                f"Mean reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}"
            )

        save_results(results, env_name, args.agent)

        best_result = max(results, key=lambda x: x["mean_reward"])
        print("\nBest configuration:")
        print(f"Hyperparameters: {best_result['hyperparameters']}")
        print(
            f"Mean reward: {best_result['mean_reward']:.2f} +/- {best_result['std_reward']:.2f}"
        )


if __name__ == "__main__":
    main()
