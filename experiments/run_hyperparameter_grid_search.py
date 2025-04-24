import argparse
import itertools
import json
import os
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from agents.llm_agent import LLMAgent
from agents.llm_context_agent import LLMContextAgent
from agents.random_agent import RandomAgent

# Available environments
ENVIRONMENTS = {
    "Empty": "MiniGrid-Empty-5x5-v0",
    "DoorKey": "MiniGrid-DoorKey-5x5-v0",
    "GoToObj": "MiniGrid-GoToObject-5x5-v0",
    "MemoryS7": "MiniGrid-MemoryS7-v0",
    "KeyCorridor": "MiniGrid-KeyCorridorS3R1-v0",
    "UnlockPickup": "MiniGrid-UnlockPickup-v0",
    "MultiRoom": "MiniGrid-MultiRoom-N2-S4-v0",
    "LavaGap": "MiniGrid-LavaGapS5-v0",
}

# Available agents
AGENTS = {
    "DQN": DQN,
    "PPO": PPO,
}

# Hyperparameter search spaces for each agent
HYPERPARAM_SPACES = {
    "DQN": {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "buffer_size": [10000, 50000, 100000],
        "batch_size": [32, 64, 128],
        "gamma": [0.9, 0.95, 0.99],
        "exploration_fraction": [0.1, 0.2, 0.3],
        "exploration_final_eps": [0.01, 0.05, 0.1],
    },
    "PPO": {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "n_steps": [1024, 2048, 4096],
        "batch_size": [32, 64, 128],
        "n_epochs": [5, 10, 20],
        "gamma": [0.9, 0.95, 0.99],
        "gae_lambda": [0.9, 0.95, 0.99],
        "clip_range": [0.1, 0.2, 0.3],
    },
}


def make_env(env_id: str, seed: int = 0) -> gym.Env:
    """Create a wrapped, monitored environment."""
    env = gym.make(env_id, render_mode="rgb_array")
    env = FlatObsWrapper(env)  # Get full grid observation
    return env


def train_and_evaluate(
    env_id: str,
    agent_type: str,
    hyperparams: Dict[str, Any],
    total_timesteps: int = 100000,
    seed: int = 0,
    n_eval_episodes: int = 10,
) -> Dict[str, Any]:
    """Train and evaluate an agent with given hyperparameters."""
    env = make_env(env_id, seed)

    # Get the agent class
    agent_class = AGENTS[agent_type]

    # Create the model
    model = agent_class(
        "MlpPolicy",
        env,
        **hyperparams,
        verbose=0,  # Suppress training output
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Evaluate the model
    eval_env = make_env(env_id, seed)
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    return {
        "hyperparameters": hyperparams,
        "mean_reward": float(
            mean_reward[0] if isinstance(mean_reward, list) else mean_reward
        ),
        "std_reward": float(
            std_reward[0] if isinstance(std_reward, list) else std_reward
        ),
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
        type=str,
        choices=list(ENVIRONMENTS.keys()),
        default="Empty",
        help="Environment to run",
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
        default=100000,
        help="Total number of training timesteps",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation",
    )
    args = parser.parse_args()

    env_id = ENVIRONMENTS[args.env]
    hyperparam_combinations = generate_hyperparameter_combinations(args.agent)

    print(f"Starting grid search for {args.agent} on {args.env} environment...")
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
            args.seed,
            args.n_eval_episodes,
        )
        results.append(result)
        print(
            f"Mean reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}"
        )

    # Save results
    save_results(results, args.env, args.agent)

    # Find best configuration
    best_result = max(results, key=lambda x: x["mean_reward"])
    print("\nBest configuration:")
    print(f"Hyperparameters: {best_result['hyperparameters']}")
    print(
        f"Mean reward: {best_result['mean_reward']:.2f} +/- {best_result['std_reward']:.2f}"
    )


if __name__ == "__main__":
    main()
