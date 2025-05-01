import argparse
import os
import sys
import json
import itertools
from typing import Any, Dict, List

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import torch

if sys.version_info < (3, 12):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENVIRONMENTS = {
    "Empty": "MiniGrid-Empty-5x5-v0",
    "DoorKey": "MiniGrid-DoorKey-5x5-v0",
    "GoToObj": "MiniGrid-GoToObject-8x8-N2-v0",
    "MemoryS7": "MiniGrid-MemoryS7-v0",
    "KeyCorridor": "MiniGrid-KeyCorridorS3R1-v0",
    "UnlockPickup": "MiniGrid-UnlockPickup-v0",
    "MultiRoom": "MiniGrid-MultiRoom-N2-S4-v0",
    "LavaGap": "MiniGrid-LavaGapS5-v0",
}

AGENTS = {
    "DQN": DQN,
    "PPO": PPO,
}

HYPERPARAM_SPACES = {
    "DQN": {
        "learning_rate": [1e-4],
        "buffer_size": [100000],
        "learning_starts": [1000],
        "batch_size": [32],
        "tau": [1.0],
        "gamma": [0.99],
        "train_freq": [4],
        "gradient_steps": [1],
        "target_update_interval": [1000],
        "exploration_fraction": [0.1], # Increase from 0.1 -> 0.5 for more exploration
        "exploration_initial_eps": [1.0],
        "exploration_final_eps": [0.10], # Increase from 0.05 -> 0.10 for more exploration
    },
    "PPO": {
        "learning_rate": [3e-4],
        "n_steps": [2048],
        "batch_size": [64],
        "n_epochs": [10],
        "gamma": [0.99],
        "gae_lambda": [0.95],
        "clip_range": [0.2],
        "ent_coef": [0.0],
    },
}

def make_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    env = FlatObsWrapper(env)
    return env

def generate_standard_hyperparams(agent_type: str) -> Dict[str, Any]:
    """Return the first (standard) combination of hyperparameters."""
    keys = list(HYPERPARAM_SPACES[agent_type].keys())
    values = list(HYPERPARAM_SPACES[agent_type].values())
    combo = list(itertools.product(*values))[0]
    return dict(zip(keys, combo))

def train_and_evaluate(
    env_id: str,
    agent_type: str,
    hyperparams: Dict[str, Any],
    total_timesteps: int = 1_000_000,
    n_eval_episodes: int = 10,
) -> Dict[str, Any]:
    vec_env = DummyVecEnv([lambda: make_env(env_id)])
    vec_env = VecMonitor(vec_env)

    agent_class = AGENTS[agent_type]
    device = "cuda" if agent_type == "DQN" and torch.cuda.is_available() else "cpu"

    model = agent_class(
        "MlpPolicy",
        vec_env,
        device=device,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=2,
        **hyperparams,
    )

    model.learn(total_timesteps=total_timesteps)

    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    return {
        "hyperparameters": hyperparams,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
    }

def save_results(results: Dict[str, Any], results_dir: str, env_name: str, agent_type: str):
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"{agent_type.lower()}_{env_name}_baseline.json")
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments on MiniGrid environments")
    parser.add_argument("--env", type=str, choices=list(ENVIRONMENTS.keys()), required=True)
    parser.add_argument("--agent", type=str, choices=list(AGENTS.keys()), required=True)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--results_dir", type=str, default="./experiment_data/baseline")
    args = parser.parse_args()

    env_id = ENVIRONMENTS[args.env]
    hyperparams = generate_standard_hyperparams(args.agent)

    print(f"Training {args.agent} on {args.env} for {args.timesteps} timesteps...")

    results = train_and_evaluate(
        env_id=env_id,
        agent_type=args.agent,
        hyperparams=hyperparams,
        total_timesteps=args.timesteps,
        n_eval_episodes=args.n_eval_episodes,
    )

    save_results(results, args.results_dir, args.env, args.agent)
    print(f"Saved results for {args.agent} on {args.env} to {args.results_dir}")

if __name__ == "__main__":
    main()
