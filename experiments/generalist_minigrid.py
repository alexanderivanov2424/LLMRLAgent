import argparse
import os
from pathlib import Path
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from minigrid.wrappers import FlatObsWrapper

from agent_configs import get_hyperparams


def make_env(env_id, seed=0):
    env = gym.make(env_id, render_mode="rgb_array")
    env = FlatObsWrapper(env)
    return env


class MultiEnvSampler(gym.Env):
    # (same as you had)
    def __init__(self, env_ids, seed=0):
        self.env_ids = env_ids
        self.envs = [make_env(env_id, seed) for env_id in env_ids]
        self.current_env_idx = 0
        self.current_env = self.envs[self.current_env_idx]
        self.need_new_env = True

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def sample_env(self):
        self.current_env_idx = (self.current_env_idx + 1) % len(self.envs)
        self.current_env = self.envs[self.current_env_idx]
        self.need_new_env = False
        return self.current_env


    def reset(self, **kwargs):
        if self.need_new_env:
            self.sample_env()
        return self.current_env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.current_env.step(action)
        if done or truncated:
            self.need_new_env = True
        return obs, reward, done, truncated, info

    def render(self, **kwargs):
        return self.current_env.render(**kwargs)

    def close(self):
        for env in self.envs:
            env.close()


def create_agent(agent_type, env, hyperparams):
    if agent_type == "DQN":
        model = DQN("MlpPolicy", env, verbose=1, **hyperparams)
    elif agent_type == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, **hyperparams)
    elif agent_type == "Random":
        model = None  # Handle random separately later
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return model


def train_and_evaluate(agent_type, param_type, total_timesteps=300000, seed=0):
    env_ids = [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-GoToObject-8x8-N2-v0",
        # "MiniGrid-MemoryS7-v0",
        # "MiniGrid-KeyCorridorS3R3-v0",
        # "MiniGrid-UnlockPickup-v0",
        # "MiniGrid-MultiRoom-N2-S4-v0",
        # "MiniGrid-LavaGapS5-v0",
    ] 
    
    env = MultiEnvSampler(env_ids, seed)

    hyperparams = None
    if agent_type != "Random":
        hyperparams = get_hyperparams(agent_type, param_type)

    model = create_agent(agent_type, env, hyperparams)

    if agent_type != "Random":
        eval_callback = EvalCallback(
            make_env(env_ids[0], seed),
            best_model_save_path=f"./experiment_data/{agent_type}_{param_type}/",
            log_path=f"./experiment_data/{agent_type}_{param_type}/",
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)

        results = evaluate_generalist_agent(model, env_ids)
    else:
        results = evaluate_random_agent(env_ids, n_eval_episodes=10, seed=seed)

    plot_results(results, env_ids, agent_type, param_type)


def evaluate_generalist_agent(model, env_ids, n_eval_episodes=10, seed=0):
    results = {}
    for env_id in env_ids:
        eval_env = make_env(env_id, seed)
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
        )
        results[env_id] = (mean_reward, std_reward)
        print(f"Environment {env_id}: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")

    overall_mean = np.mean([r[0] for r in results.values()])
    overall_std = np.mean([r[1] for r in results.values()])
    print(f"Overall performance: Mean reward = {overall_mean:.2f} ± {overall_std:.2f}")

    return results


def evaluate_random_agent(env_ids, n_eval_episodes=10, seed=0):
    results = {}
    for env_id in env_ids:
        eval_env = make_env(env_id, seed)
        rewards = []
        for _ in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            truncated = False
            total_reward = 0
            while not (done or truncated):
                action = eval_env.action_space.sample()
                obs, reward, done, truncated, info = eval_env.step(action)
                total_reward += reward
            rewards.append(total_reward)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        results[env_id] = (mean_reward, std_reward)
        print(f"Environment {env_id}: Random agent reward = {mean_reward:.2f} ± {std_reward:.2f}")

    return results


def plot_results(results, env_ids, agent_type, param_type):
    print(env_ids)
    means = [results[env_id][0] for env_id in env_ids]
    stds = [results[env_id][1] for env_id in env_ids]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(env_ids))
    plt.bar(x, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    plt.xticks(x, [''.join(env_id.split('-')[1:-2]) for env_id in env_ids], rotation=45)
    plt.ylabel('Mean Reward')
    plt.title('Agent Performance Across Environments')
    plt.tight_layout()

    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/{agent_type}_{param_type}_performance.png")

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", type=str, required=True, choices=["DQN", "PPO", "Random"],
                        help="Agent to train: DQN, PPO, or Random")
    parser.add_argument("--param_type", type=str, required=False, choices=["standard", "optimized", "generalist"],
                        help="Hyperparameter set to use: standard, optimized, or generalist (DQN only rn).")
    args = parser.parse_args()

    train_and_evaluate(args.agent_type, args.param_type)


if __name__ == "__main__":
    main()
