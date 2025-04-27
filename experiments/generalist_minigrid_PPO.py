# train_generalist_agent_ppo.py

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from pathlib import Path
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

# References:
# Proximal Policy Optimization Algorithms (PPO) - Schulman et al., 2017
# https://arxiv.org/abs/1707.06347
# Stable-Baselines3 default configs

def make_env(env_id, seed=0):
    """Create a wrapped MiniGrid environment."""
    env = gym.make(env_id, render_mode="rgb_array")
    env = FlatObsWrapper(env)  # Flatten observations
    env.reset(seed=seed)
    return env

class MultiEnvSampler(gym.Env):
    """Environment wrapper that samples randomly between multiple environments."""

    def __init__(self, env_ids, seed=0):
        self.env_ids = env_ids
        self.envs = [make_env(env_id, seed) for env_id in env_ids]
        self.current_env_idx = 0
        self.current_env = self.envs[self.current_env_idx]
        self.need_new_env = True

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def sample_env(self):
        """Sample a random environment."""
        self.current_env_idx = random.randint(0, len(self.envs) - 1)
        self.current_env = self.envs[self.current_env_idx]
        self.need_new_env = False
        return self.current_env

    def reset(self, **kwargs):
        """Reset environment, possibly changing to a new sampled environment."""
        if self.need_new_env:
            self.sample_env()
        return self.current_env.reset(**kwargs)

    def step(self, action):
        """Step in current environment."""
        obs, reward, done, truncated, info = self.current_env.step(action)
        if done or truncated:
            self.need_new_env = True
        return obs, reward, done, truncated, info

    def render(self, **kwargs):
        return self.current_env.render(**kwargs)

    def close(self):
        for env in self.envs:
            env.close()

def train_generalist_agent(env_ids, hyperparams, save_path, total_timesteps=300000, seed=0):
    """Train PPO agent across multiple environments and collect learning curve."""
    env = MultiEnvSampler(env_ids, seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hyperparams["learning_rate"],
        n_steps=hyperparams["n_steps"],
        batch_size=hyperparams["batch_size"],
        n_epochs=hyperparams["n_epochs"],
        gamma=hyperparams["gamma"],
        gae_lambda=hyperparams["gae_lambda"],
        clip_range=hyperparams["clip_range"],
        ent_coef=hyperparams["ent_coef"],
        verbose=1,
    )

    eval_env = make_env(env_ids[0], seed)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Load evaluation rewards
    eval_rewards_path = os.path.join(save_path, "evaluations.npz")
    learning_curve = None
    if os.path.exists(eval_rewards_path):
        data = np.load(eval_rewards_path)
        learning_curve = data["results"].squeeze()  # shape: (N evals,)

    return model, learning_curve


def evaluate_generalist_agent(model, env_ids, n_eval_episodes=10, seed=0):
    """Evaluate trained agent separately on each environment."""
    results = {}

    for env_id in env_ids:
        eval_env = make_env(env_id, seed)
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )
        results[env_id] = (mean_reward, std_reward)
        print(f"Environment {env_id}: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")

    overall_mean = np.mean([r[0] for r in results.values()])
    overall_std = np.mean([r[1] for r in results.values()])
    print(f"Overall generalist performance: Mean reward = {overall_mean:.2f} ± {overall_std:.2f}")

    return results

def plot_generalist_results(results, env_ids, save_name):
    """Plot performance results across environments."""
    means = [results[env_id][0] for env_id in env_ids]
    stds = [results[env_id][1] for env_id in env_ids]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(env_ids))
    plt.bar(x, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    plt.xticks(x, [env_id.split('-')[-1] for env_id in env_ids], rotation=45)
    plt.ylabel('Mean Reward')
    plt.title(f'Generalist PPO Agent Performance: {save_name}')
    plt.tight_layout()

    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/generalist_ppo_performance_{save_name}.png")
    print(f"./plots/generalist_ppo_performance_{save_name}.png")
    plt.close()

def plot_learning_curve(rewards, save_name):
    """Plot the learning curve (reward over training)."""
    plt.figure(figsize=(10, 6))
    eval_timesteps = np.arange(len(rewards)) * 10000  # assuming eval_freq = 10000
    plt.plot(eval_timesteps, rewards, marker="o")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Evaluation Reward")
    plt.title(f"Learning Curve: {save_name}")
    plt.grid(True)
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/learning_curve_{save_name}.png")
    print(f"./plots/learning_curve_{save_name}.png")
    plt.close()


def main():
    # Choose environments
    env_ids = [
        "MiniGrid-Empty-5x5-v0",
        # "MiniGrid-DoorKey-5x5-v0",
        # "MiniGrid-GoToObject-8x-N2-v0",
        # "MiniGrid-MemoryS7-v0",
        # "MiniGrid-KeyCorridorS3R3-v0",
        # "MiniGrid-UnlockPickup-v0",
        # "MiniGrid-MultiRoom-N2-S4-v0",
        # "MiniGrid-LavaGapS5-v0",
    ] 

    # --- Hyperparameters ---

    # Standard PPO (Stable Baselines3 + PPO Paper defaults)
    standard_hyperparams = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
    }

    # Optimized PPO (for small MiniGrid environments)
    optimized_hyperparams = {
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 32,
        "n_epochs": 5,
        "gamma": 0.995,
        "gae_lambda": 0.9,
        "clip_range": 0.1,
        "ent_coef": 0.001,
    }

    # --- Train and evaluate ---

    print("\nTraining generalist PPO with standard hyperparameters...")
    model_standard = train_generalist_agent(env_ids, standard_hyperparams, save_path="./experiment_data/ppo_standard")
    results_standard = evaluate_generalist_agent(model_standard, env_ids)
    plot_generalist_results(results_standard, env_ids, "standard")
    if learning_curve_standard is not None:
        plot_learning_curve(learning_curve_standard, "ppo_standard")

    print("\nTraining generalist PPO with optimized hyperparameters...")
    model_optimized = train_generalist_agent(env_ids, optimized_hyperparams, save_path="./experiment_data/ppo_optimized")
    results_optimized = evaluate_generalist_agent(model_optimized, env_ids)
    plot_generalist_results(results_optimized, env_ids, "optimized")
    plot_generalist_results(results_optimized, env_ids, "optimized")
    if learning_curve_optimized is not None:
        plot_learning_curve(learning_curve_optimized, "ppo_optimized")

if __name__ == "__main__":
    main()
