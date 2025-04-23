import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Running DQN on minigrid environment using standardized stable baselines 3 functions
# TODO: cite hyperparams for baseline and SOTA models and where they came from
# Optimized (SOTA): https://arxiv.org/abs/1811.12323
# Standard from stable baselines 3 docs


def make_env(env_id, seed=0):
    """Create a wrapped, monitored environment."""
    env = gym.make(env_id, render_mode="rgb_array")
    env = FlatObsWrapper(env)  # Get full grid observation
    return env


def train_dqn_agent(env_id, hyperparams, total_timesteps=100000, seed=0):
    """Train a DQN agent with given hyperparameters."""
    env = make_env(env_id, seed)

    # Create the model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=hyperparams["learning_rate"],
        buffer_size=hyperparams["buffer_size"],
        learning_starts=hyperparams["learning_starts"],
        batch_size=hyperparams["batch_size"],
        tau=hyperparams["tau"],
        gamma=hyperparams["gamma"],
        train_freq=hyperparams["train_freq"],
        gradient_steps=hyperparams["gradient_steps"],
        target_update_interval=hyperparams["target_update_interval"],
        exploration_fraction=hyperparams["exploration_fraction"],
        exploration_initial_eps=hyperparams["exploration_initial_eps"],
        exploration_final_eps=hyperparams["exploration_final_eps"],
        verbose=1,
    )

    # Create evaluation callback
    eval_env = make_env(env_id, seed)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./experiment_data/dqn_{env_id.split('/')[-1]}/",
        log_path=f"./experiment_data/dqn_{env_id.split('/')[-1]}/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    return model


# Stable baselines 3 has a built in function to evaluate the agent's performance
def evaluate_agent(model, env_id, n_eval_episodes=10, seed=0):
    eval_env = make_env(env_id, seed)
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    return mean_reward, std_reward


# edit based on what metrics are important
def plot_results(standard_rewards, optimized_rewards, env_name):
    """Plot the training results."""
    plt.figure(figsize=(10, 6))
    plt.plot(standard_rewards, label="Standard DQN")
    plt.plot(optimized_rewards, label="Research-Optimized DQN")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Mean Reward")
    plt.title(f"DQN Performance on {env_name}")
    plt.legend()
    plt.grid(True)

    # Save the plot
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/dqn_comparison_{env_name}.png")
    plt.close()


def main():
    # Environment setup
    env_id = "MiniGrid-Empty-5x5-v0"

    # Standard DQN hyperparameters (from Stable Baselines3 defaults)
    standard_hyperparams = {
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    }

    # Hyperparams from "Learning to Learn with Active Adaptive Perception -
    # TODO: check this
    # https://arxiv.org/abs/1811.12323
    optimized_hyperparams = {
        "learning_rate": 0.0001,  # Lower learning rate for stability
        "buffer_size": 10000,  # Smaller buffer for grid environments
        "learning_starts": 1000,  # Standard warmup period
        "batch_size": 128,  # Larger batch size for better gradient estimates
        "tau": 1.0,  # Full target network update
        "gamma": 0.99,  # Standard discount factor
        "train_freq": 1,  # Train every step
        "gradient_steps": 1,  # Single gradient step per update
        "target_update_interval": 100,  # More frequent target updates
        "exploration_fraction": 0.2,  # Moderate exploration
        "exploration_initial_eps": 1.0,  # Start with full exploration
        "exploration_final_eps": 0.01,  # Lower final exploration rate
    }

    # Train and evaluate standard DQN
    print("Training standard DQN...")
    standard_model = train_dqn_agent(env_id, standard_hyperparams)
    standard_reward, standard_std = evaluate_agent(standard_model, env_id)
    print(f"Standard DQN - Mean reward: {standard_reward:.2f} +/- {standard_std:.2f}")

    # Train and evaluate SOTA DQN
    print("\nTraining research-optimized DQN...")
    optimized_model = train_dqn_agent(env_id, optimized_hyperparams)
    optimized_reward, optimized_std = evaluate_agent(optimized_model, env_id)
    print(
        f"Research-Optimized DQN - Mean reward: {optimized_reward:.2f} +/- {optimized_std:.2f}"
    )


if __name__ == "__main__":
    main()
