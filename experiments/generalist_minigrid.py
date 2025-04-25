import os
from pathlib import Path
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


# NOTE: Maybe move this to envs folder? 
# ...originally did not plan to make a whole environment wrapper



def make_env(env_id, seed=0):
    """Create a wrapped, monitored environment."""
    env = gym.make(env_id, render_mode="rgb_array")
    env = FlatObsWrapper(env)  # Get full grid observation
    return env


class MultiEnvSampler(gym.Env):
    """Environment wrapper that samples from multiple environments."""
    
    def __init__(self, env_ids, seed=0):
        self.env_ids = env_ids
        self.envs = [make_env(env_id, seed) for env_id in env_ids]
        self.current_env_idx = 0
        self.current_env = self.envs[self.current_env_idx]
        self.need_new_env = True
        
        # NOTE: this assumes all environments have the same observation and action spaces
        # should be true for minigrid environments, but be careful with other environments
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
    def sample_env(self):
        """Sample a random environment."""
        self.current_env_idx = random.randint(0, len(self.envs) - 1)
        self.current_env = self.envs[self.current_env_idx]
        self.need_new_env = False
        return self.current_env
        
    def reset(self, **kwargs):
        """Reset the environment, potentially sampling a new one."""
        if self.need_new_env:
            self.sample_env()
        return self.current_env.reset(**kwargs)
    
    def step(self, action):
        """Take a step in the current environment."""
        obs, reward, done, truncated, info = self.current_env.step(action)
        
        if done or truncated:
            self.need_new_env = True
            
        return obs, reward, done, truncated, info
    
    def render(self, **kwargs):
        """Render the current environment."""
        return self.current_env.render(**kwargs)
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

def train_generalist_agent(env_ids, hyperparams, total_timesteps=300000, seed=0):
    """Train a generalist DQN agent across multiple environments."""
    # Create multi-environment sampler
    env = MultiEnvSampler(env_ids, seed)
    
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
    
    eval_envs = [make_env(env_id, seed) for env_id in env_ids]
    
    eval_callback = EvalCallback(
        eval_envs[0],
        best_model_save_path=f"./experiment_data/generalist_dqn/",
        log_path=f"./experiment_data/generalist_dqn/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    
    return model


def evaluate_generalist_agent(model, env_ids, n_eval_episodes=10, seed=0):
    """Evaluate a generalist agent on multiple environments."""
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
    
    # Calculate overall performance
    overall_mean = np.mean([r[0] for r in results.values()])
    overall_std = np.mean([r[1] for r in results.values()])
    print(f"Overall performance: Mean reward = {overall_mean:.2f} ± {overall_std:.2f}")
    
    return results


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


def plot_generalist_results(results, env_ids):
    """Plot the performance of the generalist agent across environments."""
    means = [results[env_id][0] for env_id in env_ids]
    stds = [results[env_id][1] for env_id in env_ids]
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(env_ids))
    plt.bar(x, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    plt.xticks(x, [env_id.split('-')[-1] for env_id in env_ids], rotation=45)
    plt.ylabel('Mean Reward')
    plt.title('Generalist Agent Performance Across Environments')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("./plots", exist_ok=True)
    plt.savefig("./plots/generalist_dqn_performance.png")
    plt.close()


def main():

    env_ids = [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-LavaGapS5-v0",
        # "MiniGrid-Fetch-5x5-N2-v0",
        # "MiniGrid-Dynamic-Obstacles-5x5-v0",
        # "MiniGrid-FourRooms-v0"
    ]

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

    # Generalist hyperparameters (based on optimized but with larger buffer)
    generalist_hyperparams = {
        "learning_rate": 0.0001,  
        "buffer_size": 50000,  # Larger buffer to store experiences from multiple environments
        "learning_starts": 5000,  # Longer warmup to collect diverse experiences
        "batch_size": 128,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "target_update_interval": 100,
        "exploration_fraction": 0.3,  # More exploration for diverse environments
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    }
    
    # Train and evaluate generalist agent
    print("\nTraining generalist DQN across multiple environments...")
    generalist_model = train_generalist_agent(env_ids, generalist_hyperparams)
    results = evaluate_generalist_agent(generalist_model, env_ids)
    
    # Plot generalist results
    plot_generalist_results(results, env_ids)


if __name__ == "__main__":
    main()
