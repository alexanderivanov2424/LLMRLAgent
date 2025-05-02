import argparse
import os
import sys
from pathlib import Path
import random

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from minigrid.wrappers import FlatObsWrapper

from agent_configs import get_hyperparams
from utils.experiment_data import ExperimentData


def make_env(env_id, seed=0):
    env = gym.make(env_id, render_mode="rgb_array")
    env = FlatObsWrapper(env)
    return env


class MultiEnvSampler(gym.Env):
    # (same as you had)
    def __init__(self, env_ids, seed=0, episodes_per_env=1):
        self.env_ids = env_ids
        self.envs = [make_env(env_id, seed) for env_id in env_ids]
        self.current_env_idx = 0
        self.current_env = self.envs[self.current_env_idx]
        self.episodes_per_env = episodes_per_env
        self.episodes_in_current_env = 0

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # Start with the first environment
        self._select_env(0) 

    def _select_env(self, env_idx):
        """Selects a specific environment index and resets episode count."""
        self.current_env_idx = env_idx
        self.current_env = self.envs[self.current_env_idx]
        self.episodes_in_current_env = 0
        # print(f"Switched to env: {self.env_ids[self.current_env_idx]}") # Optional debug print

    def sample_env(self):
        """Samples the next environment sequentially."""
        next_env_idx = (self.current_env_idx + 1) % len(self.envs)
        self._select_env(next_env_idx)
        return self.current_env

    def reset(self, **kwargs):
        # Check if we need to switch environments before resetting
        if self.episodes_in_current_env >= self.episodes_per_env:
            self.sample_env()
            
        # Reset the current environment
        obs, info = self.current_env.reset(**kwargs) 
        # Note: We return obs, info because gymnasium reset can return info
        return obs, info 

    def step(self, action):
        obs, reward, done, truncated, info = self.current_env.step(action)
        if done or truncated:
            # Increment episode count for the current environment
            self.episodes_in_current_env += 1 
            # The reset method will handle switching if needed
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


class GeneralistEvalCallback(EvalCallback):
    def __init__(self, train_env_ids, experiment_data, agent_id, n_eval_episodes=5, eval_freq=10000, 
                 log_path=None, best_model_save_path=None, deterministic=True, render=False, verbose=1, warn=True):
        
        # We override __init__ completely as we don't use a single eval_env in the same way
        # We call the parent's parent __init__ directly if needed (BaseCallback)
        super(EvalCallback, self).__init__(verbose=verbose)
        
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        
        self.experiment_data = experiment_data
        self.agent_id = agent_id
        self.episode_num = 0 # Represents evaluation event number
        self.last_eval_step = 0
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        
        # Store train_env_ids and create evaluation environments for them
        self.eval_env_ids = train_env_ids
        self.eval_envs = [make_env(env_id) for env_id in self.eval_env_ids]
         
        # Create a simple agent-like object for logging (needed for ExperimentData methods)
        self.agent_wrapper = type('AgentWrapper', (), {'get_agent_ID': lambda self: agent_id})()
         
    def _init_callback(self) -> None:
        # This is called before the first call to _on_step
        # Ensure log paths exist
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)
        # Ensure save paths exist
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            all_env_rewards = {}
            all_env_lengths = []

            # Evaluate on each environment
            for i, env_id in enumerate(self.eval_env_ids):
                eval_env = self.eval_envs[i]
                
                # Use evaluate_policy from stable_baselines3
                episode_rewards, episode_lengths = evaluate_policy(
                    self.model, # The model is accessed via self.model
                    eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                )

                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                mean_ep_length = np.mean(episode_lengths)

                if self.verbose > 0:
                    print(f"Eval {self.episode_num} Env {env_id}: reward={mean_reward:.2f} +/- {std_reward:.2f}, len={mean_ep_length:.2f}")

                # Store result for this env
                all_env_rewards[env_id] = mean_reward
                all_env_lengths.append(mean_ep_length)

            # Calculate overall mean reward across environments for this evaluation event
            overall_mean_reward = np.mean(list(all_env_rewards.values()))
            overall_mean_length = np.mean(all_env_lengths)

            if self.verbose > 0:
                print(f"Eval {self.episode_num} Summary: overall_mean_reward={overall_mean_reward:.2f}")

            # Log the dictionary of rewards using the new method
            self.experiment_data.log_agent_multi_env_eval_rewards(
                self.agent_id,
                self.episode_num,
                all_env_rewards
            )

            # Log the *overall average* episode length for this eval event
            self.experiment_data.log_agent_episode_length(
                self.agent_wrapper,
                self.episode_num,
                overall_mean_length
            )
            
            # Also log the overall mean reward as the 'sum'/'avg' for compatibility with old plots if needed
            # Or potentially for deciding the best model based on overall performance
            self.experiment_data.log_agent_episode_reward_meta_stats(
                self.agent_wrapper,
                self.episode_num,
                overall_mean_reward, # Use overall mean for sum
                overall_mean_reward # Use overall mean for avg
            )

            self.last_mean_reward = overall_mean_reward # Update for potential best model saving
            self.last_eval_step = self.n_calls
            self.episode_num += 1

            # Save the model if it's the best overall performance so far
            if overall_mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(f"New best mean reward! {overall_mean_reward:.2f} > {self.best_mean_reward:.2f}")
                self.best_mean_reward = overall_mean_reward
                if self.best_model_save_path is not None:
                    save_path = os.path.join(self.best_model_save_path, "best_model")
                    if self.verbose > 0:
                        print(f"Saving best model to {save_path}")
                    self.model.save(save_path)

            # Trigger experiment save after logging
            self.experiment_data.save()
        
        return continue_training


def train_and_evaluate(agent_type, param_type, total_timesteps=300000, seed=0, episodes_per_env=5, eval_freq=10000):
    print("Training and evaluating...")
    train_env_ids = [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-GoToObject-8x8-N2-v0",
        "MiniGrid-MemoryS7-v0", 
        # "MiniGrid-KeyCorridorS3R3-v0",
        # "MiniGrid-UnlockPickup-v0",
        # "MiniGrid-MultiRoom-N2-S4-v0",
        # "MiniGrid-LavaGapS5-v0",
    ]

    # eval_env_id = "MiniGrid-Empty-8x8-v0" # Changed: Use a training env for periodic eval
    eval_env_id = "MiniGrid-Empty-5x5-v0"

    train_env = MultiEnvSampler(train_env_ids, seed, episodes_per_env=episodes_per_env)
    # We only need the single eval_env for the Random agent now
    if agent_type == "Random":
        eval_env = make_env(eval_env_id, seed)
    else:
        eval_env = None # Not used by the callback anymore

    # Experiment name should reflect the environment used for evaluation logging
    # Let's simplify the name now, as the eval env is consistent
    experiment_name = f"Generalist_{agent_type}_{param_type}"
    experiment = ExperimentData.load(experiment_name)
    
    hyperparams = None
    if agent_type != "Random":
        hyperparams = get_hyperparams(agent_type, param_type)

    model = create_agent(agent_type, train_env, hyperparams)
    
    agent_id = f"{agent_type}_{param_type}"

    if agent_type != "Random":
        eval_callback = GeneralistEvalCallback(
            train_env_ids,
            experiment,
            agent_id,
            best_model_save_path=f"./experiment_data/{agent_type}_{param_type}/",
            log_path=f"./experiment_data/{agent_type}_{param_type}/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)

        results = evaluate_generalist_agent(model, train_env_ids)
    else:
        agent_id = "RandomAgent"
        
        agent_wrapper = type('AgentWrapper', (), {'get_agent_ID': lambda self: agent_id})()
        
        # Simulate periodic evaluation for the random agent
        num_evals = total_timesteps // eval_freq
        n_eval_episodes_per_period = 10 # Number of episodes to run per evaluation period

        try:
            # Simulate timesteps and evaluate periodically
            for eval_idx in range(num_evals):
                current_timestep = (eval_idx + 1) * eval_freq 
                rewards = []
                episode_lengths = [] # Also track lengths if needed
                
                # print(f"Evaluating Random Agent at timestep {current_timestep}...") # Optional debug
                
                for _ in range(n_eval_episodes_per_period):  
                    try:
                        obs, _ = eval_env.reset()
                        done = False
                        truncated = False
                        episode_reward = 0
                        episode_length = 0
                        while not (done or truncated):
                            action = eval_env.action_space.sample()
                            obs, reward, done, truncated, info = eval_env.step(action)
                            episode_reward += reward
                            episode_length += 1
                        rewards.append(episode_reward)
                        episode_lengths.append(episode_length)
                    except Exception as e:
                        print(f"Error during random agent evaluation episode: {e}")
                
                if rewards:  
                    mean_reward = np.mean(rewards)
                    mean_length = np.mean(episode_lengths) # Calculate mean length
                    
                    experiment.log_agent_episode_reward_meta_stats(
                        agent_wrapper, 
                        eval_idx, # Log against evaluation index (pseudo-episode)
                        mean_reward,  # sum_reward (using mean here for consistency)
                        mean_reward   # avg_reward 
                    )
                    
                    experiment.log_agent_episode_length(
                        agent_wrapper,
                        eval_idx, # Log against evaluation index
                        mean_length # Log mean episode length for this eval period
                    )
                    
                    experiment.save()
                    # print(f"Random agent evaluation {eval_idx} (at step {current_timestep}): mean reward = {mean_reward:.2f}") # Optional debug
                
        except Exception as e:
            print(f"Error in random agent evaluation loop: {e}")
            
        results = evaluate_random_agent(train_env_ids, n_eval_episodes=10, seed=seed)

    experiment.save()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", type=str, required=True, choices=["DQN", "PPO", "Random"],
                        help="Agent to train: DQN, PPO, or Random")
    parser.add_argument("--param_type", type=str, required=False, default="standard", choices=["standard", "optimized", "generalist"],
                        help="Hyperparameter set to use: standard, optimized, or generalist (DQN only rn).")
    parser.add_argument("--timesteps", type=int, default=300000, help="Total training timesteps.")
    parser.add_argument("--episodes_per_env", type=int, default=5, help="Number of episodes per training environment before switching.")
    parser.add_argument("--eval_freq", type=int, default=10000, help="Evaluate the agent every N timesteps.")
    args = parser.parse_args()

    train_and_evaluate(
        args.agent_type, 
        args.param_type, 
        total_timesteps=args.timesteps, 
        episodes_per_env=args.episodes_per_env,
        eval_freq=args.eval_freq
    )


if __name__ == "__main__":
    main()
