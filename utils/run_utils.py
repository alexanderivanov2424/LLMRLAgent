import gymnasium as gym
import numpy as np
from agents.base_agent import BaseAgent
from utils.experiment_data import ExperimentData
from environment.base_environment import BaseEnvironment


def run_episode(
    experiment: ExperimentData,
    env: BaseEnvironment,
    agent: BaseAgent,
    episode_number,
    max_step=1000,
    seed=0,
    verbose=True
):
    """
    Run a single episode of the environment with the given agent.

    Args:
        experiment: The experiment data object to log results
        env: The environment to run the episode in
        agent: The agent to control the environment
        episode_number: The number of this episode
        max_step: Maximum number of steps before truncating the episode
    """
    rewards = []
    observation, _ = env.reset(seed=seed)

    if verbose:
        print(f"Starting Episode {episode_number}")

    for step in range(max_step):
        if verbose:
            print(f"{step}/{max_step}", end="\r")

        action = agent.policy(observation)
        observation, reward, terminated, truncated, _ = env.step(action)

        if step == max_step and not (terminated or truncated):
            truncated = True

        agent.update(observation, action, reward, terminated, truncated)

        rewards.append(reward)

        if terminated or truncated:
            break

    sum_rewards = float(np.sum(rewards))
    avg_rewards = float(sum_rewards / float(len(rewards)))

    if verbose:
        print(f"average reward: {avg_rewards}         ")

    experiment.log_agent_episode_length(agent, episode_number, len(rewards))
    experiment.log_agent_episode_reward_meta_stats(agent, episode_number, sum_rewards, avg_rewards)
