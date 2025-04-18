import gymnasium as gym
import numpy as np
import time

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

    agent_policy_time = 0
    agent_update_time = 0

    for step in range(max_step):
        if verbose:
            print(f"{step}/{max_step}", end="\r")

        start_time = time.time()
        action = agent.policy(observation)
        agent_policy_time += float(time.time() - start_time)

        observation, reward, terminated, truncated, _ = env.step(action)

        if step == max_step and not (terminated or truncated):
            truncated = True

        start_time = time.time()
        agent.update(observation, action, reward, terminated, truncated)
        agent_update_time += float(time.time() - start_time)

        rewards.append(reward)

        if terminated or truncated:
            break

    sum_rewards = float(np.sum(rewards))
    avg_rewards = float(sum_rewards / float(len(rewards)))

    if verbose:
        print(f"average reward: {avg_rewards}         ")

    experiment.log_agent_episode_length(agent, episode_number, len(rewards))
    experiment.log_agent_episode_reward_meta_stats(agent, episode_number, sum_rewards, avg_rewards)
    experiment.log_agent_episode_policy_time(agent, episode_number, agent_policy_time)
    experiment.log_agent_episode_update_time(agent, episode_number, agent_update_time)