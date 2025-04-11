import gymnasium as gym
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

    for step in range(max_step):
        action = agent.policy(observation)
        observation, reward, terminated, truncated, _ = env.step(action)

        if step == max_step and not (terminated or truncated):
            truncated = True

        agent.update(observation, action, reward, terminated, truncated)

        rewards.append(reward)

        if terminated or truncated:
            break

    sum_rewards = np.sum(rewards)

    experiment.log_agent_episode_length(agent, episode_number, len(rewards))
    experiment.log_agent_episode_reward_meta_stats(agent, episode_number, sum_rewards, sum_rewards / float(len(rewards)))
