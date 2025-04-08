import gymnasium as gym
from agents.base_agent import BaseAgent
from utils.experiment_data import ExperimentData


def run_episode(
    experiment: ExperimentData,
    env: gym.Env,
    agent: BaseAgent,
    episode_number,
    max_step=1000,
):

    rewards = []
    observation, _ = env.reset()

    for step in range(max_step):
        action = agent.policy(observation)
        observation, reward, terminated, truncated, _ = env.step(action)

        if step == max_step and not (terminated or truncated):
            truncated = True

        agent.update(observation, action, reward, terminated, truncated)

        rewards.append(reward)

        if terminated or truncated:
            break   
    
    experiment.log_agent_episode_rewards(agent, episode_number, rewards)