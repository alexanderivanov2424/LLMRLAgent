import gym  
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
    
    observation = env.reset()
    
    for _ in range(max_step):
        action = agent.policy(observation)
        
        observation, reward, done, _ = env.step(action)

        agent.update(observation, action, reward, done, False)
        rewards.append(reward)

        if done:
            break

    experiment.log_agent_episode_rewards(agent, episode_number, rewards)
