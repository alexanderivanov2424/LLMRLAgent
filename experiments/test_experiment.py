


import gymnasium as gym
import minigrid
from gymnasium import envs

from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent

from utils.experiment import ExperimentData

# show all available envs
#print(envs.registry.keys())


def run_episode(experiment : ExperimentData, env, agent : BaseAgent, episode_number, max_step=1000):
   
   rewards = []

   observation, info = env.reset()
   for step in range(max_step):
      action = agent.policy(observation)
      observation, reward, terminated, truncated, info = env.step(action)
      agent.update(observation, action, reward, terminated, truncated)

      rewards.append(reward)

      if terminated or truncated:
          break
    
      experiment.log_agent_episode_rewards(agent, episode_number, rewards)


experiment = ExperimentData("test_random_agent")

env = gym.make("MiniGrid-Empty-5x5-v0")#, render_mode="human")
agent = RandomAgent(env.action_space, env.observation_space)

env.reset(seed=0)

for episode in range(50):
   run_episode(experiment, env, agent, episode)

env.close()

experiment.save()