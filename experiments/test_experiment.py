


import gymnasium as gym
import minigrid
from gymnasium import envs

from agents.random_agent import RandomAgent
from utils.experiment import ExperimentData

# show all available envs
#print(envs.registry.keys())

env = gym.make("MiniGrid-Empty-5x5-v0")#, render_mode="human")


# demo save/load experiment
experiment = ExperimentData("test_exp")
experiment.log_meta_data("test", "some data")
experiment.save()

experiment_2 = ExperimentData.load("test_exp")
print(experiment_2.meta_data)


# demo random agent in environment

agent = RandomAgent(env.action_space, env.observation_space)


observation, info = env.reset(seed=42)
for _ in range(1000):
   
   action = agent.policy(observation)
   observation, reward, terminated, truncated, info = env.step(action)

   agent.update(observation, action, reward, terminated, truncated)

   if terminated or truncated:
      observation, info = env.reset()

env.close()