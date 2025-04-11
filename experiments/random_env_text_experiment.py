import gymnasium as gym
import minigrid
from gymnasium import envs

from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent

from utils.run_utils import run_episode
from utils.experiment_data import ExperimentData

env_id = "MiniGrid-Empty-5x5-v0"
env = gym.make(env_id)

experiment = ExperimentData(f"test_agent_{env_id}")

agent = RandomAgent(env.action_space, env.observation_space)


for episode in range(50):
    run_episode(experiment, env, agent, episode)

env.close()

experiment.save()
