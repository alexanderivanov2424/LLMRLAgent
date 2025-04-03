import gymnasium as gym
import minigrid
from gymnasium import envs

from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent

from utils.run_utils import run_episode
from utils.experiment_data import ExperimentData
from environment.environment import create_environment, get_available_environments


# Choose test environment here based on dictionary keys in environment/environment.py
env_name = "minigrid_empty"
experiment = ExperimentData(f"test_agent_{env_name}")
env = create_environment(env_name=env_name)  
agent = RandomAgent(env.action_space, env.observation_space)

env.reset(seed=0)

for episode in range(50):
    run_episode(experiment, env, agent, episode)

env.close()

experiment.save()
