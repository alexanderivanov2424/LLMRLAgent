import gymnasium as gym
import minigrid
from gymnasium import envs

# If python version is 3.11 or lower
import sys
import os

if sys.version_info < (3, 12):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.configs.agent_config import GridConfig
from environment.minigrid_environment import MiniGridEnvironment

from utils.run_utils import run_episode
from utils.experiment_data import ExperimentData

# Choose test environment here
env_name = "MiniGrid-Empty-5x5-v0"
experiment = ExperimentData(f"test_agent_{env_name}")

# Create the environment using our new MiniGridEnvironment class
env = MiniGridEnvironment(env_name=env_name)

random_agent = RandomAgent(env.action_space, env.observation_space)


agents = [
    random_agent,
    # llmagent,
]


for agent in agents:
    for episode in range(50):
        run_episode(experiment, env, agent, episode, seed=0)

env.close()

experiment.save()
