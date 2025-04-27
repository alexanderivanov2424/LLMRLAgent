import os

# If python version is 3.11 or lower
import sys

import gymnasium as gym
import minigrid
from gymnasium import envs

if sys.version_info < (3, 12):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from environment.env_wrappers import MiniGridEnvironment
from utils.experiment_data import ExperimentData
from utils.run_utils import run_episode

# agent imports
from agents.random_agent import RandomAgent
from agents.base_agent import BaseAgent
from agents.llm_agent import LLMAgent
from agents.llm_context_agent import LLMContextAgent

# config imports
from agents.configs.grid_config import GridConfig_1
from agents.configs.grid_context_config import GridContextConfig_1
from agents.configs.grid_memory_config import GridMemoryConfig_1

# Choose test environment here
env_name = "MiniGrid-Empty-5x5-v0"
experiment = ExperimentData.load(f"LLMRL_Agent_Comp_{env_name}")

# Create the environment using our new MiniGridEnvironment class
env = MiniGridEnvironment(env_name=env_name)

random_agent = RandomAgent(env.action_space, env.observation_space)

llm_agent = LLMAgent(
    env.get_action_descriptions(),
    env.get_valid_response(),
    env.observation_space,
    model="llama3.2",
    config=GridConfig_1(),
)

llm_context_agent = LLMContextAgent(
    env.get_action_descriptions(),
    env.get_valid_response(),
    env.observation_space,
    model="llama3.2",
    config=GridContextConfig_1(),
)

llm_memory_agent = LLMMemoryAgent(
    env.get_action_descriptions(),
    env.get_valid_response(),
    env.observation_space,
    model="llama3.2",
    config=GridMemoryConfig_1(),
)

agents = [
    #random_agent,
    #llm_agent,
    #llm_context_agent,
    llm_memory_agent,
]


for agent in agents:

    existing_epsiodes = experiment.get_agent_epsiode_count(agent.get_agent_ID())

    for episode in range(100):
        if episode < existing_epsiodes:
            continue
        run_episode(experiment, env, agent, episode, max_step=50, seed=0)

        experiment.save()

env.close()

experiment.save()
