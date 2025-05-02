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
from agents.llm_memory_agent import LLMMemoryAgent

# config imports
from agents.configs.grid_config import GridConfig_1
from agents.configs.grid_context_config import GridContextConfig_1
from agents.configs.grid_memory_config import GridMemoryConfig_1

def test_env(env_name):
    experiment = ExperimentData.load(f"LLM_Model_Comparison_{env_name}")

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

    llm_memory_agent_codellama = LLMMemoryAgent(
        env.get_action_descriptions(),
        env.get_valid_response(),
        env.observation_space,
        model="codellama",
        config=GridMemoryConfig_1(),
    )

    llm_memory_agent_gemma = LLMMemoryAgent(
        env.get_action_descriptions(),
        env.get_valid_response(),
        env.observation_space,
        model="gemma2",
        config=GridMemoryConfig_1(),
    )

    agents = [
        #random_agent,
        #llm_agent,
        #llm_context_agent,
        llm_memory_agent,
        llm_memory_agent_codellama,
        llm_memory_agent_gemma,
    ]


    for agent in agents:
        print("Starting Agent:", agent.get_agent_ID())
        print()
        existing_epsiodes = experiment.get_agent_epsiode_count(agent.get_agent_ID())

        if existing_epsiodes >= 50:
            continue

        for episode in range(50):
            run_episode(experiment, env, agent, episode, max_step=100, seed=0)

            experiment.save()

    env.close()

    experiment.save()


ENVIRONMENTS = {
    "Empty": "MiniGrid-Empty-5x5-v0",
    "DoorKey": "MiniGrid-DoorKey-5x5-v0",
    "GoToObj": "MiniGrid-GoToObject-6x6-N2-v0",
    "MemoryS7": "MiniGrid-MemoryS11-v0",

    # "KeyCorridor": "MiniGrid-KeyCorridorS6R3-v0",
    # "UnlockPickup": "MiniGrid-Unlock-v0",
    # "MultiRoom": "MiniGrid-MultiRoom-N4-S5-v0",
    # "LavaGap": "MiniGrid-LavaGapS5-v0",
}

for env_name in ENVIRONMENTS.values():
    print("Starting env", env_name)
    print()
    test_env(env_name)
