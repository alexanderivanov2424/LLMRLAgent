import gymnasium as gym
import minigrid
from gymnasium import envs

# If python version is 3.11 or lower
import sys
import os

if sys.version_info < (3, 11):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

random_agent = RandomAgent(env.action_space, env.observation_space)
llmagent = LLMAgent(
    env.action_space,
    env.observation_space,
    model="llama3.3",
    #  dict_keys(['image', 'direction', 'mission'])
    format_prompt_fn=lambda observation, action_space: """
    You are an AI agent trying to navigate a puzzle maze. Your goal is to {mission}.
    
    The maze is represented as a 7x7 grid where:
    - You are represented by the number 8 (agent)
    - Empty spaces are represented by 1
    - Walls are represented by 2
    - The goal is represented by a different number
    
    Current direction: {direction} (0=right, 1=down, 2=left, 3=up)
    
    Available actions:
    {actions}
    
    Please choose the best action to reach the goal efficiently.
    """.format(
        mission=observation.get("mission"),
        direction=observation.get("direction"),
        actions=action_space,
    ),
)

agents = [random_agent, llmagent]

env.reset(seed=0)

for agent in agents:
    for episode in range(50):
        run_episode(experiment, env, llmagent, episode)

env.close()

experiment.save()
