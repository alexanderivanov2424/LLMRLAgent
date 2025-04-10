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
from agents.llm_agent import LLMAgent
from environment.minigrid_environment import MiniGridEnvironment

from utils.run_utils import run_episode
from utils.experiment_data import ExperimentData

# Choose test environment here
env_name = "MiniGrid-Empty-5x5-v0"
experiment = ExperimentData(f"test_agent_{env_name}")

# Create the environment using our new MiniGridEnvironment class
env = MiniGridEnvironment(env_name=env_name)

random_agent = RandomAgent(env.action_space, env.observation_space)
llmagent = LLMAgent(
    env.get_action_descriptions(),
    env.get_valid_response(),
    env.observation_space,
    model="llama3.2",
    format_prompt_fn=lambda observation, available_actions: """
You are an AI agent navigating a MiniGrid environment. Your goal is to reach the green goal square marked as 'G'.

Current Grid Layout:
{state}

Legend:
- █ = Wall (cannot pass through)
- · = Empty floor (can move onto)
- G = Goal (your target destination)
- ▶/▼/◀/▲ = Your current position and direction
  ▶ = facing right
  ▼ = facing down
  ◀ = facing left
  ▲ = facing up

Available Actions:
{action_list}

Navigation Strategy:
1. First, determine if you need to turn to face the goal
2. Once facing the right direction, move forward if the path is clear
3. If blocked, plan a route around obstacles

Choose the most efficient action to reach the goal. Remember:
- You can only move in the direction you're facing
- You must turn left/right to change direction
- Moving forward advances one square in your current direction

Please return your chosen action number and detailed reasoning in this format:
{{
    "action": <number>,
    "reasoning": "I chose this action because... [explain how it helps reach the goal]"
}}
    """.format(
        mission=observation.get("mission"),
        state=observation.get("grid_text"),
        action_list="\n".join(
            [
                f"{key}: {action.action_name}: {action.action_description}"
                for key, action in available_actions.items()
            ]
        ),
    ),
)

agents = [
    random_agent,
    # llmagent,
]


for agent in agents:
    for episode in range(50):
        run_episode(experiment, env, agent, episode, seed=0)

env.close()

experiment.save()
