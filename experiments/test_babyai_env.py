import gym
import babyai  
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.random_agent import RandomAgent

from utils.run_utils_babyai import run_episode
from utils.experiment_data import ExperimentData
from environment.babyai_env import create_babyai_env, get_available_babyai_envs

# Choose test environment 
env_name = "goto_red_ball"  
experiment = ExperimentData(f"test_babyai_agent_{env_name}")

# Create BabyAI environment
env = create_babyai_env(env_name=env_name)
env.reset()

# Select agent 
agent = RandomAgent(env.action_space, env.observation_space)

# Run episodes
for episode in range(50):
    run_episode(experiment, env, agent, episode)

env.close()
experiment.save()
