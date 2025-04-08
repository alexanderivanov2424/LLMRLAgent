import gym
import minigrid
from minigrid.wrappers import *
import babyai_text
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.random_agent import RandomAgent

from utils.run_utils import run_episode
from utils.experiment_data import ExperimentData

env_name = "BabyAI-GoToRedBallGrey-v0"
env_name = "BabyAI-MixedTrainLocal"
experiment = ExperimentData(f"test_babyai_agent_{env_name}")

# Create BabyAI environment
env = gym.make(env_name)
env.reset()

# Select agent 
agent = RandomAgent(env.action_space, env.observation_space)

# Run episodes
for episode in range(50):
    run_episode(experiment, env, agent, episode)

env.close()
experiment.save()
