import os
import numpy as np
import matplotlib.pyplot as plt
from utils.experiment_data import ExperimentData


PLOT_SAVE_DIR = os.path.join("./plots","figures_generated")

# quick demo of what plot generation code could look like

experiment = ExperimentData.load("LLMRL_Agent_Comp_MiniGrid-Empty-5x5-v0")
experiment.exp_name = "LLMRL_Memory_Agent_MiniGrid-Empty-5x5-v0"

for agent_ID in experiment.get_agents():
  X = []
  Y = []

  total_reward_sum = 0

  episode_count = experiment.get_agent_epsiode_count(agent_ID)
  for episode_number in range(episode_count):
    reward = experiment.get_agent_episode_sum_reward(agent_ID, episode_number)

    total_reward_sum += reward

    X.append(episode_number)
    Y.append(reward)

  plt.plot(X, Y, label=agent_ID)
  print("Average Reward Across Episodes", agent_ID, total_reward_sum / episode_count)
plt.title("Random Agent Cumulative Reward Over Episode #")
plt.legend()

# TODO we probably want automatically generated file names for the plots too. We want it to be easier to trace which plot came from which experiment
path = os.path.join(PLOT_SAVE_DIR, experiment.exp_name + ".png")
plt.savefig(path)
