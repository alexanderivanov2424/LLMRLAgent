import os
import numpy as np
import matplotlib.pyplot as plt
from utils.experiment import ExperimentData


PLOT_SAVE_DIR = os.path.join("./plots","figures_generated")

# quick demo of what plot generation code could look like

experiment = ExperimentData.load("test_random_agent")

X = []
Y = []

for agent_ID in experiment.get_agents():
  episode_count = experiment.get_agent_epsiode_count(agent_ID)
  for episode_number in range(episode_count):
    rewards = experiment.get_agent_episode_rewards(agent_ID, episode_number)
    # TODO probably just save cumulative reward directly
    cumulative_reward = np.sum(rewards)

    X.append(episode_number)
    Y.append(cumulative_reward)

plt.plot(X, Y)
plt.title("Random Agent Cumulative Reward Over Episode #")

# TODO we probably want automatically generated file names for the plots too. We want it to be easier to trace which plot came from which experiment
path = os.path.join(PLOT_SAVE_DIR, experiment.exp_name + ".png")
plt.savefig(path)
