import os
import numpy as np
import matplotlib.pyplot as plt
from utils.experiment_data import ExperimentData


PLOT_SAVE_DIR = os.path.join("./plots","figures_generated")


AGENT_NAME = {}
AGENT_NAME["RandomAgent"] = "random"
AGENT_NAME["LLMAgent_llama3.2_GridConfig_1"] = "Llama 3.2 (no mem)"

AGENT_NAME["LLMMemoryAgent_codellama_GridMemoryConfig_1"] = "CodeLlama"
AGENT_NAME["LLMMemoryAgent_gemma2_GridMemoryConfig_1"] = "Gemma 2"
AGENT_NAME["LLMMemoryAgent_llama3.2_GridMemoryConfig_1"] = "Llama 3.2"

AGENT_NAME["LLMMemoryAgent_codellama_GridMemoryConfig_2"] = "CodeLlama"
AGENT_NAME["LLMMemoryAgent_gemma2_GridMemoryConfig_2"] = "Gemma 2"
AGENT_NAME["LLMMemoryAgent_llama3.2_GridMemoryConfig_2"] = "Llama 3.2"


def generate_plot(exp_name, title, plot_name):

  experiment = ExperimentData.load(exp_name)

  for agent_ID in experiment.get_agents():
    X = []
    Y = []

    episode_count = experiment.get_agent_epsiode_count(agent_ID)
    for episode_number in range(episode_count):
      reward = experiment.get_agent_episode_average_reward(agent_ID, episode_number)

      X.append(episode_number)
      Y.append(reward)

    print(agent_ID)
    plt.plot(X, Y, alpha=.7, label=AGENT_NAME[agent_ID])

  path = os.path.join(PLOT_SAVE_DIR, plot_name + ".png")

  plt.title(title)
  plt.xlabel("Episode #")
  plt.ylabel("Average Episode Reward")
  plt.xticks(rotation=45)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path)
  plt.cla()


exp_name = "LLM_Model_Comparison_config2_MiniGrid-Empty-5x5-v0"
title = "Imperative Tone Reward Over Episodes on Empty-5x5 Environment"
plot_name = "LLM_Comp_2_MiniGrid-Empty-5x5-v0"
generate_plot(exp_name, title, plot_name)

exp_name = "LLM_Model_Comparison_config2_MiniGrid-DoorKey-5x5-v0"
title = "Imperative Tone Reward Over Episodes on DoorKey-5x5 Environment"
plot_name = "LLM_Comp_2_MiniGrid-DoorKey-5x5-v0"
generate_plot(exp_name, title, plot_name)

exp_name = "LLM_Model_Comparison_config2_MiniGrid-GoToObject-6x6-N2-v0"
title = "Imperative Tone Reward Over Episodes on GoToObject-6x6 Environment"
plot_name = "LLM_Comp_2_MiniGrid-GoToObject-6x6-N2-v0"
generate_plot(exp_name, title, plot_name)

exp_name = "LLM_Model_Comparison_config2_MiniGrid-MemoryS11-v0"
title = "Imperative Tone Reward Over Episodes on MemoryS11 Environment"
plot_name = "LLM_Comp_2_MiniGrid-MemoryS11-v0"
generate_plot(exp_name, title, plot_name)

exit()

exp_name = "SAVED_DATA/LLM_Model_Comparison_MiniGrid-DoorKey-5x5-v0"
title = "LLMRL Agent Reward Over Episodes on DoorKey Environment"
plot_name = "LLM_Comp_MiniGrid-DoorKey-5x5-v0"
generate_plot(exp_name, title, plot_name)


exp_name = "SAVED_DATA/LLM_Model_Comparison_MiniGrid-Empty-5x5-v0"
title = "LLMRL Agent Reward Over Episodes on Empty-5x5 Environment"
plot_name = "LLM_Comp_MiniGrid-Empty-5x5-v0"
generate_plot(exp_name, title, plot_name)


exp_name = "SAVED_DATA/LLM_Model_Comparison_MiniGrid-GoToObject-6x6-N2-v0"
title = "LLMRL Agent Reward Over Episodes on GoToObject-6x6 Environment"
plot_name = "LLM_Comp_MiniGrid-GoToObject-6x6-N2-v0"
generate_plot(exp_name, title, plot_name)


exp_name = "SAVED_DATA/LLM_Model_Comparison_MiniGrid-MemoryS11-v0"
title = "LLMRL Agent Reward Over Episodes on MemoryS11 Environment"
plot_name = "LLM_Comp_MiniGrid-MemoryS11-v0"
generate_plot(exp_name, title, plot_name)