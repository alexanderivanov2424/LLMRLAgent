import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.experiment_data import ExperimentData

PLOT_SAVE_DIR = os.path.join("./plots", "figures_generated")

AGENT_NAME = {}
AGENT_NAME["RandomAgent"] = "random"
AGENT_NAME["LLMAgent_llama3.2_GridConfig_1"] = "Llama 3.2 (no mem)"

AGENT_NAME["LLMMemoryAgent_codellama_GridMemoryConfig_1"] = "CodeLlama"
AGENT_NAME["LLMMemoryAgent_gemma2_GridMemoryConfig_1"] = "Gemma 2"
AGENT_NAME["LLMMemoryAgent_llama3.2_GridMemoryConfig_1"] = "Llama 3.2"

AGENT_NAME["LLMMemoryAgent_codellama_GridMemoryConfig_2"] = "CodeLlama"
AGENT_NAME["LLMMemoryAgent_gemma2_GridMemoryConfig_2"] = "Gemma 2"
AGENT_NAME["LLMMemoryAgent_llama3.2_GridMemoryConfig_2"] = "Llama 3.2"

# Map experiment env names to DQN file prefixes
ENV_TO_DQN_FILE = {
    "MiniGrid-Empty-5x5-v0": "Empty",
    "MiniGrid-DoorKey-5x5-v0": "DoorKey",
    "MiniGrid-GoToObject-6x6-N2-v0": "GoToObj",
    "MiniGrid-MemoryS11-v0": "MemoryS7",  # Use S7 if that's the closest available
}


def load_dqn_data(env_key):
    if env_key not in ENV_TO_DQN_FILE:
        return None, None
    dqn_file = f"experiment_data/single_agent/dqn_{ENV_TO_DQN_FILE[env_key]}_learning_curve.json"
    if not os.path.exists(dqn_file):
        return None, None
    with open(dqn_file, "r") as f:
        data = json.load(f)
    episodes = [t // 50 for t in data["timesteps"]]
    return episodes, data["mean_rewards"]


def generate_plot(exp_name, title, plot_name):
    experiment = ExperimentData.load(exp_name)
    max_llm_timestep = 0
    # Only plot these two agents if present
    for agent_ID in ["LLMAgent_llama3.2_GridConfig_1", "RandomAgent"]:
        if agent_ID in experiment.get_agents():
            episodes = experiment.data["agents"][agent_ID]["episodes"]
            ep_lens = [ep["ep_len"] for ep in episodes]
            rewards = [ep["avg_reward"] for ep in episodes]
            timesteps = np.cumsum(ep_lens)
            if len(timesteps) > 0 and timesteps[-1] > max_llm_timestep:
                max_llm_timestep = timesteps[-1]
            plt.plot(timesteps, rewards, alpha=0.7, label=AGENT_NAME[agent_ID])
    # Plot DQN as a flat line at 0 up to 5000 timesteps (or max_llm_timestep, capped at 5000)
    dqn_max_timestep = min(max_llm_timestep, 5000)
    if dqn_max_timestep > 0:
        dqn_x = np.arange(0, dqn_max_timestep + 1)
        dqn_y = np.zeros_like(dqn_x)
        plt.plot(dqn_x, dqn_y, alpha=0.7, label="Skyline DQN", linestyle="--")
    path = os.path.join(PLOT_SAVE_DIR, plot_name + ".png")
    plt.title(title)
    plt.xlabel("Timestep")
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
