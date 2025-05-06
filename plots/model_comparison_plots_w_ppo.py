import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.experiment_data import ExperimentData

PLOT_SAVE_DIR = os.path.join("./plots", "figures_generated")
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

AGENT_NAME = {
    "RandomAgent": "random",
    "LLMAgent_llama3.2_GridConfig_1": "Llama 3.2 (no mem)",
}

ENV_TO_PPO_FILE = {
    "MiniGrid-Empty-5x5-v0": "Empty",
    "MiniGrid-DoorKey-5x5-v0": "DoorKey",
    "MiniGrid-GoToObject-6x6-N2-v0": "GoToObj",
    "MiniGrid-MemoryS11-v0": "MemoryS7",
}

def load_ppo_data(env_key):
    """Load the PPO learning‐curve JSON for this env (5k steps)."""
    if env_key not in ENV_TO_PPO_FILE:
        return None, None
    fname = f"experiment_data/single_agent/ppo_{ENV_TO_PPO_FILE[env_key]}_learning_curve.json"
    if not os.path.exists(fname):
        return None, None
    with open(fname, "r") as f:
        data = json.load(f)
    return data["timesteps"], data["mean_rewards"]

def generate_plot(exp_name, title, plot_name):
    experiment = ExperimentData.load(exp_name)
    max_llm_t = 0
    for agent_ID in ["LLMAgent_llama3.2_GridConfig_1", "RandomAgent"]:
        if agent_ID in experiment.get_agents():
            episodes = experiment.data["agents"][agent_ID]["episodes"]
            ep_lens = [ep["ep_len"] for ep in episodes]
            rewards = [ep["avg_reward"] for ep in episodes]
            timesteps = np.cumsum(ep_lens)
            plt.plot(timesteps, rewards, alpha=0.7, label=AGENT_NAME[agent_ID])
            if timesteps.size and timesteps[-1] > max_llm_t:
                max_llm_t = timesteps[-1]

    env_key = exp_name.split("_")[-1]
    ppo_ts, ppo_rews = load_ppo_data(env_key)
    if ppo_ts is not None:
        ppo_ts = np.array(ppo_ts)
        ppo_rews = np.array(ppo_rews)
        mask = ppo_ts <= max_llm_t
        plt.plot(ppo_ts[mask], ppo_rews[mask], alpha=0.7, linestyle="--", label="Skyline PPO")

    path = os.path.join(PLOT_SAVE_DIR, plot_name + ".png")
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Average Episode Reward")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.cla()

if __name__ == "__main__":
    cases = [
        ("LLM_Model_Comparison_config2_MiniGrid-Empty-5x5-v0",
         "Imperative Tone Reward Over Episodes on Empty-5x5 Environment",
         "LLM_Comp_2_MiniGrid-Empty-5x5-v0"),
        ("LLM_Model_Comparison_config2_MiniGrid-DoorKey-5x5-v0",
         "Imperative Tone Reward Over Episodes on DoorKey-5x5 Environment",
         "LLM_Comp_2_MiniGrid-DoorKey-5x5-v0"),
        ("LLM_Model_Comparison_config2_MiniGrid-GoToObject-6x6-N2-v0",
         "Imperative Tone Reward Over Episodes on GoToObject-6x6 Environment",
         "LLM_Comp_2_MiniGrid-GoToObject-6x6-N2-v0"),
        ("LLM_Model_Comparison_config2_MiniGrid-MemoryS11-v0",
         "Imperative Tone Reward Over Episodes on MemoryS11 Environment",
         "LLM_Comp_2_MiniGrid-MemoryS11-v0"),
    ]
    for exp_name, title, fname in cases:
        generate_plot(exp_name, title, fname)
