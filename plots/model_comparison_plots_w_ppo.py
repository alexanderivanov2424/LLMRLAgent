#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from utils.experiment_data import ExperimentData


PLOT_SAVE_DIR = os.path.join("plots", "figures_generated")


AGENT_NAME = {
    "RandomAgent": "random",
    "LLMAgent_llama3.2_GridConfig_1": "Llama 3.2 (no mem)",
}


ENV_TO_DQN_FILE = {
    "MiniGrid-Empty-5x5-v0":    "Empty",
    "MiniGrid-DoorKey-5x5-v0":   "DoorKey",
    "MiniGrid-GoToObject-6x6-N2-v0": "GoToObj",
    "MiniGrid-MemoryS11-v0":     "MemoryS7",
}

def load_ppo_data(env_key):
    """
    Loads the PPO learning curve JSON for the given env_key.
    Returns (timesteps, mean_rewards) or (None, None) if missing.
    """
    prefix = ENV_TO_DQN_FILE.get(env_key)
    if prefix is None:
        print(f"[PPO] No file prefix configured for env '{env_key}'")
        return None, None

    ppo_file = f"experiment_data/single_agent/ppo_{prefix}_learning_curve.json"
    print(f"[PPO] Loading PPO data from {ppo_file}")
    if not os.path.exists(ppo_file):
        print(f"[PPO] WARNING: file not found: {ppo_file}")
        return None, None

    with open(ppo_file, "r") as f:
        data = json.load(f)

    ts = data.get("timesteps", [])
    mr = data.get("mean_rewards", [])
    print(f"[PPO] Loaded {len(ts)} timesteps, {len(mr)} mean_rewards")
    return ts, mr

def generate_plot(exp_name, title, plot_name):
    print(f"\n=== Generating '{title}' for experiment '{exp_name}' ===")
    experiment = ExperimentData.load(exp_name)

    plt.figure()
    max_llm_timestep = 0

    # Plot LLMRL (no-mem) and RandomAgent
    for agent_id in ["LLMAgent_llama3.2_GridConfig_1", "RandomAgent"]:
        if agent_id in experiment.get_agents():
            episodes = experiment.data["agents"][agent_id]["episodes"]
            ep_lens = [ep["ep_len"] for ep in episodes]
            rewards = [ep["avg_reward"] for ep in episodes]
            timesteps = np.cumsum(ep_lens)
            if len(timesteps):
                max_llm_timestep = max(max_llm_timestep, timesteps[-1])
            print(f"[LLM] {agent_id}: final timestep = {timesteps[-1] if len(timesteps) else 'N/A'}")
            plt.plot(timesteps, rewards, alpha=0.7, label=AGENT_NAME[agent_id])

    # Overlay the PPO skyline curve
    env_key = exp_name.split("_")[-1]
    print(f"[ENV] Derived environment key = '{env_key}'")
    ppo_ts, ppo_rew = load_ppo_data(env_key)
    if ppo_ts is not None:
        ppo_ts = np.array(ppo_ts)
        ppo_rew = np.array(ppo_rew)
        mask = ppo_ts <= max_llm_timestep
        print(f"[PPO] Plotting {mask.sum()} / {len(ppo_ts)} PPO points (<= {max_llm_timestep})")
        plt.plot(
            ppo_ts[mask],
            ppo_rew[mask],
            alpha=0.7,
            linestyle="-.",
            label="Skyline PPO",
        )
    else:
        print(f"[PPO] No PPO data to plot for env '{env_key}'")

    # Finalize & save
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    out_path = os.path.join(PLOT_SAVE_DIR, plot_name + ".png")
    print(f"[SAVE] writing plot to {out_path}")
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Average Episode Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.cla()

if __name__ == "__main__":
    generate_plot(
        "LLM_Model_Comparison_config2_MiniGrid-Empty-5x5-v0",
        "LLMRL (no memory) vs Random vs PPO on Empty-5x5",
        "PPO_Comp_MiniGrid-Empty-5x5-v0",
    )
    generate_plot(
        "LLM_Model_Comparison_config2_MiniGrid-DoorKey-5x5-v0",
        "LLMRL (no memory) vs Random vs PPO on DoorKey-5x5",
        "PPO_Comp_MiniGrid-DoorKey-5x5-v0",
    )
    generate_plot(
        "LLM_Model_Comparison_config2_MiniGrid-GoToObject-6x6-N2-v0",
        "LLMRL (no memory) vs Random vs PPO on GoToObject-6x6",
        "PPO_Comp_MiniGrid-GoToObject-6x6-N2-v0",
    )
    generate_plot(
        "LLM_Model_Comparison_config2_MiniGrid-MemoryS11-v0",
        "LLMRL (no memory) vs Random vs PPO on MemoryS11",
        "PPO_Comp_MiniGrid-MemoryS11-v0",
    )
