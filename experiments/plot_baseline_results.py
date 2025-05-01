import os
import json
import matplotlib.pyplot as plt

AGENTS = ["DQN", "PPO"]
ENVS = [
    "Empty",
    "DoorKey",
    "MemoryS7",
    "KeyCorridor",
    "UnlockPickup",
    "MultiRoom",
    "LavaGap"
]

def plot_baselines(agents, envs):
    plt.figure(figsize=(12, 6))

    for agent in agents:
        means = []
        stds = []
        for env in envs:
            filename = f"../experiment_data/baseline/{agent.lower()}_{env}_baseline.json"
            if not os.path.exists(filename):
                print(f"Missing: {filename}")
                means.append(0)
                stds.append(0)
                continue

            with open(filename, "r") as f:
                result = json.load(f)

            means.append(result.get("mean_reward", 0.0))
            stds.append(result.get("std_reward", 0.0))

        plt.errorbar(envs, means, yerr=stds, label=agent, capsize=4, marker='o')

    plt.title("Baseline Performance on MiniGrid Environments")
    plt.xlabel("Environment")
    plt.ylabel("Mean Reward")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../experiment_data/baseline/baseline_results.png")
    plt.show()

if __name__ == "__main__":
    plot_baselines(AGENTS, ENVS)
