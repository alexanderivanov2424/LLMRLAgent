import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

PLOT_SAVE_DIR = os.path.join("./plots", "figures_generated")


def plot_learning_curve(data_path, title):
    # Load the JSON data
    with open(data_path, "r") as f:
        data = json.load(f)

    timesteps = data["timesteps"]
    mean_rewards = data["mean_rewards"]
    std_rewards = data["std_rewards"]

    # Convert to numpy arrays for easier manipulation
    timesteps = np.array(timesteps)
    mean_rewards = np.array(mean_rewards)
    std_rewards = np.array(std_rewards)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the mean reward
    plt.plot(timesteps, mean_rewards, label="Mean Reward")

    # Add shaded region for standard deviation
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label="Standard Deviation",
    )

    # Customize the plot
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    # Save the plot
    if not os.path.exists(PLOT_SAVE_DIR):
        os.makedirs(PLOT_SAVE_DIR)

    save_path = os.path.join(PLOT_SAVE_DIR, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learning curves from JSON data")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the JSON data file"
    )
    parser.add_argument("--title", type=str, required=True, help="Title for the plot")

    args = parser.parse_args()

    plot_learning_curve(args.data, args.title)
