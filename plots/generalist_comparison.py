import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.experiment_data import ExperimentData

PLOT_SAVE_DIR = os.path.join("./plots", "figures_generated")
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

def plot_agent_comparison(experiment_name, title=None, save_filename=None):
    """
    Generate a plot comparing different agents from the same experiment.
    
    Args:
        experiment_name: Name of the experiment to load
        title: Custom title for the plot (default: uses experiment name)
        save_filename: Custom filename for saving (default: uses experiment name)
    """
    experiment = ExperimentData.load(experiment_name)
    
    plt.figure(figsize=(10, 6))
    
    # Track average rewards for final stats
    avg_rewards = {}
    
    for agent_id in experiment.get_agents():
        X = []
        Y = []
        
        total_reward_sum = 0
        episode_count = experiment.get_agent_epsiode_count(agent_id)
        
        for episode_number in range(episode_count):
            reward = experiment.get_agent_episode_sum_reward(agent_id, episode_number)
            total_reward_sum += reward
            
            X.append(episode_number)
            Y.append(reward)
        
        plt.plot(X, Y, label=agent_id)
        avg_reward = total_reward_sum / episode_count if episode_count > 0 else 0
        avg_rewards[agent_id] = avg_reward
        print(f"Average Reward Across Episodes ({agent_id}): {avg_reward:.4f}")
    
    plt.title(title or f"Agent Performance: {experiment_name}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    filename = save_filename or f"{experiment_name}.png"
    path = os.path.join(PLOT_SAVE_DIR, filename)
    plt.savefig(path)
    plt.close()
    
    print(f"Plot saved to {path}")
    return avg_rewards

def compare_multiple_experiments(experiment_names, agents_to_include=None, title=None, save_filename=None):
    """
    Generate a comparison plot of specific agents across multiple experiments.
    
    Args:
        experiment_names: List of experiment names to load
        agents_to_include: List of agent IDs to include (if None, include all)
        title: Custom title for the plot
        save_filename: Custom filename for saving
    """
    plt.figure(figsize=(12, 7))
    
    for exp_name in experiment_names:
        experiment = ExperimentData.load(exp_name)
        
        for agent_id in experiment.get_agents():
            if agents_to_include and agent_id not in agents_to_include:
                continue
                
            X = []
            Y = []
            
            episode_count = experiment.get_agent_epsiode_count(agent_id)
            for episode_number in range(episode_count):
                reward = experiment.get_agent_episode_sum_reward(agent_id, episode_number)
                X.append(episode_number)
                Y.append(reward)
            
            label = f"{agent_id}"
            plt.plot(X, Y, label=label)
    
    plt.title(title or "Agent Comparison Across Experiments")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    filename = save_filename or "multi_experiment_comparison.png"
    path = os.path.join(PLOT_SAVE_DIR, filename)
    plt.savefig(path)
    plt.close()
    
    print(f"Plot saved to {path}")

if __name__ == "__main__":
    # Example usage:
    
    # Single experiment plot (replace with your actual experiment name)
    plot_agent_comparison("Generalist_DQN_standard_MiniGrid-Empty-8x8-v0", 
                          title="Random Agent Cumulative Reward Over Episode #")
    
    # Compare agents from multiple experiments (uncomment and customize)
    # compare_multiple_experiments(
    #     experiment_names=[
    #         "Generalist_DQN_standard_MiniGrid-Empty-8x8-v0",
    #         "Generalist_PPO_standard_MiniGrid-Empty-8x8-v0",
    #         "Generalist_DQN_generalist_MiniGrid-Empty-8x8-v0"
    #     ],
    #     title="Generalist Agent Comparison"
    # ) 