import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.experiment_data import ExperimentData
from utils.experiment_data import KEY_AGENT

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
    
    print(f"\nAttempting to plot for experiments: {experiment_names}") # DEBUG
    found_data_to_plot = False # DEBUG Flag
    
    for exp_name in experiment_names:
        print(f"  Loading experiment: {exp_name}") # DEBUG
        try:
            experiment = ExperimentData.load(exp_name)
            if not experiment or not experiment.data.get(KEY_AGENT): # Check if load failed or no agents
                print(f"    [Warning] Failed to load or no agent data found for {exp_name}")
                continue
        except Exception as e:
            print(f"    [Error] Exception loading {exp_name}: {e}")
            continue
            
        agents_in_exp = list(experiment.get_agents())
        print(f"    Agents found in {exp_name}: {agents_in_exp}") # DEBUG
        
        for agent_id in agents_in_exp:
            print(f"      Processing agent: {agent_id}") # DEBUG
            if agents_to_include and agent_id not in agents_to_include:
                print(f"        Skipping {agent_id} (not in agents_to_include)") # DEBUG
                continue
                
            X = []
            Y = []
            
            eval_event_count = experiment.get_agent_epsiode_count(agent_id)
            print(f"        Evaluation event count for {agent_id}: {eval_event_count}") # DEBUG
            
            if eval_event_count == 0:
                print(f"        [Warning] No evaluation events logged for {agent_id} in {exp_name}")
                continue
                
            for eval_event_num in range(eval_event_count):
                # Get the dictionary of rewards for this event
                env_rewards_dict = experiment.get_agent_multi_env_eval_rewards(agent_id, eval_event_num)
                
                if env_rewards_dict is None:
                    print(f"          [Warning] Missing multi-env reward data for {agent_id}, event {eval_event_num}")
                    # Handle missing data: skip this point or plot a default value (e.g., 0 or None)
                    # Let's skip for now
                    continue 
                
                # Calculate the average reward across environments for this event
                if env_rewards_dict:
                    avg_reward = np.mean(list(env_rewards_dict.values()))
                else:
                    avg_reward = 0 # Or handle as appropriate if dict is empty
                    
                X.append(eval_event_num)
                Y.append(avg_reward)
            
            print(f"        Avg Reward Data points for {agent_id}: X={X[:5]}..., Y={Y[:5]}...") # DEBUG (show first 5)
            
            if X and Y: # Only plot if we have data
                label = f"{agent_id}"
                plt.plot(X, Y, label=label)
                found_data_to_plot = True # Mark that we found something
            else:
                print(f"        [Warning] No data points (X, Y) generated for {agent_id} in {exp_name}")
    
    plt.title(title or "Agent Comparison Across Experiments")
    plt.xlabel("Evaluation Event Number") # Changed X label for clarity
    plt.ylabel("Mean Reward (Avg over Training Envs)") # Changed Y label
    
    if not found_data_to_plot:
        print("\n[Error] No data was found to plot across all specified experiments and agents.") # DEBUG
        # Optionally add placeholder text to the blank plot
        plt.text(0.5, 0.5, 'No data found to plot', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    else:
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    filename = save_filename or "multi_experiment_comparison.png"
    path = os.path.join(PLOT_SAVE_DIR, filename)
    plt.savefig(path)
    plt.close()
    
    print(f"Plot saved to {path}")

def plot_agent_multi_env_performance(experiment_name, agent_id, title=None, save_filename=None):
    """
    Generate a plot showing a single agent's performance on each evaluated 
    environment over time.
    
    Args:
        experiment_name: Name of the experiment to load (e.g., "Generalist_DQN_generalist")
        agent_id: The specific agent ID within the experiment (e.g., "DQN_generalist")
        title: Custom title for the plot.
        save_filename: Custom filename for saving.
    """
    print(f"\nGenerating multi-env performance plot for agent '{agent_id}' from experiment '{experiment_name}'")
    try:
        experiment = ExperimentData.load(experiment_name)
        if not experiment or not experiment.data.get(KEY_AGENT) or agent_id not in experiment.get_agents():
            print(f"  [Error] Could not load experiment or find agent '{agent_id}' in '{experiment_name}'")
            return
    except Exception as e:
        print(f"  [Error] Exception loading {experiment_name}: {e}")
        return

    eval_event_count = experiment.get_agent_epsiode_count(agent_id)
    print(f"  Found {eval_event_count} evaluation events.")
    if eval_event_count == 0:
        print("  [Warning] No evaluation events to plot.")
        return

    # Dictionary to hold data series for each environment {env_id: {'X': [], 'Y': []}}
    env_data = {}
    
    for eval_event_num in range(eval_event_count):
        env_rewards_dict = experiment.get_agent_multi_env_eval_rewards(agent_id, eval_event_num)
        
        if env_rewards_dict is None:
            # Data missing for this event, skip
            continue 
            
        for env_id, reward in env_rewards_dict.items():
            if env_id not in env_data:
                env_data[env_id] = {'X': [], 'Y': []}
            env_data[env_id]['X'].append(eval_event_num)
            env_data[env_id]['Y'].append(reward)

    if not env_data:
        print("  [Error] No environment reward data extracted.")
        return

    plt.figure(figsize=(12, 7))
    
    found_lines = False
    for env_id, data in env_data.items():
        if data['X'] and data['Y']:
            # Extract a shorter label if possible (e.g., "Empty-5x5")
            short_label = '-'.join(env_id.split('-')[1:-1]) or env_id
            plt.plot(data['X'], data['Y'], label=short_label)
            print(f"    Plotting data for env: {short_label}")
            found_lines = True
        else:
             print(f"    [Warning] No data points found for env: {env_id}")


    plot_title = title or f"Performance of {agent_id} Across Environments"
    plt.title(plot_title)
    plt.xlabel("Evaluation Event Number")
    plt.ylabel("Mean Reward")
    
    if not found_lines:
        print("  [Error] No lines were plotted.")
        plt.text(0.5, 0.5, 'No data found to plot', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    else:
         # Place legend outside the plot
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to prevent legend overlap
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for legend

    # Save the plot
    default_filename = f"{experiment_name}_{agent_id}_multi_env_perf.png"
    filename = save_filename or default_filename
    path = os.path.join(PLOT_SAVE_DIR, filename)
    plt.savefig(path)
    plt.close()
    
    print(f"  Plot saved to {path}")

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