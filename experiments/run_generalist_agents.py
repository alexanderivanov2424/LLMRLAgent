#!/usr/bin/env python3
import os
import subprocess
import argparse
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plots.generalist_comparison import plot_agent_comparison, compare_multiple_experiments, plot_agent_multi_env_performance

def run_generalist_experiment(agent_type, param_type, timesteps, episodes_per_env, eval_freq, verbose=False):
    """Run a single generalist experiment with specified parameters"""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "experiments" / "generalist_minigrid.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--agent_type", agent_type,
        "--param_type", param_type,
        "--timesteps", str(timesteps),
        "--episodes_per_env", str(episodes_per_env),
        "--eval_freq", str(eval_freq)
    ]
    
    env = os.environ.copy()
    
    print(f"Running experiment: {agent_type} ({param_type}), t={timesteps}, ep/env={episodes_per_env}, eval_freq={eval_freq}...")
    
    try:
        if verbose:
            result = subprocess.run(cmd, check=True, env=env)
        else:
            result = subprocess.run(cmd, check=True, env=env, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"✅ Completed {agent_type}_{param_type} experiment")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {agent_type}_{param_type}: Process returned {e.returncode}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error details: {e.stderr.decode('utf-8')}")
        return False


def run_all_experiments(timesteps=1000000, agents=None, param_types=None, episodes_per_env=5, eval_freq=10000, verbose=False):
    """Run experiments for all specified agent and parameter combinations"""
    if agents is None:
        agents = ["DQN", "PPO", "Random"]
    
    if param_types is None:
        param_types = ["standard", "generalist"]
    
    experiment_names = []
    
    for agent in agents:
        for param in param_types:
            if agent == "Random" and param != "standard":
                continue
                
            try:
                success = run_generalist_experiment(agent, param, timesteps, episodes_per_env, eval_freq, verbose)
                if success:
                    experiment_names.append(f"Generalist_{agent}_{param}")
            except Exception as e:
                print(f"Unexpected error running {agent}_{param}: {e}")
    
    return experiment_names


def main():
    parser = argparse.ArgumentParser(description="Run and compare generalist agents")
    parser.add_argument("--run", action="store_true", help="Run experiments")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--timesteps", type=int, default=300000, help="Number of timesteps for training")
    parser.add_argument("--agents", nargs="+", choices=["DQN", "PPO", "Random"], 
                        default=["DQN", "PPO", "Random"], help="Agents to run")
    parser.add_argument("--episodes_per_env", type=int, default=5, help="Episodes per training env before switching.")
    parser.add_argument("--param_types", nargs="+", choices=["standard", "optimized", "generalist"], 
                        default=["standard", "generalist"], help="Parameter types to use")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--eval_freq", type=int, default=10000, help="Evaluate the agent every N timesteps.")
    
    args = parser.parse_args()
    
    # Generate expected experiment names based on the new simpler naming convention
    experiment_names = [f"Generalist_{agent}_{param}" 
                         for agent in args.agents 
                         for param in args.param_types 
                         if not (agent == "Random" and param != "standard")]
    
    if args.run:
        experiment_names = run_all_experiments(
            timesteps=args.timesteps, 
            agents=args.agents, 
            param_types=args.param_types, 
            episodes_per_env=args.episodes_per_env,
            eval_freq=args.eval_freq,
            verbose=args.verbose
        )
    
    if args.plot and experiment_names:
        # We don't need the single experiment plot here usually
        # for exp_name in experiment_names:
        #     try:
        #         plot_agent_comparison(exp_name)
        #     except Exception as e:
        #         print(f"Error plotting {exp_name}: {e}")
        
        try:
            # Define a generic title and filename for the comparison plot
            plot_title = "Generalist Agent Learning Comparison"
            plot_filename = "generalist_agent_learning_comparison.png"
            
            print(f"Generating comparison plot for experiments: {experiment_names}")
            compare_multiple_experiments(
                experiment_names=experiment_names,
                title=plot_title, # Use generic title
                save_filename=plot_filename # Use generic filename
            )
            print(f"Comparison plot saved to plots/figures_generated/{plot_filename}")
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
        
        # <<< ADD CALLS TO THE NEW PLOTTING FUNCTION HERE >>>
        print("\nGenerating per-agent multi-environment plots...")
        processed_agents = set() # Keep track of agents we've plotted
        for exp_name in experiment_names:
            try:
                # Extract agent_id from experiment name (assuming format Generalist_AgentID)
                # This relies on the simplified name format we set earlier.
                parts = exp_name.split('_')
                if len(parts) >= 2 and parts[0] == 'Generalist':
                    agent_id = '_'.join(parts[1:]) # Handle agent IDs like DQN_generalist
                    if agent_id not in processed_agents:
                        plot_agent_multi_env_performance(exp_name, agent_id)
                        processed_agents.add(agent_id)
                else:
                     print(f"  [Warning] Could not parse agent_id from experiment name: {exp_name}")

            except Exception as e:
                print(f"  [Error] Failed to generate multi-env plot for {exp_name}: {e}")

    elif args.plot:
        print("No experiment data available or specified for plotting. Ensure experiments were run or provide names.")


if __name__ == "__main__":
    main() 