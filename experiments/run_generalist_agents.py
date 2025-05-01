#!/usr/bin/env python3
import os
import subprocess
import argparse
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plots.generalist_comparison import plot_agent_comparison, compare_multiple_experiments

def run_generalist_experiment(agent_type, param_type, timesteps=300000, verbose=False):
    """Run a single generalist experiment with specified parameters"""
    cmd = [
        "python", "experiments/generalist_minigrid.py",
        "--agent_type", agent_type,
        "--param_type", param_type,
    ]
    
    env = os.environ.copy()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = project_root
    
    print(f"Running experiment with {agent_type} agent and {param_type} parameters...")
    
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


def run_all_experiments(timesteps=300000, agents=None, param_types=None, verbose=False):
    """Run experiments for all specified agent and parameter combinations"""
    if agents is None:
        agents = ["DQN", "PPO", "Random"]
    
    if param_types is None:
        param_types = ["standard", "generalist"]
    
    experiment_names = []
    
    eval_env = "MiniGrid-Empty-8x8-v0"
    
    for agent in agents:
        for param in param_types:
            if agent == "Random" and param != "standard":
                continue
                
            try:
                success = run_generalist_experiment(agent, param, timesteps, verbose)
                if success:
                    experiment_names.append(f"Generalist_{agent}_{param}_{eval_env}")
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
    parser.add_argument("--param_types", nargs="+", choices=["standard", "optimized", "generalist"], 
                        default=["standard", "generalist"], help="Parameter types to use")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    eval_env = "MiniGrid-Empty-8x8-v0"
    experiment_names = [f"Generalist_{agent}_{param}_{eval_env}" 
                         for agent in args.agents 
                         for param in args.param_types 
                         if not (agent == "Random" and param != "standard")]
    
    if args.run:
        experiment_names = run_all_experiments(args.timesteps, args.agents, args.param_types, args.verbose)
    
    if args.plot and experiment_names:
        for exp_name in experiment_names:
            try:
                plot_agent_comparison(exp_name)
            except Exception as e:
                print(f"Error plotting {exp_name}: {e}")
        
        try:
            compare_multiple_experiments(
                experiment_names=experiment_names,
                title="Generalist Agent Comparison",
                save_filename="generalist_agent_comparison.png"
            )
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
    elif args.plot:
        print("No experiment data available for plotting")


if __name__ == "__main__":
    main() 