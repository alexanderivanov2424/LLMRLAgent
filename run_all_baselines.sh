#!/bin/bash

# Request SLURM resources (only include if needed for a cluster job)
# #SBATCH -p gpu --gres=gpu:1
# #SBATCH -n 4
# #SBATCH -t 24:00:00
# #SBATCH --mem=192G

# Move to your experiments directory
cd experiments

# List of environments
ENVS=("Empty" "DoorKey" "GoToObj" "MemoryS7" "KeyCorridor" "UnlockPickup" "MultiRoom" "LavaGap")

# List of agents
AGENTS=("DQN" "PPO")

# Number of timesteps to train
TIMESTEPS=1500000

# Results folder
RESULTS_DIR="../experiment_data/baseline_long"
mkdir -p $RESULTS_DIR

# Run experiments
for agent in "${AGENTS[@]}"
do
    for env in "${ENVS[@]}"
    do
        echo "Running $agent on $env"
        python run_baseline.py --agent $agent --env $env --timesteps $TIMESTEPS --results_dir $RESULTS_DIR
    done
done

# After all runs, generate plots
echo "Plotting baseline results..."
python plot_baseline_results.py --results_dir $RESULTS_DIR

echo "Done!"