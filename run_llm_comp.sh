#!/bin/bash

# Request SLURM resources (only include if needed for a cluster job)
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH -m 32G

python3 -m experiments.LLMRL_reward_experiment