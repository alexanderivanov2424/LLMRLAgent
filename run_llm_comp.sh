#!/bin/bash

# Request SLURM resources (only include if needed for a cluster job)
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mem=16G

module load ollama
ollama serve &>- &

python3 -m experiments.LLMRL_reward_experiment
