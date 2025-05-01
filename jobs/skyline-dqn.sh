#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --mem=192G

# Enable python logging
export PYTHONUNBUFFERED=TRUE

# Load a CUDA module
module load cuda

# Load a Python module
module load python/3.11.0s-ixrhc3q

# Activate the virtual environment
source /users/jfinberg/scratch/LLMRLAgent/.venv/bin/activate

# Move to the project directory
cd /users/jfinberg/scratch/LLMRLAgent/experiments

# Run program
python run_hyperparameter_grid_search.py --agent PPO