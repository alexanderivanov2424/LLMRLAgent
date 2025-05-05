# -------------PPO-------------
# Standard PPO (Stable Baselines3 defaults)
ppo_standard_hyperparams = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
}

# Optimized PPO: TODO
ppo_optimized_hyperparams = {
        "learning_rate": 1e-4,
        "n_steps": 1024,
        "batch_size": 32,
        "n_epochs": 5,
        "gamma": 0.995,
        "gae_lambda": 0.9,
        "clip_range": 0.1,
        "ent_coef": 0.001,
}

# Generalist PPO: Modified for multi-environment training
ppo_generalist_hyperparams = {
    "learning_rate": 1e-4,
    "n_steps": 2048,           # Longer steps to collect diverse experiences
    "batch_size": 64,          # Standard batch size
    "n_epochs": 8,             # More epochs for better generalization
    "gamma": 0.99,             # Standard discount factor
    "gae_lambda": 0.95,        # Standard GAE lambda
    "clip_range": 0.2,         # Standard clip range
    "ent_coef": 0.01,          # Higher entropy to encourage exploration across different environments
    "vf_coef": 0.5,            # Standard value function coefficient
    "max_grad_norm": 0.5,      # Standard gradient clipping
}

# -------------DQN-------------
# Standard DQN hyperparameters (from Stable Baselines3 defaults)
dqn_standard_hyperparams = {
    "learning_rate": 1e-4,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 32,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1000,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
}

# Hyperparams from "Learning to Learn with Active Adaptive Perception -
# TODO: check this
# https://arxiv.org/abs/1811.12323
dqn_optimized_hyperparams = {
    "learning_rate": 0.0001,  # Lower learning rate for stability
    "buffer_size": 10000,  # Smaller buffer for grid environments
    "learning_starts": 1000,  # Standard warmup period
    "batch_size": 128,  # Larger batch size for better gradient estimates
    "tau": 1.0,  # Full target network update
    "gamma": 0.99,  # Standard discount factor
    "train_freq": 1,  # Train every step
    "gradient_steps": 1,  # Single gradient step per update
    "target_update_interval": 100,  # More frequent target updates
    "exploration_fraction": 0.2,  # Moderate exploration
    "exploration_initial_eps": 1.0,  # Start with full exploration
    "exploration_final_eps": 0.01,  # Lower final exploration rate
}

# Generalist hyperparameters (based on optimized but with larger buffer)
dqn_generalist_hyperparams = {
    "learning_rate": 0.0001,  
    "buffer_size": 500000,  # Larger buffer to store experiences from multiple environments
    "learning_starts": 50000,  # Longer warmup to collect diverse experiences
    "batch_size": 128,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "target_update_interval": 100,
    "exploration_fraction": 0.3,  # More exploration for diverse environments
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
}


def get_hyperparams(agent_type: str, param_type: str):
    if agent_type == "DQN":
        if param_type == "standard":
            return dqn_standard_hyperparams
        elif param_type == "optimized":
            return dqn_optimized_hyperparams
        elif param_type == "generalist":
            return dqn_generalist_hyperparams
    elif agent_type == "PPO":
        if param_type == "standard":
            return ppo_standard_hyperparams
        elif param_type == "optimized":
            return ppo_optimized_hyperparams
        elif param_type == "generalist":
            return ppo_generalist_hyperparams
    return None