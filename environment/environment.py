import gymnasium as gym

# Sample environments we can test agents on
AVAILABLE_ENVIRONMENTS = {
    "minigrid_empty": "MiniGrid-Empty-5x5-v0",
    "minigrid_doorkey": "MiniGrid-DoorKey-5x5-v0",
    "cartpole": "CartPole-v1",
    "mountain_car": "MountainCar-v0",
    "acrobot": "Acrobot-v1",
    "frozen_lake": "FrozenLake-v1",
    "taxi": "Taxi-v3",
    "lunar_lander": "LunarLander-v2",
    "bipedal_walker": "BipedalWalker-v3"
}

def get_available_environments():
    """
    Returns a list of available environments.
    
    Returns:
        list: List of available environments.
    """
    return list(AVAILABLE_ENVIRONMENTS.keys())

def print_all_envs_in_registry():
    print(gym.envs.registry.all())

def create_environment(env_name, seed=0):
    """
    Creates and returns a Gymnasium environment.
    
    Args:
        env_name (str): The name of the environment (key from AVAILABLE_ENVIRONMENTS).
        seed (int): Random seed for reproducibility.
        render_mode (str or None): Rendering mode (e.g., "human" for visualization).
    
    Returns:
        env: Initialized Gymnasium environment.
    """
    if env_name not in AVAILABLE_ENVIRONMENTS:
        raise ValueError(f"Environment name not found. Choose from this saved list: {list(AVAILABLE_ENVIRONMENTS.keys())}")

    env_id = AVAILABLE_ENVIRONMENTS[env_name]
    env = gym.make(env_id) #, render_mode=render_mode)
    env.reset(seed=seed)
    return env
