import gym
import babyai  

AVAILABLE_BABYAI_ENVIRONMENTS = {
    "goto_red_ball": "BabyAI-GoToRedBall-v0",
    "pickup_loc": "BabyAI-PickupLoc-v0",
    "put_next_local": "BabyAI-PutNextLocal-v0",
    "goto_local": "BabyAI-GoToLocal-v0",
    "unlock": "BabyAI-Unlock-v0",
    "open": "BabyAI-Open-v0",
    "goto_obj_these": "BabyAI-GoToObjInRoom-v0",
    "goto_seq": "BabyAI-GoToSeq-v0",
    "boss_level": "BabyAI-BossLevel-v0"
}

def get_available_babyai_envs():
    """
    Returns a list of available BabyAI environments.
    """
    return list(AVAILABLE_BABYAI_ENVIRONMENTS.keys())

def print_all_envs_in_registry():
    """
    Prints all registered environments.
    """
    print([env_spec.id for env_spec in gym.envs.registry.values()])

def create_babyai_env(env_name, seed=0):
    if env_name not in AVAILABLE_BABYAI_ENVIRONMENTS:
        raise ValueError(f"Invalid BabyAI environment. Choose from: {list(AVAILABLE_BABYAI_ENVIRONMENTS.keys())}")

    env_id = AVAILABLE_BABYAI_ENVIRONMENTS[env_name]

    env = gym.make(env_id)  
    env.seed(seed)  

    return env


