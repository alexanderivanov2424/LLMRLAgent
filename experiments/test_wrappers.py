import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.env_wrappers import (
    TaxiEnvironment,
    FrozenLakeEnvironment,
    CartPoleEnvironment,
)

def test_env(env_class, name):
    print(f"\n=== Testing {name} ===")
    env = env_class()
    obs, info = env.reset(seed=42)

    print("Initial Observation Description:")
    print(obs["description"])

    if "grid_text" in obs:
        print("\nGrid View:")
        print(obs["grid_text"])

    print("\nAvailable Actions:")
    for i, a in env.get_action_descriptions().items():
        print(f"{i}: {a}")

    action = list(env.get_action_descriptions().keys())[1]
    print(f"\nTaking action {action}...\n")
    new_obs, reward, terminated, truncated, info = env.step(action)

    print("Next Observation Description:")
    print(new_obs["description"])

    if "grid_text" in new_obs:
        print("\nUpdated Grid View:")
        print(new_obs["grid_text"])

    print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

    env.close()

if __name__ == "__main__":
    test_env(TaxiEnvironment, "Taxi")
    test_env(FrozenLakeEnvironment, "Frozen Lake")
    test_env(CartPoleEnvironment, "CartPole")
