import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


def main():
    # Create the environment
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    # vec_env = DummyVecEnv([lambda: env])
    vec_env = env

    # Create the DQN model with tuned hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        train_freq=16,
        gradient_steps=8,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.07,
        target_update_interval=600,
        learning_starts=1000,
        buffer_size=10000,
        batch_size=128,
        learning_rate=4e-3,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=2,
    )

    # Evaluate the agent before training
    print("Evaluating untrained agent...")
    mean_reward, std_reward = evaluate_policy(
        model,
        vec_env,
        deterministic=True,
        n_eval_episodes=20,
    )
    print(f"Mean reward before training: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Train the agent
    print("Training agent...")
    model.learn(total_timesteps=int(1.2e5), log_interval=10)

    # Evaluate the agent after training
    print("Evaluating trained agent...")
    mean_reward, std_reward = evaluate_policy(
        model,
        vec_env,
        deterministic=True,
        n_eval_episodes=20,
    )
    print(f"Mean reward after training: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Visualize the trained agent
    print("Visualizing trained agent...")
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()

    print(f"Total reward during visualization: {total_reward}")


if __name__ == "__main__":
    main()
