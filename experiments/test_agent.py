import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def main():
    # Create the environment
    def make_env():
        env = gym.make("MiniGrid-Empty-5x5-v0")  # no render_mode while training
        env = FlatObsWrapper(env)  # 1-D numeric Box, no mission
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecMonitor(vec_env)

    # Create the DQN model with tuned hyperparameters
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=50_000,
        learning_starts=10_000,
        batch_size=128,
        train_freq=16,
        gradient_steps=8,
        target_update_interval=800,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
        gamma=0.99,
        seed=2,
    )

    # Evaluate the agent before training
    print("Evaluating untrained agent...")
    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(),
        deterministic=True,
        n_eval_episodes=20,
    )
    print(f"Mean reward before training: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Train the agent
    print("Training agent...")
    model.learn(total_timesteps=int(300_000), log_interval=10, progress_bar=True)

    # Evaluate the agent after training
    print("Evaluating trained agent...")
    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(),
        deterministic=True,
        n_eval_episodes=20,
    )
    print(f"Mean reward after training: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Visualize the trained agent
    print("Visualizing trained agent...")
    env = make_env()
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
