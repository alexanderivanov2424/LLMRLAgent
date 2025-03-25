import gymnasium as gym
import minigrid
from gymnasium import envs


print(envs.registry.keys())

env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")

print(env.action_space)

def policy(obs):
   return env.action_space.sample()

observation, info = env.reset(seed=42)
for _ in range(1000):
   action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()