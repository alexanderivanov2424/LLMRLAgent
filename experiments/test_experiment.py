import gymnasium as gym
import minigrid
from gymnasium import envs

from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent

from utils.experiment import ExperimentData
from environment.environment import create_environment, get_available_environments

# Show all available envs (can add more in environment/environment.py)
#print(get_available_environments())


def run_episode(
    experiment: ExperimentData,
    env: gym.Env,
    agent: BaseAgent,
    episode_number,
    max_step=1000,
):

    rewards = []
    observation, _ = env.reset()
    for _ in range(max_step):
        action = agent.policy(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        agent.update(observation, action, reward, terminated, truncated)

        rewards.append(reward)

        if terminated or truncated:
            break

        experiment.log_agent_episode_rewards(agent, episode_number, rewards)



# Choose test environment here based on dictionary keys in environment/environment.py
env_name = "minigrid_empty"
experiment = ExperimentData(f"test_random_agent_{env_name}")
env = create_environment(env_name=env_name)  
# agent = RandomAgent(env.action_space, env.observation_space)
agent = LLMAgent(
    env.action_space,
    env.observation_space,
    model="llama3.1:8b",
    #  dict_keys(['image', 'direction', 'mission'])
    format_prompt_fn=lambda observation, action_space: """
    You are an AI agent trying to navigate a puzzle maze. Your goal is to {mission}.
    
    The maze is represented as a 7x7 grid where:
    - You are represented by the number 8 (agent)
    - Empty spaces are represented by 1
    - Walls are represented by 2
    - The goal is represented by a different number
    
    Current direction: {direction} (0=right, 1=down, 2=left, 3=up)
    
    Available actions:
    {actions}
    
    Please choose the best action to reach the goal efficiently.
    """.format(
        mission=observation.get("mission"),
        direction=observation.get("direction"),
        actions=action_space,
    ),
)


env.reset(seed=0)

for episode in range(50):
    run_episode(experiment, env, agent, episode)

env.close()

experiment.save()
