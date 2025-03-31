from agents.base_agent import BaseAgent
from gymnasium import Space


class RandomAgent(BaseAgent):

  def __init__(self, action_space : Space, observation_space : Space):
    self.action_space = action_space
    self.observation_space = observation_space

  def get_agent_name(self):
    return "BaseAgent"

  def policy(self, observation):
    action = self.action_space.sample()
    return action

  def update(self, observation, action, reward, terminated, truncated):
    pass