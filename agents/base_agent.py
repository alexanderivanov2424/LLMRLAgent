class BaseAgent:

  def __init__(self, action_space, observation_space):
    self.action_space = action_space
    self.observation_space = observation_space

  def get_agent_name(self):
    return "BaseAgent"

  def policy(self, observation):
    # given the observation return the action the agent selects
    pass

  def update(self, observation, action, reward, terminated, truncated):
    # after taking an action update the agent with the full tuple of the last time step
    pass
