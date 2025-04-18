from agents.configs.base_config import LLMRLAgentConfig

class GridContextConfig_1(LLMRLAgentConfig):

  full_prompt = """
You are an AI agent navigating a MiniGrid environment. Your goal is to reach the green goal square marked as 'G'.

Current Grid Layout:
{{state}}

List of revious Grid Layouts, Actions, and Rewards:
[
{{context}}
]

Legend:
- █ = Wall (cannot pass through)
- · = Empty floor (can move onto)
- G = Goal (your target destination)
- ▶/▼/◀/▲ = Your current position and direction
  ▶ = facing right
  ▼ = facing down
  ◀ = facing left
  ▲ = facing up

Available Actions:
{{action_list}}

Navigation Strategy:
1. First, determine if you need to turn to face the goal
2. Once facing the right direction, move forward if the path is clear
3. If blocked, plan a route around obstacles

Choose the most efficient action to reach the goal. Remember:
- You can only move in the direction you're facing
- You must turn left/right to change direction
- Moving forward advances one square in your current direction

Please return your chosen action number and detailed reasoning in this format:
{response_type}
    """

  response_full = """
{{
    "reasoning": "Okay, lets think about the best plan of attack... [reason your way to an action]",
    "action": <number>,
    "rationalization": "I chose this action because... [explain why this action is the best choice]",
}}
    """

  response_action_only = """
{{
    "action": <number>
}}
    """

  history_context_line = """observation: {}
    action: {}
    reward: {}

    """

  def __init__(self, with_reasoning=False):
    self.prompt = __class__.full_prompt.format(response_type = __class__.response_full if with_reasoning else __class__.response_action_only)
    self.context = ""
  
  def generate_prompt(self, observation, available_actions):
    action_list = "\n".join(
            [
                f"{key}: {action.action_name}: {action.action_description}"
                for key, action in available_actions.items()
            ]
        )
    
    return self.prompt.format(
        # mission=observation.get("mission"),
        state=observation.get("grid_text"),
        action_list=action_list,
        context=self.context
    )

  def update_context(observation, action, reward):
    self.context += __class__.history_context_line.format(observation=observation, action=str(action), reward=str(reward))

  def clear_context():
    self.context = ""
