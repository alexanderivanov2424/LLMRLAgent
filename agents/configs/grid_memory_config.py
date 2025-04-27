from agents.configs.base_config import LLMRLAgentConfig


class GridMemoryConfig_1(LLMRLAgentConfig):

    full_prompt = """
You are an AI agent navigating a MiniGrid environment. Your goal is to reach the green goal square marked as 'G'.

Current Grid Layout:
{{state}}

Context sumarizing your past experience and learnings:
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

    memory_update_prompt = """You are producing a block of text to inform an inteligent agent interacting with its environment.
Summarize the following trajectory, a list of observations, actions, and rewards, and combine it with the previous summary. 
Present the summary as a list of key rules to follow and limit the response to {word_limit} words or less.

Previous Memory:
{previous_memory}

Trajectory:
{trajectory}
"""

    def __init__(self, with_reasoning=False, memory_word_limit=500):
        self.prompt = self.full_prompt.format(
            response_type=(
                self.response_full if with_reasoning else self.response_action_only
            )
        )
        self.context = ""
        self.memory_word_limit = memory_word_limit

    def generate_prompt(self, observation, available_actions):
        action_list = "\n".join(
            [
                f"{key}: {action.action_name}: {action.action_description}"
                for key, action in available_actions.items()
            ]
        )

        return self.prompt.format(
            state=observation.get("grid_text"),
            action_list=action_list,
            context=self.context,
        )
    
    def generate_memory_update_prompt(self, history):
      trajectory_text = ""
      for i, step_context in enumerate(history):
        trajectory_text += f"step: {i}\nobservation: {step_context['observation']['grid_text']}\naction:{step_context['action']}\nreward{step_context['reward']}\n\n"

      return self.memory_update_prompt.format(
            word_limit=self.memory_word_limit,
            previous_memory=self.context,
            trajectory=trajectory_text
        )

    def update_context(self, new_memory):
        self.context = new_memory

    def clear_context(self):
        self.context = ""
