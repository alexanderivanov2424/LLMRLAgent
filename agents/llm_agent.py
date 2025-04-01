from pydantic import BaseModel, Field
from typing import List
from ollama import chat

from agents.base_agent import BaseAgent


class Action(BaseModel):
    """Possible actions the agent can take"""

    action_name: str = Field(description="The name of the action to take")
    action_description: str = Field(description="A description of the action to take")

    def __str__(self) -> str:
        return f"{self.action_name}: {self.action_description}"


class ActionResponse(BaseModel):
    """Structure for the model's action choice and reasoning"""

    action: str = Field(description="The chosen action from the available actions")
    reasoning: str = Field(description="The reasoning behind choosing this action")


# TODO move prompts out into a config file so we can do precise versioning.
# Doesn't matter if it is still string constants in code it just shouldn't be part of the LLMRL agent itself
# we want to be able to create multiple of these agents with different sets of prompts and save all the different prompts we have tried


class LLMAgent(BaseAgent):
    def __init__(
        self,
        model: str,
        base_prompt: str,
        action_space: List[Action],
    ):
        """
        Initialize the RL agent.

        Args:
            model: A LangChain language model
            base_prompt: The initial prompt/instructions for the agent
            action_space: List of possible actions the agent can take
        """
        self.action_space = action_space
        self.context_history = []
        self.base_prompt = base_prompt
        self.model = model

    def get_agent_name(self):
        return "LLMAgent"

    def policy(self, observation):
        # TODO: I don't know how to implement this
        return super().policy(observation)

    def update(self, observation, action, reward, terminated, truncated):
        # TODO: I don't know how to implement this
        return super().update(observation, action, reward, terminated, truncated)

    def _format_prompt(self, situation_description: str) -> str:
        # Setup the prompt template
        return """
        {base_prompt}

        Available actions: {actions}

        Context history:
        {context}

        Current situation: {situation}

        Choose an action and provide reasoning:""".format(
            base_prompt=self.base_prompt,
            actions="\n".join(str(action) for action in self.action_space),
            context="\n".join(self.context_history),
            situation=situation_description,
        )

    def _call_agent(self, prompt: str) -> ActionResponse:
        """Call the agent with the given prompt"""
        response = chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
            format=ActionResponse.model_json_schema(),
        )

        if (
            response is None
            or response.message is None
            or response.message.content is None
        ):
            raise ValueError("Model returned invalid response")

        return ActionResponse.model_validate_json(response.message.content)

    def _choose_action(self, situation_description: str) -> ActionResponse:
        """
        Choose an action based on the current situation.

        Args:
            situation_description: Description of the current state/situation

        Returns:
            ActionChoice object containing the chosen action and reasoning
        """
        # Get model response
        response = self._call_agent(self._format_prompt(situation_description))

        if response is None or not isinstance(response, ActionResponse):
            raise ValueError("Model returned invalid response")

        return response

    def _update_context(self, new_context: str):
        """
        Update the context history with new information.

        Args:
            new_context: New context information to add
        """
        self.context_history.append(new_context)
