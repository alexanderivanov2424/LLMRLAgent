from pydantic import BaseModel, Field, create_model
from typing import List, Callable, Any, Literal
from typing_extensions import Annotated
from ollama import chat
from gymnasium import Space
from agents.base_agent import BaseAgent


class Action(BaseModel):
    """Possible actions the agent can take"""

    action_name: str = Field(description="The name of the action to take")
    action_description: str = Field(description="A description of the action to take")

    def __str__(self) -> str:
        return f"{self.action_name}: {self.action_description}"


class LLMAgent(BaseAgent):
    def __init__(
        self,
        action_space: Space,
        observation_space: Space,
        model: str,
        format_prompt_fn: Callable[[Any, Space], str],
    ):
        """
        Initialize the RL agent.

        Args:
            model: A LangChain language model
            base_prompt: The initial prompt/instructions for the agent
            action_space: List of possible actions the agent can take
        """
        super().__init__(action_space, observation_space)

        self.context_history = []
        self.format_prompt_fn = format_prompt_fn
        self.model = model

        # Create dynamic ActionResponse model
        allowed_actions = [str(action) for action in self.action_space]
        ActionLiteral = Literal[tuple(allowed_actions)]  # type: ignore

        self.ActionResponse = create_model(
            "ActionResponse",
            action=(Annotated[ActionLiteral, Field(description="The chosen action from the available actions")]),  # type: ignore
            reasoning=(
                str,
                Field(description="The reasoning behind choosing this action"),
            ),
        )

    def get_agent_name(self):
        return "LLMAgent"

    def policy(self, observation):
        prompt = self.format_prompt_fn(observation, self.action_space)
        action = self._choose_action(prompt)
        return action

    def update(self, observation, action, reward, terminated, truncated):
        # TODO: I don't know how to implement this
        return super().update(observation, action, reward, terminated, truncated)

    def _call_agent(self, prompt: str) -> Any:
        """Call the agent with the given prompt"""
        response = chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
            format=self.ActionResponse.model_json_schema(),
        )

        if (
            response is None
            or response.message is None
            or response.message.content is None
        ):
            raise ValueError("Model returned invalid response")

        return self.ActionResponse.model_validate_json(response.message.content)

    def _choose_action(self, prompt: str) -> Any:
        """
        Choose an action based on the current situation.

        Args:
            prompt: The prompt to send to the agent

        Returns:
            ActionChoice object containing the chosen action and reasoning
        """
        # Get model response
        response = self._call_agent(prompt)

        if response is None or not isinstance(response, self.ActionResponse):
            raise ValueError("Model returned invalid response")

        return response

    def _update_context(self, new_context: str):
        """
        Update the context history with new information.

        Args:
            new_context: New context information to add
        """
        self.context_history.append(new_context)
