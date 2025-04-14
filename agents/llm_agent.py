from pydantic import BaseModel, Field, create_model
from typing import List, Callable, Any, Dict, Literal, Type, Union
from ollama import chat
from gymnasium import Space
from agents.base_agent import BaseAgent
from environment.base_environment import Action, ActionResponse
from agents.configs.agent_config import LLMRLAgentConfig


class LLMAgent(BaseAgent):
    def __init__(
        self,
        action_space: Dict[int, Action],
        valid_response: ActionResponse,
        observation_space: Space,
        model: str,
        config : LLMRLAgentConfig
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
        self.config = config
        self.model = model
        self.valid_response = valid_response

    def get_agent_ID(self):
        return "LLMAgent"

    def policy(self, observation: Dict[str, Any]) -> int:
        """
        Select an action based on the current observation.
        
        Args:
            observation: The formatted observation from the environment
            
        Returns:
            int: The selected action
        """
        # Get the available actions from the formatted observation
        available_actions = observation.get("available_actions", {})
        
        # Format the prompt using the formatted observation and action descriptions
        prompt = self.config.generate_prompt(observation, available_actions)
        
        # Get the action response from the LLM
        response = self._choose_action(prompt)
        
        return response.action

    def update(self, observation, action, reward, terminated, truncated):
        # TODO: I don't know how to implement this
        pass
        # return# super().update(observation, action, reward, terminated, truncated)

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
            format=self.valid_response.model_json_schema(),
        )

        if (
            response is None
            or response.message is None
            or response.message.content is None
        ):
            raise ValueError("Model returned invalid response")

        return self.valid_response.model_validate_json(response.message.content)

    def _choose_action(self, prompt: str) -> ActionResponse:
        """
        Choose an action based on the current situation.

        Args:
            prompt: The prompt to send to the agent

        Returns:
            ActionResponse object containing the chosen action and reasoning
        """
        # Get model response
        response = self._call_agent(prompt)

        if response is None:
            raise ValueError("Model returned invalid response")

        return response

    def _get_action_number(self, action_name: str, available_actions: Dict[int, Action]) -> int:
        """
        Convert an action name to its corresponding action number.
        
        Args:
            action_name: The name of the action to find
            available_actions: Dictionary mapping action numbers to their descriptions
            
        Returns:
            int: The action number corresponding to the given action name
            
        Raises:
            ValueError: If the action name is not found in the available actions
        """
        for action_number, action in available_actions.items():
            if action.action_name == action_name:
                return action_number
        raise ValueError(f"Action '{action_name}' not found in available actions")

    def _update_context(self, new_context: str):
        """
        Update the context history with new information.

        Args:
            new_context: New context information to add
        """
        self.context_history.append(new_context)
