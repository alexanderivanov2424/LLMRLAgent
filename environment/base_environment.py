from abc import ABC, abstractmethod
from gymnasium import Space, Env
from typing import Any, Dict, Tuple, Optional, SupportsFloat
from pydantic import BaseModel, Field

class Action(BaseModel):
    """Possible actions the agent can take"""

    action_name: str = Field(description="The name of the action to take")
    action_description: str = Field(description="A description of the action to take")

    def __str__(self) -> str:
        return f"{self.action_name}: {self.action_description}"
    
class ActionResponse(BaseModel):
    """Base class for action responses"""
    action: int = Field(description="The action number to take")
    reasoning: str = Field(description="The reasoning behind choosing this action")

class BaseEnvironment(Env):
    """
    Base class for all environments in the RL framework.
    This class defines the interface that all environments must implement.
    """
    
    action_space: Space
    observation_space: Space
    
    @abstractmethod
    def __init__(self):
        """Initialize the environment."""
        pass
    
    @abstractmethod
    def get_action_space(self) -> Space:
        """
        Returns the action space of the environment.
        
        Returns:
            Space: The action space defining valid actions in the environment.
        """
        pass
    
    @abstractmethod
    def get_observation_space(self) -> Space:
        """
        Returns the observation space of the environment.
        
        Returns:
            Space: The observation space defining valid observations in the environment.
        """
        pass
    
    @abstractmethod
    def get_action_descriptions(self) -> Dict[int, Action]:
        """
        Returns a dictionary describing all available actions.
        
        Returns:
            Dict[int, Action]: Dictionary mapping action numbers to their names and descriptions.
            Each action should have a 'name' and 'description' key.
            Example:
            {
                0: Action(name="action1", description="Description of action1"),
                1: Action(name="action2", description="Description of action2")
            }
        """
        pass

    @abstractmethod
    def get_valid_response(self) -> BaseModel:
        """
        Returns the valid response for the environment.
        """
        pass
    
    @abstractmethod
    def format_observation(self, observation: Any) -> Dict[str, Any]:
        """
        Formats the raw observation into a structured format that can be used by agents.
        
        Args:
            observation: The raw observation from the environment.
            
        Returns:
            Dict[str, Any]: A dictionary containing the formatted observation with
                           relevant information for the agent. This must include
                           an 'available_actions' key containing the action descriptions.
        """
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to its initial state.
        
        Args:
            seed: Optional seed for reproducibility.
            
        Returns:
            Tuple[Any, Dict[str, Any]]: Initial observation and info dictionary.
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: The action to take in the environment.
            
        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]: 
                - observation: The new observation after taking the action
                - reward: The reward received from taking the action
                - terminated: Whether the episode has terminated
                - truncated: Whether the episode was truncated
                - info: Additional information about the step
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Clean up any resources used by the environment.
        """
        pass 