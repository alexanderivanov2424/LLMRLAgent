from typing import Any, Dict, Tuple, Optional, SupportsFloat, List, Literal, Type
import gymnasium as gym
from gymnasium import Space
import minigrid.core
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
import minigrid.wrappers
from environment.base_environment import BaseEnvironment, Action, ActionResponse
from pydantic import BaseModel, Field, create_model

class MiniGridEnvironment(BaseEnvironment):
    """
    Concrete implementation of BaseEnvironment for MiniGrid environments.
    This class wraps the Gymnasium MiniGrid environment and provides
    a standardized interface for our RL framework.
    """
    
    # Action space description
    ACTION_DESCRIPTIONS = {
        0: Action(action_name="left", action_description="Turn left"),
        1: Action(action_name="right", action_description="Turn right"),
        2: Action(action_name="forward", action_description="Move forward"),
        3: Action(action_name="pickup", action_description="Unused"),
        4: Action(action_name="drop", action_description="Unused"),
        5: Action(action_name="toggle", action_description="Unused"),
        6: Action(action_name="done", action_description="Unused")
    }

    # Type for valid action keys
    ValidActionKey = Literal[0, 1, 2, 3, 4, 5, 6]
    VALID_RESPONSE: ActionResponse = create_model(
        "ActionResponse",
        action=(Literal[0, 1, 2, 3, 4, 5, 6], Field(description="The action to take")),
        reasoning=(str, Field(description="The reasoning for the action"))
    ) # type: ignore

    def __init__(self, env_name: str = "MiniGrid-Empty-5x5-v0", seed: Optional[int] = None, render_mode: Optional[Literal["human", "rgb_array", "rgb_array_debug"]] = None):
        """
        Initialize the MiniGrid environment.
        
        Args:
            env_name: The name of the MiniGrid environment to create
            seed: Optional seed for reproducibility
        """
        self.env = gym.make(env_name, render_mode=render_mode)
        self.env = minigrid.wrappers.FullyObsWrapper(self.env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.seed = seed
        
    def get_action_space(self) -> Space:
        """Returns the action space of the environment."""
        return self.action_space
    
    def get_observation_space(self) -> Space:
        """Returns the observation space of the environment."""
        return self.observation_space
    
    def get_action_descriptions(self) -> Dict[int, Action]:
        """
        Returns a dictionary describing all available actions.
        
        Returns:
            Dict[int, Action]: Dictionary mapping action numbers to their names and descriptions
        """
        return self.ACTION_DESCRIPTIONS
    
    def get_valid_response(self) -> ActionResponse:
        """
        Returns the valid response for the environment.
        """
        return self.VALID_RESPONSE
    
    def grid_to_text(self, image: Any) -> str:
        """
        Convert the grid observation to a text representation.
        The image is a NxNx3 numpy array with values between 0 and 255.
        Each cell contains (OBJECT_IDX, COLOR_IDX, STATE).

        OBJECT_TO_IDX = {
            "unseen": 0,
            "empty": 1,
            "wall": 2,
            "floor": 3,
            "door": 4,
            "key": 5,
            "ball": 6,
            "box": 7,
            "goal": 8,
            "lava": 9,
            "agent": 10,
        }
        """
        height, width = image.shape[0], image.shape[1]
        text_grid = []
        for i in range(height):
            row = []
            for j in range(width):
                obj_type = image[i, j, 0]
                color = image[i, j, 1]  # 0=red, 1=green, 2=blue, 3=purple, 4=yellow, 5=grey
                state = image[i, j, 2]  # 0=open, 1=closed, 2=locked

                # Convert object index to character
                if obj_type == 0:  # unseen
                    row.append(" ")
                elif obj_type == 1:  # empty
                    row.append("·")
                elif obj_type == 2:  # wall
                    row.append("█")
                elif obj_type == 3:  # floor
                    row.append("·")
                elif obj_type == 4:  # door
                    if state == 0:  # open
                        row.append("╝")
                    elif state == 1:  # closed
                        row.append("║")
                    else:  # locked
                        row.append("▓")
                elif obj_type == 5:  # key
                    row.append("⚷")
                elif obj_type == 6:  # ball
                    row.append("○")
                elif obj_type == 7:  # box
                    row.append("□")
                elif obj_type == 8:  # goal
                    row.append("G")
                elif obj_type == 9:  # lava
                    row.append("L")
                elif obj_type == 10:  # agent
                    # Add direction indicator for agent
                    direction_symbols = ["▶", "▼", "◀", "▲"]  # 0=right, 1=down, 2=left, 3=up
                    row.append(direction_symbols[self.last_direction])
                else:
                    row.append("?")  # unknown
            text_grid.append(" ".join(row))
        return "\n".join(text_grid)

    def _find_first_floor(self, image: Any) -> Tuple[int, int]:
        """Find the first floor position in the grid."""
        for i in range(7):
            for j in range(7):
                if image[i, j, 0] in [1, 3]:  # empty or floor
                    return (i, j)
        return (0, 0)  # fallback if no floor found

    def format_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats the raw observation into a structured format.
        
        The MiniGrid observation is a dictionary containing:
        - image: A 7x7x3 array representing the agent's view
        - direction: The agent's current direction (0=right, 1=down, 2=left, 3=up)
        - mission: A string describing the current mission
        
        Returns:
            Dict containing the formatted observation with additional context
        """
        # Extract the raw observation components
        image = observation["image"]
        direction = observation["direction"]
        mission = observation["mission"]
        
        # Store the direction for the grid_to_text method
        self.last_direction = direction

        # Format the observation into a more structured format
        formatted_obs = {
            "grid_text": self.grid_to_text(image),  # Text representation of the grid
            "direction": direction,  # Current direction
            "mission": mission,  # Current mission
            "available_actions": self.ACTION_DESCRIPTIONS,  # Available actions and their descriptions
        }
        
        return formatted_obs
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to its initial state.
        
        Args:
            seed: Optional seed for reproducibility
            
        Returns:
            Tuple containing:
            - The initial observation
            - Additional information about the environment state
        """
        if seed is not None:
            self.seed = seed
        observation, info = self.env.reset(seed=self.seed)
        return self.format_observation(observation), info
    
    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: The action to take in the environment
            
        Returns:
            Tuple containing:
            - The new observation
            - The reward received
            - Whether the episode has terminated
            - Whether the episode was truncated
            - Additional information about the step
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.format_observation(observation), reward, terminated, truncated, info
    
    def close(self) -> None:
        """Clean up any resources used by the environment."""
        self.env.close() 