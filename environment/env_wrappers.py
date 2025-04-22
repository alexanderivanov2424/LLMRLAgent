import gymnasium as gym
import minigrid.wrappers
from gymnasium import Space
from pydantic import BaseModel, Field, create_model
from typing_extensions import Any, Dict, Literal, Optional, SupportsFloat, Tuple

from environment.base_environment import Action, ActionResponse, BaseEnvironment

"""
Implementations of Environments for
- Taxi-v3
- FrozenLake-v1
- CartPole-v1
- LunarLander-v2
- Reacher-v5
- MiniGrid environments (e.g., MiniGrid-Empty-5x5-v0, MiniGrid-DoorKey-5x5-v0)

TODO:
- Add BabyAI environment wrapper
"""

class TaxiEnvironment(BaseEnvironment):
    ACTIONS = {
        0: Action(action_name="south", action_description="Move the taxi south"),
        1: Action(action_name="north", action_description="Move the taxi north"),
        2: Action(action_name="east", action_description="Move the taxi east"),
        3: Action(action_name="west", action_description="Move the taxi west"),
        4: Action(action_name="pickup", action_description="Pick up the passenger"),
        5: Action(action_name="dropoff", action_description="Drop off the passenger"),
    }

    VALID_RESPONSE = create_model(
        "TaxiActionResponse",
        action=(Literal[0, 1, 2, 3, 4, 5], Field(description="The action to take")),
        reasoning=(str, Field(description="The reasoning for the action")),
    )

    def __init__(self):
        self.env = gym.make("Taxi-v3")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def get_action_space(self) -> Space:
        return self.action_space

    def get_observation_space(self) -> Space:
        return self.observation_space

    def get_action_descriptions(self) -> Dict[int, Action]:
        return self.ACTIONS

    def get_valid_response(self) -> BaseModel:
        return self.VALID_RESPONSE

    def grid_to_text(self, obs: int) -> str:
        taxi_row = obs // 100
        rem = obs % 100
        taxi_col = rem // 20
        rem %= 20
        pass_idx = rem // 4
        dest_idx = rem % 4

        locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        grid = [[" ." for _ in range(5)] for _ in range(5)]

        if pass_idx < 4:
            pr, pc = locs[pass_idx]
            grid[pr][pc] = "🧍"
        dr, dc = locs[dest_idx]
        grid[dr][dc] = "🎯"

        tr, tc = taxi_row, taxi_col
        if pass_idx == 4:
            grid[tr][tc] = "🧍🎯"
        else:
            grid[tr][tc] = "🚕"

        return "\n".join("".join(row) for row in grid)

    def format_observation(self, obs: int) -> Dict[str, Any]:
        taxi_row = obs // 100
        rem = obs % 100
        taxi_col = rem // 20
        rem %= 20
        passenger_loc = rem // 4
        destination = rem % 4

        locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

        passenger_coords = (
            "in the taxi" if passenger_loc == 4 else f"at {locs[passenger_loc]}"
        )
        destination_coords = locs[destination]

        return {
            "description": (
                f"The taxi is at ({taxi_row}, {taxi_col}). "
                f"The passenger is {passenger_coords}. "
                f"The destination is at {destination_coords}."
            ),
            "available_actions": self.ACTIONS,
            "grid_text": self.grid_to_text(obs),
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        return self.format_observation(observation), info

    def step(
        self, action: int
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.format_observation(obs), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()


class FrozenLakeEnvironment(BaseEnvironment):
    ACTIONS = {
        0: Action(action_name="left", action_description="Move left"),
        1: Action(action_name="down", action_description="Move down"),
        2: Action(action_name="right", action_description="Move right"),
        3: Action(action_name="up", action_description="Move up"),
    }

    VALID_RESPONSE = create_model(
        "FrozenLakeActionResponse",
        action=(Literal[0, 1, 2, 3], Field(description="The action to take")),
        reasoning=(str, Field(description="The reasoning for the action")),
    )

    def __init__(self):
        self.env = gym.make("FrozenLake-v1", is_slippery=False)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.grid_size = int(self.observation_space.n**0.5)

    def get_action_space(self) -> Space:
        return self.action_space

    def get_observation_space(self) -> Space:
        return self.observation_space

    def get_action_descriptions(self) -> Dict[int, Action]:
        return self.ACTIONS

    def get_valid_response(self) -> BaseModel:
        return self.VALID_RESPONSE

    def grid_to_text(self, observation: int) -> str:
        grid = self.env.unwrapped.desc.astype(str)
        size = grid.shape[0]
        grid = [[c for c in row] for row in grid]

        row = observation // size
        col = observation % size
        grid[row][col] = "A"

        return "\n".join(" ".join(cell for cell in row) for row in grid)

    def format_observation(self, observation: Any) -> Dict[str, Any]:
        row = observation // self.grid_size
        col = observation % self.grid_size

        return {
            "description": f"The agent is at position ({row}, {col}) on a {self.grid_size}x{self.grid_size} frozen lake.",
            "available_actions": self.ACTIONS,
            "grid_text": self.grid_to_text(observation),
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        return self.format_observation(observation), info

    def step(
        self, action: int
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.format_observation(obs), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()


class CartPoleEnvironment(BaseEnvironment):
    ACTIONS = {
        0: Action(action_name="left", action_description="Push the cart to the left"),
        1: Action(action_name="right", action_description="Push the cart to the right"),
    }

    VALID_RESPONSE = create_model(
        "CartPoleActionResponse",
        action=(Literal[0, 1], Field(description="The action to take")),
        reasoning=(str, Field(description="The reasoning for the action")),
    )

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def get_action_space(self) -> Space:
        return self.action_space

    def get_observation_space(self) -> Space:
        return self.observation_space

    def get_action_descriptions(self) -> Dict[int, Action]:
        return self.ACTIONS

    def get_valid_response(self) -> BaseModel:
        return self.VALID_RESPONSE

    def format_observation(self, observation: Any) -> Dict[str, Any]:
        pos, vel, angle, ang_vel = observation
        return {
            "description": (
                f"The cart is at position {pos:.2f} with velocity {vel:.2f}. "
                f"The pole is at angle {angle:.2f} radians with angular velocity {ang_vel:.2f}."
            ),
            "available_actions": self.ACTIONS,
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        return self.format_observation(observation), info

    def step(
        self, action: int
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.format_observation(obs), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()


class LunarLanderEnvironment(BaseEnvironment):
    ACTIONS = {
        0: Action(action_name="do nothing", action_description="Fires no engine"),
        1: Action(
            action_name="fire left engine",
            action_description="Fires the left orientation engine",
        ),
        2: Action(
            action_name="fire main engine",
            action_description="Fires the central engine to go up",
        ),
        3: Action(
            action_name="fire right engine",
            action_description="Fires the right orientation engine",
        ),
    }

    VALID_RESPONSE = create_model(
        "LunarLanderActionResponse",
        action=(Literal[0, 1, 2, 3], Field(description="The action to take")),
        reasoning=(str, Field(description="The reasoning for the action")),
    )

    def __init__(self):
        self.env = gym.make("LunarLander-v2")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def get_action_space(self) -> Space:
        return self.action_space

    def get_observation_space(self) -> Space:
        return self.observation_space

    def get_action_descriptions(self) -> Dict[int, Action]:
        return self.ACTIONS

    def get_valid_response(self) -> BaseModel:
        return self.VALID_RESPONSE

    def format_observation(self, observation: Any) -> Dict[str, Any]:
        x, y, vel_x, vel_y, angle, ang_vel, left_contact, right_contact = observation
        return {
            "description": (
                f"Position=({x:.2f}, {y:.2f}), Velocity=({vel_x:.2f}, {vel_y:.2f}), "
                f"Angle={angle:.2f} rad, Angular Velocity={ang_vel:.2f}, "
                f"Left Leg Contact={'yes' if left_contact else 'no'}, "
                f"Right Leg Contact={'yes' if right_contact else 'no'}."
            ),
            "available_actions": self.ACTIONS,
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        return self.format_observation(observation), info

    def step(
        self, action: int
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.format_observation(obs), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

class ReacherEnvironment(BaseEnvironment):
    """
    Wrapper for the Gymnasium Reacher-v5 environment.
    """
    # The action space is continuous: 2D torques in [-1, 1]
    ACTIONS = {
        "action": Action(
            action_name="torques",
            action_description="A tuple (a, b) representing torques applied at the two joints, each in [-1, 1]."
        )
    }

    VALID_RESPONSE = create_model(
        "ReacherActionResponse",
        action=(Tuple[float, float], Field(description="Tuple of two floats in [-1, 1] for joint torques")),
        reasoning=(str, Field(description="The reasoning for the action")),
    )

    def __init__(self):
        self.env = gym.make("Reacher-v5")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def get_action_space(self) -> Space:
        return self.action_space

    def get_observation_space(self) -> Space:
        return self.observation_space

    def get_action_descriptions(self) -> Dict[str, Action]:
        return self.ACTIONS

    def get_valid_response(self) -> BaseModel:
        return self.VALID_RESPONSE

    def format_observation(self, observation: Any) -> Dict[str, Any]:
        # Observation: [cos(q1), cos(q2), sin(q1), sin(q2), qpos_target_x, qpos_target_y, qvel_1, qvel_2, xpos_1, xpos_2]
        obs = observation
        desc = (
            f"cos(joint angles): ({obs[0]:.2f}, {obs[1]:.2f}), "
            f"sin(joint angles): ({obs[2]:.2f}, {obs[3]:.2f}), "
            f"target position: ({obs[4]:.2f}, {obs[5]:.2f}), "
            f"joint velocities: ({obs[6]:.2f}, {obs[7]:.2f}), "
            f"vector fingertip-to-target: ({obs[8]:.2f}, {obs[9]:.2f})"
        )
        return {
            "description": desc,
            "available_actions": self.ACTIONS,
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        return self.format_observation(observation), info

    def step(
        self, action: Tuple[float, float]
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.format_observation(obs), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

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
        6: Action(action_name="done", action_description="Unused"),
    }

    # Type for valid action keys
    ValidActionKey = Literal[0, 1, 2, 3, 4, 5, 6]
    VALID_RESPONSE: ActionResponse = create_model(
        "ActionResponse",
        action=(Literal[0, 1, 2, 3, 4, 5, 6], Field(description="The action to take")),
        reasoning=(str, Field(description="The reasoning for the action")),
    )  # type: ignore

    def __init__(
        self,
        env_name: str = "MiniGrid-Empty-5x5-v0",
        seed: Optional[int] = None,
        render_mode: Optional[Literal["human", "rgb_array", "rgb_array_debug"]] = None,
    ):
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

        # Iterate through rows from top to bottom (reversed to match MiniGrid visualization)
        for j in range(width):
            row = []
            # Iterate through columns from left to right, but using transposed coordinates
            for i in range(height):
                obj_type = image[i, j, 0]
                color = image[
                    i, j, 1
                ]  # 0=red, 1=green, 2=blue, 3=purple, 4=yellow, 5=grey
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
                    direction_symbols = [
                        "▶",
                        "▼",
                        "◀",
                        "▲",
                    ]  # 0=right, 1=down, 2=left, 3=up
                    row.append(direction_symbols[self.last_direction])
                else:
                    row.append("?")  # unknown
            text_grid.append(" ".join(row))
        return "\n".join(
            text_grid
        )  # Reverse the rows to match MiniGrid's top-to-bottom orientation

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

    def step(
        self, action: int
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
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
