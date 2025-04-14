from typing_extensions import Any, Dict, Tuple, Optional, SupportsFloat, Literal
from gymnasium import Env, Space
from pydantic import BaseModel, Field, create_model
from environment.base_environment import BaseEnvironment, Action, ActionResponse
import gymnasium as gym

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

    def format_observation(self, observation: Any) -> Dict[str, Any]:
        taxi_row = observation // 100
        observation %= 100
        taxi_col = observation // 20
        observation %= 20
        passenger_loc = observation // 4
        destination = observation % 4

        return {
            "description": (
                f"The taxi is at ({taxi_row}, {taxi_col}). "
                f"The passenger is at location {passenger_loc}. "
                f"The destination is location {destination}."
            ),
            "available_actions": self.ACTIONS,
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        return self.format_observation(observation), info

    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
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
        self.grid_size = int(self.observation_space.n ** 0.5)

    def get_action_space(self) -> Space:
        return self.action_space

    def get_observation_space(self) -> Space:
        return self.observation_space

    def get_action_descriptions(self) -> Dict[int, Action]:
        return self.ACTIONS

    def get_valid_response(self) -> BaseModel:
        return self.VALID_RESPONSE

    def format_observation(self, observation: Any) -> Dict[str, Any]:
        row = observation // self.grid_size
        col = observation % self.grid_size
        return {
            "description": f"The agent is at position ({row}, {col}) on a {self.grid_size}x{self.grid_size} frozen lake.",
            "available_actions": self.ACTIONS,
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        return self.format_observation(observation), info

    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
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

    def step(self, action: int) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.format_observation(obs), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()
