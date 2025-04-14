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

        passenger_coords = "in the taxi" if passenger_loc == 4 else f"at {locs[passenger_loc]}"
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
