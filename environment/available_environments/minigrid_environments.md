# MiniGrid environments

---

## Navigation & Exploration

- **Empty**:An empty room where the agent must reach the green goal square. Useful for validating RL algorithms and experimenting with sparse rewards

- **Four Rooms**:A classic environment with four interconnected rooms. The agent must navigate to the goal, which can be in any room

- **Crossing**:The agent must reach the goal while avoiding obstacles like lava or walls. Useful for studying safe exploration

- **Multi Room**:A series of connected rooms with doors. The agent must traverse through them to reach the goal

- **Memory**:Tests the agent's memory. The agent sees an object, then must choose the matching object after navigating a hallway

- **Dynamic Obstacles**:An empty room with moving obstacles. The agent must reach the goal without collisions

---

## Object Interaction & Manipulation

- **Door Key** The agent must pick up a key to unlock a door and reach the goal. Useful for experimenting with sparse reward.

- **Key Corridor** Similar to Door Key, but the key is hidden in another room. Encourages exploration and planning.

- **Unlock** The agent must unlock a door to process.

- **Unlock Pickup** The agent must unlock a door and pick up an object.

- **Blocked Unlock Pickup** The agent must unblock, unlock, and pick up an object.

- **Put Near** The agent must place an object near another specified object.

---

## Complex Reasoning & Planning

- **Fetch*: The agent must fetch a specified object and bring it to a target location.

- **Obstructed Maze*: A maze with obstacles that the agent must navigate to reach the goal.

- **Playground*: A large room with various objects and obstacles. Encourages exploration and interaction.

- **Red Blue Door*: The agent must choose between red and blue doors to reach the goal.

- **Go To Door*: The agent must navigate to a specified door.

- **Go To Object*: The agent must navigate to a specified object.

- **Dist Shift*: Tests the agent's ability to adapt to distributional shifts in the environment.

- **Lava Gap*: The agent must cross a gap filled with lava to reach the goal.

- **Locked Room*: The agent must find a key to unlock a room and reach the goal.
