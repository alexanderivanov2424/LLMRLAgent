# Gymnasium Environments

---

## Classic Control

These are fundamental reinforcement learning environments based on classic control problems.

- **Acrobot**: A two-link pendulum where the goal is to swing the end-effector to a target height.
- **CartPole**: Balance a pole on a moving cart by applying forces to the cart.
- **MountainCar**: Drive a car up a steep hill; the car is underpowered and must build momentum.
- **MountainCarContinuous**: A continuous version of MountainCar with continuous action space.
- **Pendulum**: Swing a pendulum upright and keep it balanced.

---

## Box2D

Physics-based environments using the Box2D engine.

- **BipedalWalker**: Control a two-legged robot to walk across terrain.
- **CarRacing**: Drive a car around a track using top-down view.
- **LunarLander**: Land a spacecraft on the moon's surface.

---

## Toy Text

Simple text-based environments for testing and debugging algorithms.

- **Blackjack**: Play the card game Blackjack against a dealer.
- **Taxi**: Navigate a taxi to pick up and drop off passengers.
- **CliffWalking**: Navigate a gridworld while avoiding cliffs.
- **FrozenLake**: Navigate a frozen lake without falling into holes.

---

## MuJoCo

Environments using the MuJoCo physics engine for complex robotic simulations.

- **Ant**: Control a four-legged robot to move forward.
- **HalfCheetah**: Control a two-legged robot resembling a cheetah to run forward.
- **Hopper**: Control a one-legged robot to hop forward.
- **Humanoid**: Control a humanoid robot to walk forward.
- **InvertedPendulum**: Balance a pole on a cart.
- **Reacher**: Control a two-link arm to reach a target.
- **Swimmer**: Control a three-link robot to swim forward.
- **Walker2D**: Control a two-legged robot to walk forward.

---

## Atari

Classic Atari 2600 games for benchmarking reinforcement learning algorithms.

Some examples include:

- **Pong**: Classic table tennis game.
- **Breakout**: Break bricks with a ball.
- **Space Invaders**: Shoot down alien invaders.
- **Seaquest**: Rescue divers while avoiding enemies.
- **Qbert**: Change the colors of a pyramid by hopping on cubes.

For a complete list, refer to the [Atari environments documentation](https://gymnasium.farama.org/environments/atari/complete_list/).

---

## Third-Party Environments

These are environments developed by the community and compatible with Gymnasium.

### Autonomous Driving

- **BlueSky-Gym**: Air traffic control simulations.
- **gym-electric-motor**: Simulate electric motor control.
- **racecar_gym**: Miniature racecar simulation using PyBullet.
- **sumo-rl**: Traffic signal control using the SUMO simulator.

### Biological / Medical

- **ICU-Sepsis**: Simulate sepsis treatment in ICU settings.

### Economic / Financial

- **gym-anytrading**: Trading environments for stocks and forex.
- **gym-mtsim**: Simulate trading on MetaTrader 5 platform.
- **gym-trading-env**: Customizable trading environment using historical data.

### Electrical / Energy

- **EV2Gym**: Electric vehicle smart charging simulations.

### Games

- **Craftium**: Minecraft-like environment using Minetest.
- **flappy-bird-env**: Flappy Bird game environment.
- **Generals.io bots**: Strategy game on a 2D grid.
- **pystk2-gymnasium**: SuperTuxKart racing game environment.
- **QWOP**: Simulate the QWOP running game.
- **Tetris Gymnasium**: Tetris game environment.
- **tmrl**: TrackMania 2020 racing game environment.

### Mathematics / Computational

- **spark-sched-sim**: Simulate job scheduling in Apache Spark.

### Robotics

- **BSK-RL**: Spacecraft planning and scheduling.
- **Connect-4-gym**: Connect Four game environment.
- **FlyCraft**: Fixed-wing UAV simulation.
- **gym-pybullet-drones**: Quadcopter control using PyBullet.
- **Itomori**: UAV risk-aware flight environment.
- **OmniIsaacGymEnvs**: Robotics environments using NVIDIA Omniverse Isaac.
- **panda-gym**: Robotic arm simulations using PyBullet.
- **PyFlyt**: UAV flight simulator for reinforcement learning research.
- **safe-control-gym**: Evaluate safety in reinforcement learning algorithms.
- **Safety-Gymnasium**: Safety-focused reinforcement learning environments.

### Telecommunication Systems

- **mobile-env**: Coordination of wireless mobile networks.
