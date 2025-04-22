from pydantic import BaseModel
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from agents.base_agent import BaseAgent


class DQNHyperparameters(BaseModel):
    learning_rate: float = 1e-4
    buffer_size: int = 100000
    learning_starts: int = 1000
    batch_size: int = 32
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05


class DQNAgent(BaseAgent):
    """DQN agent implementation with baseline and optimized hyperparameters."""

    def __init__(
        self,
        action_space,
        observation_space,
        env_id: str,
        use_optimized: bool = False,
        hyperparams: DQNHyperparameters = DQNHyperparameters(),
    ):
        """
        Initialize the DQN agent.

        Args:
            action_space: The action space of the environment
            observation_space: The observation space of the environment
            env_id: The environment ID
            use_optimized: Whether to use optimized hyperparameters
        """
        super().__init__(action_space, observation_space)
        self.hyperparams = hyperparams
        self.env_id = env_id
        self.use_optimized = use_optimized
        self.model = None

    def get_agent_ID(self):
        """Return the agent's identifier."""
        return "DQNAgent"

    def policy(self, observation):
        """Select an action based on the current observation."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def update(self, observation, action, reward, terminated, truncated):
        """Update the agent's internal state after taking an action."""
        # DQN updates are handled internally by the model during training
        pass

    def train(self, env, total_timesteps: int = 100000, seed: int = 0):
        """Train the DQN agent."""
        # Create the model
        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=self.hyperparams.learning_rate,
            buffer_size=self.hyperparams.buffer_size,
            learning_starts=self.hyperparams.learning_starts,
            batch_size=self.hyperparams.batch_size,
            tau=self.hyperparams.tau,
            gamma=self.hyperparams.gamma,
            train_freq=self.hyperparams.train_freq,
            gradient_steps=self.hyperparams.gradient_steps,
            target_update_interval=self.hyperparams.target_update_interval,
            exploration_fraction=self.hyperparams.exploration_fraction,
            exploration_initial_eps=self.hyperparams.exploration_initial_eps,
            exploration_final_eps=self.hyperparams.exploration_final_eps,
            verbose=1,
        )

        # Create evaluation callback
        eval_callback = EvalCallback(
            env,
            best_model_save_path=f"./experiment_data/dqn_{self.env_id.split('/')[-1]}/",
            log_path=f"./experiment_data/dqn_{self.env_id.split('/')[-1]}/",
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        # Train the model
        self.model.learn(total_timesteps=total_timesteps, callback=eval_callback)

        return self.model

    def evaluate(self, model, env, n_eval_episodes: int = 10):
        """Evaluate the agent's performance."""
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )
        return mean_reward, std_reward
