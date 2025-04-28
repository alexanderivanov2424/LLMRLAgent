from pydantic import BaseModel
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from agents.base_agent import BaseAgent


class DQNHyperparameters(BaseModel):
    """DQN hyperparameters configuration."""

    policy: str = Field(
        default="CnnPolicy",
        description="Policy architecture to use (e.g., 'CnnPolicy', 'MlpPolicy')",
    )
    learning_rate: float = Field(
        default=1e-4, ge=0, description="Learning rate for the optimizer"
    )
    buffer_size: int = Field(
        default=100000, gt=0, description="Size of the replay buffer"
    )
    learning_starts: int = Field(
        default=100000, gt=0, description="Number of steps before starting to learn"
    )
    batch_size: int = Field(default=32, gt=0, description="Batch size for training")
    tau: float = Field(
        default=1.0,
        ge=0,
        description="Soft update coefficient for the target network, 'Polyak Update' (between 0 and 1)",
    )
    gamma: float = Field(
        default=0.99, ge=0, le=1, description="Discount factor for future rewards"
    )
    train_freq: int = Field(
        default=4, gt=0, description="Frequency of training the model (in steps)"
    )
    gradient_steps: int = Field(
        default=1,
        gt=0,
        description="Number of gradient steps to take after each rollout. Set to -1 to use the same number of steps as train_freq",
    )
    target_update_interval: int = Field(
        default=1000,
        gt=0,
        description="Number of steps before updating the target network",
    )
    exploration_fraction: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Fraction of the total training time over which the exploration rate is reduced",
    )
    exploration_initial_eps: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Initial value of the exploration rate (epsilon)",
    )
    exploration_final_eps: float = Field(
        default=0.01,
        ge=0,
        le=1,
        description="Final value of the exploration rate (epsilon)",
    )
    max_grad_norm: float = Field(
        default=10.0, ge=0, description="Maximum norm for gradient clipping"
    )
    seed: int = Field(default=None, ge=0, description="Random seed for reproducibility")
    verbose: int = Field(
        default=1,
        ge=0,
        description="Verbosity level (0: no output, 1: training info, 2: debug info)",
    )

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"  # Disallow extra fields not defined in the model
        anystr_strip_whitespace = True  # Strip whitespace from strings

    @classmethod
    def from_dict(cls, data: dict) -> "DQNHyperparameters":
        """Create a DQNHyperparameters instance from a dictionary."""
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert the DQNHyperparameters instance to a dictionary."""
        return self.dict()


class DQNAgent(BaseAgent):
    """DQN agent implementation with baseline and optimized hyperparameters."""

    BASELINE_HYPERPARAMS = DQNHyperparameters()

    OPTIMIZED_HYPERPARAMS = {
        "MiniGrid-DoorkKey-5x5-v0": DQNHyperparameters(  # SOURCE: https://www.researchgate.net/publication/371290959_Hyperparameters_in_Reinforcement_Learning_and_How_To_Tune_Them
            policy="CnnPolicy",
            learning_rate=5e-7,
            buffer_size=1000000,
            learning_starts=100,
            batch_size=64,
            tau=1.0,
            gamma=0.999,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
        ),
        "MiniGrid-Empty-5x5-v0": DQNHyperparameters(  # SOURCE: https://www.researchgate.net/publication/371290959_Hyperparameters_in_Reinforcement_Learning_and_How_To_Tune_Them
            policy="CnnPolicy",
            learning_rate=5e-6,
            buffer_size=500000,
            learning_starts=1000,
            batch_size=128,
            tau=0.6,
            gamma=0.99,
            train_freq=2,
            gradient_steps=4,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
        ),
        "CartPole-v1": DQNHyperparameters(  # SOURCE: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
            # STATED IT WAS "ALMOST TUNED"
            policy="MlpPolicy",
            learning_rate=2.3e-3,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=64,
            tau=0.1,  # not in YML
            gamma=0.99,
            train_freq=256,
            gradient_steps=128,
            target_update_interval=10,
            exploration_fraction=0.16,
            exploration_initial_eps=1.0,  # not in YML
            exploration_final_eps=0.04,
        ),
        "LunarLander-v2": DQNHyperparameters(  # SOURCE: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
            policy="MlpPolicy",
            learning_rate=6.3e-4,
            buffer_size=50000,
            learning_starts=0,
            batch_size=128,
            tau=0.1,  # not in YML
            gamma=0.99,
            train_freq=4,
            gradient_steps=-1,
            target_update_interval=250,
            exploration_fraction=0.12,
            exploration_initial_eps=1.0,  # not in YML
            exploration_final_eps=0.1,
        ),
    }

    def __init__(
        self,
        action_space,
        observation_space,
        env_id: str,
        use_optimized: bool = False,
        hyperparams: Optional[DQNHyperparameters] = None,
    ):
        """
        Initialize the DQN agent.

        Args:
            action_space: The action space of the environment
            observation_space: The observation space of the environment
            env_id: The environment ID
            use_optimized: Whether to use optimized hyperparameters
            hyperparams: Custom hyperparameters to use instead of defaults
        """
        super().__init__(action_space, observation_space)
        self.env_id = env_id
        self.use_optimized = use_optimized
        self.model = None

        if hyperparams is None:
            if use_optimized and env_id in self.OPTIMIZED_HYPERPARAMS:
                self.hyperparams = self.OPTIMIZED_HYPERPARAMS[env_id]
            else:
                self.hyperparams = self.BASELINE_HYPERPARAMS
        else:
            self.hyperparams = hyperparams

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

    def train(self, env, total_timesteps: int = 1000, seed: int = 0):
        """Train the DQN agent."""
        # Create the model
        self.model = DQN(
            policy=self.hyperparams.policy,
            env=env,
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
            max_grad_norm=self.hyperparams.max_grad_norm,
            seed=self.hyperparams.seed,
            verbose=self.hyperparams.verbose,
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
