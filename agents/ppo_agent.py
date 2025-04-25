from typing import Dict, Optional

from pydantic import BaseModel, Field
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from agents.base_agent import BaseAgent

class PPOHyperparam(BaseModel):
    """PPO hyperparameters configuration."""
    policy: str = Field(
        default="MlpPolicy",
        description="Policy architecture to use (e.g., 'CnnPolicy', 'MlpPolicy')"
    )
    learning_rate: float = Field(
        default=3e-4, ge=0, description="Learning rate for the optimizer"
    )
    n_steps: int = Field(
        default=2048,
        gt=0,
        description="Number of steps to run for each environment per update",
    )
    batch_size: int = Field(
        default=64, 
        gt=0, 
        description="Minibatch size"
    )
    n_epochs: int = Field(
        default=10,
        gt=0,
        description="Number of epochs when optimizing the surrogate loss",
    )
    gamma: float = Field(
        default=0.99, 
        ge=0, 
        le=1, 
        description="Discount factor"
    )
    gae_lambda: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="Factor for trade-off of bias vs variance for Generalized Advantage Estimator",
    )
    clip_range: float = Field(
        default=0.2, ge=0, description="Clipping parameter for the surrogate objective"
    )
    ent_coef: float = Field(
        default=0.0, ge=0, description="Entropy coefficient for the loss calculation"
    )
    clip_range_vf: Optional[float] = Field(
        default=None,
        ge=0,
        description="Clipping parameter for the value function",
    )
    normalize_advantage: bool = Field(
        default=False,
        description="Whether to normalize the advantage function",
    )
    vf_coef: float = Field(
        default=0.5, 
        ge=0, 
        description="Value function coefficient for the loss calculation"
    )
    max_grad_norm: float = Field(
        default=0.5, 
        ge=0, 
        description="Maximum norm for the gradient clipping"
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Random seed for reproducibility",
    )
    verbose: int = Field(
        default=1,
        ge=0,
        description="Verbosity level (0 = no output, 1 = info, 2 = debug)",
    )

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"  # Prevent extra fields

    @classmethod
    def from_dict(cls, hyperparams: Dict) -> "PPOHyperparam":
        """Create a PPOHyperparam instance from a dictionary."""
        return cls(**hyperparams)

    def to_dict(self) -> Dict:
        """Convert the hyperparameters to a dictionary."""
        return self.dict()


class PPOAgent(BaseAgent):
    """PPO agent implementation with baseline and optimized hyperparameters."""

    # Baseline hyperparameters (from Stable Baselines3 defaults)
    BASELINE_HYPERPARAMS = PPOHyperparam()

    # Environment-specific optimized hyperparameters
    OPTIMIZED_HYPERPARAMS = {
        # TODO: check for consistent environment names
        # MINIGRID PPO HYPERPARAM SOURCES
        # SOURCE 1: https://www.researchgate.net/publication/371290959_Hyperparameters_in_Reinforcement_Learning_and_How_To_Tune_Them
        # SOURCE 2: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
        "MiniGrid-Empty-5x5-v0": PPOHyperparam(
            policy = "MlpPolicy",
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=6,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.05,
            normalize_advantage = false,
            max_grad_norm = 0.6,
            vf_coef = 0.6,
            
        ),
        "MiniGrid-DoorKey-5x5-v0": PPOHyperparam
        (
            policy = "MlpPolicy",
            learning_rate=1e-4,
            n_steps=2048, # performed comparably to 1024 in the paper
            batch_size=64,
            n_epochs=8,
            gamma=0.99,
            gae_lambda=0.9,
            clip_range=0.5,
            ent_coef=0.05,
            normalize_advantage = True,
            max_grad_norm = 0.6,
            vf_coef = 0.8,
        ),
        # GYM PPO HYPERPARAM SOURCES
        # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
        "CartPole-v1": PPOHyperparam
        (
            policy = "MlpPolicy",
            learning_rate=1e-3,
            n_steps=1024,
            batch_size=256,
            n_epochs=20,
            gamma=0.98,
            gae_lambda=0.8,
            clip_range=0.2,
            ent_coef=0.0,
        ),
        "LunarLander-v2": PPOHyperparam
        ( # May need additional tuning
            policy = "MlpPolicy",
            learning_rate=3e-4, # not in YML!
            n_steps=2048,
            batch_size=64,
            n_epochs=4,
            gamma=0.999,
            gae_lambda=0.98,
            ent_coef=0.01,
            clip_range=0.2, # not in YML!
        ),
        "Reacher-v2": PPOHyperparam
        ( # TUNED for MlpPolicy
         # SOURCE: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
            policy = "MlpPolicy",
            learning_rate=3e-4,
            n_steps=512,
            batch_size=32,
            n_epochs=5,
            gamma=0.9,
            gae_lambda=1.0,
            clip_range=0.3,
            ent_coef=7.5e-8,
            normalize_advantage = True,
            max_grad_norm = 0.9,
            vf_coef = 0.95,
        ),
        # Add more environment-specific optimized parameters as needed
    }

    def __init__(
        self,
        action_space,
        observation_space,
        env_id: str,
        use_optimized: bool = False,
        custom_hyperparams: Optional[PPOHyperparam] = None,
    ):
        """
        Initialize the PPO agent.

        Args:
            action_space: The action space of the environment
            observation_space: The observation space of the environment
            env_id: The environment ID
            use_optimized: Whether to use optimized hyperparameters
            custom_hyperparams: Optional custom hyperparameters to use instead of defaults
        """
        super().__init__(action_space, observation_space)
        self.env_id = env_id
        self.use_optimized = use_optimized

        if custom_hyperparams is not None:
            self.hyperparams = custom_hyperparams
        else:
            self.hyperparams = (
                self.OPTIMIZED_HYPERPARAMS.get(env_id, self.BASELINE_HYPERPARAMS)
                if use_optimized
                else self.BASELINE_HYPERPARAMS
            )
        self.model = None

    def get_agent_ID(self):
        """Return the agent's identifier."""
        return "PPOAgent"

    def policy(self, observation):
        """Select an action based on the current observation."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def update(self, observation, action, reward, terminated, truncated):
        """Update the agent's internal state after taking an action."""
        # PPO updates are handled internally by the model during training
        pass

    def train(self, env, total_timesteps: int = 100000, seed: int = 0):
        """Train the PPO agent."""
        # Create the model
        self.model = PPO(
            policy=self.hyperparams.policy,
            env=env,
            learning_rate=self.hyperparams.learning_rate,
            n_steps=self.hyperparams.n_steps,
            batch_size=self.hyperparams.batch_size,
            n_epochs=self.hyperparams.n_epochs,
            gamma=self.hyperparams.gamma,
            gae_lambda=self.hyperparams.gae_lambda,
            clip_range=self.hyperparams.clip_range,
            ent_coef=self.hyperparams.ent_coef,
            clip_range_vf=self.hyperparams.clip_range_vf,
            normalize_advantage=self.hyperparams.normalize_advantage,
            vf_coef=self.hyperparams.vf_coef,
            max_grad_norm=self.hyperparams.max_grad_norm,
            seed=self.hyperparams.seed,
            verbose=self.hyperparams.verbose,
        )

        # Create evaluation callback
        eval_callback = EvalCallback(
            env,
            best_model_save_path=f"./experiment_data/ppo_{self.env_id.split('/')[-1]}/",
            log_path=f"./experiment_data/ppo_{self.env_id.split('/')[-1]}/",
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
