import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, Dict, List
from agents.base_agent import BaseAgent
from gymnasium import Space

class PPONetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared(x)
        action_probs = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return action_probs, value

class PPOAgent(BaseAgent):
    def __init__(
        self,
        action_space: Space,
        observation_space: Space,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 1.0,  # Value loss coefficient
        c2: float = 0.01,  # Entropy coefficient
        batch_size: int = 64,
        n_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(action_space, observation_space)
        
        self.device = device
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Initialize networks
        input_dim = np.prod(observation_space.shape)
        action_dim = action_space.n if hasattr(action_space, 'n') else np.prod(action_space.shape)
        self.network = PPONetwork(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
    def get_agent_ID(self) -> str:
        return "PPOAgent"
    
    def policy(self, observation: np.ndarray) -> Tuple[int, float, float]:
        """Returns action, log probability, and value estimate"""
        with torch.no_grad():
            observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action_probs, value = self.network(observation)
            
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> List[float]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return advantages
    
    def update(self, observation: np.ndarray, action: int, reward: float, terminated: bool, truncated: bool):
        """Collect experience and update policy if buffer is full"""
        # Get current action, log prob, and value
        _, log_prob, value = self.policy(observation)
        
        # Store experience
        self.buffer['observations'].append(observation)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(terminated or truncated)
        
        # If episode is done, update policy
        if terminated or truncated:
            self._update_policy()
            self.buffer = {k: [] for k in self.buffer.keys()}
    
    def _update_policy(self):
        """Perform PPO update"""
        # Convert buffer to tensors
        observations = torch.FloatTensor(self.buffer['observations']).to(self.device)
        actions = torch.LongTensor(self.buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        
        # Compute advantages
        advantages = torch.FloatTensor(
            self.compute_gae(
                self.buffer['rewards'],
                self.buffer['values'],
                self.buffer['dones']
            )
        ).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.n_epochs):
            # Generate random permutation for mini-batches
            indices = torch.randperm(len(observations))
            
            for start_idx in range(0, len(observations), self.batch_size):
                idx = indices[start_idx:start_idx + self.batch_size]
                
                # Get current action probabilities and values
                action_probs, values = self.network(observations[idx])
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions[idx])
                
                # Compute ratio and clipped surrogate loss
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = nn.MSELoss()(values.squeeze(), self.buffer['values'])
                
                # Compute entropy loss
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 