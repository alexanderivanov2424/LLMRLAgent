import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import BaseAgent
from gymnasium import Space

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        action_space: Space,
        observation_space: Space,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__(action_space, observation_space)
        
        self.device = device
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        input_dim = int(np.prod(observation_space.shape))
        action_dim = action_space.n
        
        self.network = QNetwork(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def get_agent_ID(self) -> str:
        return "QLearningAgent"

    def policy(self, observation: np.ndarray) -> int:
        """ ε-greedy policy """
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.network(obs_tensor)
        return q_values.argmax(dim=1).item()

    def update(self, observation: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)

        q_values = self.network(obs_tensor)
        q_value = q_values[0, action]

        with torch.no_grad():
            next_q_values = self.network(next_obs_tensor)
            max_next_q_value = next_q_values.max(dim=1)[0]
            target = reward + (0 if done else self.gamma * max_next_q_value.item())

        loss = self.criterion(q_value, torch.tensor(target).to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)