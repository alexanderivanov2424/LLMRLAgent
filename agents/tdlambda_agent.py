import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from copy import deepcopy
from tqdm import tqdm
from agents.base_agent import BaseAgent

###
# Citation for Original Implementation: Based on the implementation provided
###

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Hidden layers. In this case, we will use 3 hidden layers.
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    # Feed forward
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TDLambdaAgent(BaseAgent):
    def __init__(self, action_space, observation_space, alpha=0.001, gamma=0.9, epsilon=0.1, lambd=0.9, batch_size=32):
        super().__init__(action_space, observation_space)
        
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon
        self.batch_size = batch_size
        
        # Determine state and action dimensions
        self.state_dim = np.prod(observation_space.shape)
        self.action_dim = action_space.n if hasattr(action_space, 'n') else np.prod(action_space.shape)
        
        # Set up device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create two neural networks: Q_main and Q_target
        self.Q_main = Net(self.state_dim, self.action_dim).to(self.device)
        self.Q_target = deepcopy(self.Q_main)
        self.optimizer = optim.Adam(self.Q_main.parameters(), lr=self.alpha)
        
        # Memory replay buffer
        self.memory = deque(maxlen=10_000)
        
        # For eligibility traces
        self.trace_dict = {}
        self.current_state = None
        self.current_action = None
        self.episode_step = 0
        self.episode_return = 0
        self.episode_count = 0
        
        # For episode management
        self.step_limit = 2_000
        self.done = False

    def get_agent_ID(self):
        return "TDLambdaAgent"

    def policy(self, observation):
        """Implementation of BaseAgent's policy method"""
        return self.epsilon_greedy_policy(observation)

    def epsilon_greedy_policy(self, state):
        """Action selection strategy"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)  # Explore by taking a random action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return torch.argmax(self.Q_main(state_tensor)).item()  # Exploit by taking the best action

    def reset(self):
        """Reset Q_main and Q_target"""
        self.Q_main = Net(self.state_dim, self.action_dim).to(self.device)
        self.Q_target = deepcopy(self.Q_main)
        self.optimizer = optim.Adam(self.Q_main.parameters(), lr=self.alpha)
        self.memory.clear()
        self.reset_episode()

    def _soft_update_Qtarget(self, tau=0.01):
        """Update Q_target (soft update)"""
        with torch.no_grad():
            for target_param, param in zip(self.Q_target.parameters(), self.Q_main.parameters()):
                target_param += tau * (param - target_param)

    def _update_Qmain_weights(self, loss):
        """Update Q_main's weights"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update_Qtarget()

    def reset_episode(self):
        """Reset the episode"""
        self.current_state = None
        self.current_action = None
        self.done = False
        self.episode_step = 0
        self.episode_return = 0
        self.trace_dict = {}  # For storing eligibility traces {(state's x coordinate, state's y coordinate, action): trace value}

    def update(self, observation, action, reward, terminated, truncated):
        """Implementation of BaseAgent's update method"""
        next_state = observation
        done = terminated or truncated
        
        # Initialize if this is the first step
        if self.current_state is None:
            self.current_state = next_state
            self.current_action = action
            return
        
        # Get state and action
        state = self.current_state
        action_taken = self.current_action
        
        # Determine next action using epsilon-greedy
        next_action = self.epsilon_greedy_policy(next_state)
        
        # Increment trace for current state
        state_key = tuple(state)  # Convert state array to tuple for hashing
        trace_key = (*state_key, action_taken)
        if trace_key not in self.trace_dict:
            self.trace_dict[trace_key] = 0
        self.trace_dict[trace_key] += 1
        
        # Get trace list for batch processing
        trace = list(self.trace_dict.values())
        
        # Store in memory buffer
        self.memory.append((self.episode_count, state, action_taken, reward, next_state, next_action, done, trace))
        
        # Decay trace for past visited states
        self.trace_dict[trace_key] = (self.gamma**self.episode_step) * (self.lambd**self.episode_step)
        
        # Update state, action, episode_return, step
        self.current_state = next_state
        self.current_action = next_action
        self.episode_return += reward
        self.episode_step += 1
        
        # If episode ends or step limit reached
        if done or self.episode_step >= self.step_limit:
            if self.episode_step >= self.step_limit and not done:
                # Remove the current episode from memory if step limit reached
                self.memory = deque([tup for tup in self.memory if tup[0] != self.episode_count], maxlen=10_000)
            
            # Reset for next episode
            self.episode_count += 1
            self.reset_episode()
        
        # Once there are sufficient samples in memory, randomly sample a batch to update Q network
        if len(self.memory) >= self.batch_size:
            batch = random.choices(list(self.memory), k=self.batch_size)
            self.replay(batch, on_policy=True)  # Using on-policy by default

    def replay(self, batch, on_policy=True):
        """Replaying the batch of episodes to train the neural networks"""
        # Unpack batch and convert to required types
        episodes, states, actions, rewards, next_states, next_actions, dones, traces = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        next_actions = torch.tensor(next_actions, dtype=torch.int64).to(self.device)
        dones = torch.tensor(dones, dtype=torch.int16).to(self.device)

        # Get next_q from Q_target
        if on_policy:
            # If acting on policy, get next_q from the next action taken
            next_q = self.Q_target(next_states)
            next_q = next_q.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
        else:
            # If NOT acting on policy, get next_q from the best possible next action
            next_q = self.Q_target(next_states).max(1)[0]

        targets = rewards + (self.gamma * next_q * (1 - dones))

        # Get current_q from Q_main
        current_q = self.Q_main(states)                                     # q values across all possible actions
        current_q = current_q.gather(1, actions.unsqueeze(-1)).squeeze(-1)  # Pick the q value for the corresponding action taken

        # Explode eligibility traces to get the trace for that state
        traces = [torch.tensor(trace, dtype=torch.float32).to(self.device) for trace in traces]
        
        # Handle case where trace lengths might differ
        expanded_q = []
        expanded_targets = []
        
        for trace, q, target in zip(traces, current_q, targets):
            if len(trace) > 0:  # Make sure trace is not empty
                expanded_q.append(torch.mul(trace, q.repeat(len(trace))))
                expanded_targets.append(torch.mul(trace, target.repeat(len(trace))))
        
        if expanded_q:  # Make sure there's at least one valid trace
            current_q = torch.cat(expanded_q)
            targets = torch.cat(expanded_targets)

            # Loss function: q_pred - q_target (where q_target = reward + gamma*next_q)
            loss = nn.MSELoss()(current_q, targets)

            # Update Q_main and perform soft update of Q_target
            self._update_Qmain_weights(loss)

    def save(self, checkpoint_path):
        """Save model weights"""
        torch.save({
            'Q_main': self.Q_main.state_dict(),
            'Q_target': self.Q_target.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, checkpoint_path)
   
    def load(self, checkpoint_path):
        """Load model weights"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.Q_main.load_state_dict(checkpoint['Q_main'])
        self.Q_target.load_state_dict(checkpoint['Q_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer']) 