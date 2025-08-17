##################################################################################
# Copyright (c) 2024 Matthew Thomas Beck                                         #
#                                                                                #
# All rights reserved. This code and its associated files may not be reproduced, #
# modified, distributed, or otherwise used, in part or in whole, by any person   #
# or entity without the express written permission of the copyright holder,      #
# Matthew Thomas Beck.                                                           #
##################################################################################





############################################################
############### IMPORT / CREATE DEPENDENCIES ###############
############################################################


########## IMPORT DEPENDENCIES ##########

##### import necessary libraries #####

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random





#############################################
############### AGENT CLASSES ###############
#############################################


########## BLIND AGENT ##########

##### actor class #####

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.layer_1(state))
        a = F.relu(self.layer_2(a))
        a = torch.tanh(self.layer_3(a)) * self.max_action
        return a


##### critic class #####

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = F.relu(self.layer_1(state_action))
        q = F.relu(self.layer_2(q))
        q = self.layer_3(q)
        return q


##### TD3 algorithm class #####

class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=3e-4)

        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=3e-4)

        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Validate model dimensions to prevent dynamic shapes
        self._validate_dimensions()
    
    def _validate_dimensions(self):
        """Validate that all model components have consistent dimensions."""
        # Test with dummy data to ensure shapes are correct
        dummy_state = torch.randn(1, self.state_dim)
        dummy_action = torch.randn(1, self.action_dim)
        
        # Test actor
        actor_output = self.actor(dummy_state)
        if actor_output.shape[1] != self.action_dim:
            raise ValueError(f"Actor output dimension mismatch: expected {self.action_dim}, got {actor_output.shape[1]}")
        
        # Test critic
        critic_output = self.critic_1(dummy_state, dummy_action)
        if critic_output.shape[1] != 1:
            raise ValueError(f"Critic output dimension mismatch: expected 1, got {critic_output.shape[1]}")
        
        print(f"✅ TD3 Agent validated: State={self.state_dim}D, Action={self.action_dim}D")

    def select_action(self, state):
        # Ensure state is exactly the expected dimension
        if len(state) != self.state_dim:
            raise ValueError(f"State dimension mismatch: expected {self.state_dim}, got {len(state)}")
        
        # Convert to tensor with explicit shape to prevent dynamic shapes
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension: (state_dim,) -> (1, state_dim)
        
        # Get action from actor
        action = self.actor(state)
        
        # Ensure action has exactly the expected shape
        if action.shape[1] != self.action_dim:
            raise ValueError(f"Action dimension mismatch: expected {self.action_dim}, got {action.shape[1]}")
        
        # Convert to numpy with explicit shape
        action_np = action.cpu().data.numpy()
        
        # Return action with guaranteed shape: (action_dim,)
        return action_np[0, :]  # Remove batch dimension, keep action dimension

    def train(self, replay_buffer, batch_size=100):
        if len(replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Validate tensor shapes to prevent dynamic shapes
        if state.shape[1] != self.state_dim:
            raise ValueError(f"Training state dimension mismatch: expected {self.state_dim}, got {state.shape[1]}")
        if action.shape[1] != self.action_dim:
            raise ValueError(f"Training action dimension mismatch: expected {self.action_dim}, got {action.shape[1]}")
        
        # Convert to tensors with explicit shapes
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        
        # Ensure reward has correct shape (batch_size, 1) without dynamic reshape
        reward_tensor = torch.FloatTensor(reward)
        if reward_tensor.ndim == 1:
            reward_tensor = reward_tensor.unsqueeze(1)  # (batch_size,) -> (batch_size, 1)
        reward = reward_tensor
        
        next_state = torch.FloatTensor(next_state)
        
        # Ensure done has correct shape (batch_size, 1) without dynamic reshape
        done_tensor = torch.FloatTensor(done)
        if done_tensor.ndim == 1:
            done_tensor = done_tensor.unsqueeze(1)  # (batch_size,) -> (batch_size, 1)
        done = done_tensor

        # Critic update
        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * 0.99 * target_Q

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        critic_1_loss = F.mse_loss(current_Q1, target_Q)
        critic_2_loss = F.mse_loss(current_Q2, target_Q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Actor update
        actor_loss = -self.critic_1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target update
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)


##### replay buffer class #####

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        # Validate data shapes before adding to buffer to prevent dynamic shapes
        if not isinstance(state, (list, np.ndarray)) or len(state) == 0:
            raise ValueError(f"Invalid state: must be non-empty list/array, got {type(state)} with length {len(state) if hasattr(state, '__len__') else 'unknown'}")
        
        if not isinstance(action, (list, np.ndarray)) or len(action) == 0:
            raise ValueError(f"Invalid action: must be non-empty list/array, got {type(action)} with length {len(action) if hasattr(action, '__len__') else 'unknown'}")
        
        if not isinstance(reward, (int, float, np.number)):
            raise ValueError(f"Invalid reward: must be numeric, got {type(reward)}")
        
        if not isinstance(next_state, (list, np.ndarray)) or len(next_state) == 0:
            raise ValueError(f"Invalid next_state: must be non-empty list/array, got {type(next_state)} with length {len(next_state) if hasattr(next_state, '__len__') else 'unknown'}")
        
        if not isinstance(done, (bool, int, np.bool_, np.integer)):
            raise ValueError(f"Invalid done: must be boolean/integer, got {type(done)}")
        
        # Ensure consistent data types
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.array([float(reward)], dtype=np.float32)  # Store as 1D array to prevent shape issues
        next_state = np.array(next_state, dtype=np.float32)
        done = np.array([bool(done)], dtype=np.float32)  # Store as 1D array to prevent shape issues
        
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert to numpy arrays with explicit dtype and shape validation
        # This prevents dynamic shapes and ragged arrays
        state_array = np.array(state, dtype=np.float32)
        action_array = np.array(action, dtype=np.float32)
        reward_array = np.array(reward, dtype=np.float32)
        next_state_array = np.array(next_state, dtype=np.float32)
        done_array = np.array(done, dtype=np.float32)
        
        # Validate array shapes to prevent dynamic shapes
        if state_array.ndim != 2 or state_array.shape[1] == 0:
            raise ValueError(f"Invalid state array shape: {state_array.shape}")
        if action_array.ndim != 2 or action_array.shape[1] == 0:
            raise ValueError(f"Invalid action array shape: {action_array.shape}")
        # Reward can be (batch_size,) or (batch_size, 1) - both are valid
        if reward_array.ndim not in [1, 2] or reward_array.shape[0] == 0:
            raise ValueError(f"Invalid reward array shape: {reward_array.shape}")
        if next_state_array.ndim != 2 or next_state_array.shape[1] == 0:
            raise ValueError(f"Invalid next_state array shape: {next_state_array.shape}")
        # Done can be (batch_size,) or (batch_size, 1) - both are valid
        if done_array.ndim not in [1, 2] or done_array.shape[0] == 0:
            raise ValueError(f"Invalid done array shape: {done_array.shape}")
        
        return state_array, action_array, reward_array, next_state_array, done_array

    def __len__(self):
        return len(self.buffer)


########## STANDARD AGENT ##########

#TODO create a standard agent when the blind agent is complete
