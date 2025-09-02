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

##### import config #####

import utilities.config as config

##### import necessary libraries #####

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import math
import logging





#############################################
############### AGENT CLASSES ###############
#############################################


########## PPO AGENT ##########

##### actor-critic network class #####

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared_layer_1 = nn.Linear(state_dim, 400)
        self.shared_layer_2 = nn.Linear(400, 300)
        
        # Actor head (policy) - outputs action means and log standard deviations
        # For 36D actions: [mid_angles(12) + target_angles(12) + velocities(12) in rad/s]
        self.actor_mean = nn.Linear(300, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))  # Fixed log std for stability
        
        # Critic head (value function)
        self.critic = nn.Linear(300, 1)
        
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize weights for better training stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        # Shared layers
        shared = F.relu(self.shared_layer_1(state))
        shared = F.relu(self.shared_layer_2(shared))
        
        # Actor outputs
        action_mean = torch.tanh(self.actor_mean(shared)) * self.max_action
        action_logstd = self.actor_logstd.expand_as(action_mean)
        
        # Critic output
        value = self.critic(shared)
        
        return action_mean, action_logstd, value
    
    def get_action_and_value(self, state, action=None):
        """Get action distribution and optionally sample an action"""
        action_mean, action_logstd, value = self.forward(state)
        
        # Create action distribution
        action_std = torch.exp(action_logstd)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            # Sample action during training
            action = action_dist.sample()
        
        # Calculate log probability
        log_prob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Calculate entropy for exploration
        entropy = action_dist.entropy().sum(dim=-1, keepdim=True)
        
        return action, log_prob, entropy, value
    
    def get_value(self, state):
        """Get value function estimate"""
        _, _, value = self.forward(state)
        return value


##### PPO algorithm class #####

class PPO:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_ratio=0.2, target_kl=0.01, train_pi_iters=80, train_v_iters=80, 
                 target_entropy=0.01):
        self.actor_critic = ActorCritic(state_dim, action_dim, max_action)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_entropy = target_entropy
        
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Validate model dimensions to prevent dynamic shapes
        self._validate_dimensions()
    
    def _validate_dimensions(self):
        """Validate that all model components have consistent dimensions."""
        # Test with dummy data to ensure shapes are correct
        dummy_state = torch.randn(1, self.state_dim)
        
        # Test actor-critic
        action_mean, action_logstd, value = self.actor_critic(dummy_state)
        if action_mean.shape[1] != self.action_dim:
            raise ValueError(f"Actor output dimension mismatch: expected {self.action_dim}, got {action_mean.shape[1]}")
        if action_logstd.shape[1] != self.action_dim:
            raise ValueError(f"Actor logstd dimension mismatch: expected {self.action_dim}, got {action_logstd.shape[1]}")
        if value.shape[1] != 1:
            raise ValueError(f"Critic output dimension mismatch: expected 1, got {value.shape[1]}")
        
        print(f"✅ PPO Agent validated: State={self.state_dim}D, Action={self.action_dim}D")

    def select_action(self, state, deterministic=False):
        """Select action from policy - deterministic for deployment, stochastic for training"""
        # Ensure state is exactly the expected dimension
        if len(state) != self.state_dim:
            raise ValueError(f"State dimension mismatch: expected {self.state_dim}, got {len(state)}")
        
        # Convert to tensor with explicit shape to prevent dynamic shapes
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension: (state_dim,) -> (1, state_dim)
        
        with torch.no_grad():
            if deterministic:
                # For deployment: use mean action (no sampling)
                action_mean, _, _ = self.actor_critic(state)
                action = action_mean
            else:
                # For training: sample from distribution
                action, _, _, _ = self.actor_critic.get_action_and_value(state)
        
        # Ensure action has exactly the expected shape
        if action.shape[1] != self.action_dim:
            raise ValueError(f"Action dimension mismatch: expected {self.action_dim}, got {action.shape[1]}")
        
        # Convert to numpy with explicit shape
        action_np = action.cpu().data.numpy()
        
        # Return action with guaranteed shape: (action_dim,)
        return action_np[0, :]  # Remove batch dimension, keep action dimension

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = next_value
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_advantage = delta
            else:
                delta = rewards[t] + self.gamma * last_value - values[t]
                last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
            
            advantages[t] = last_advantage
            last_value = values[t]
        
        returns = advantages + values
        return advantages, returns

    def train(self, states, actions, old_log_probs, rewards, dones, next_value):
        """Train PPO agent on collected episode data"""
        # Convert to tensors with explicit shapes to prevent dynamic shapes
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Validate tensor shapes
        if states.shape[1] != self.state_dim:
            raise ValueError(f"Training state dimension mismatch: expected {self.state_dim}, got {states.shape[1]}")
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"Training action dimension mismatch: expected {self.action_dim}, got {actions.shape[1]}")
        
        # Compute advantages and returns
        with torch.no_grad():
            values = self.actor_critic.get_value(states).squeeze(-1)
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy optimization
        for _ in range(self.train_pi_iters):
            _, log_probs, entropy, _ = self.actor_critic.get_action_and_value(states, actions)
            
            # PPO ratio
            ratio = torch.exp(log_probs - old_log_probs)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            # Policy loss
            policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
            
            # Entropy bonus for exploration
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + 0.01 * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
            
            # Early stopping if KL divergence is too high
            with torch.no_grad():
                kl_div = (old_log_probs - log_probs).mean()
                if kl_div > self.target_kl:
                    break
        
        # Value function optimization
        for _ in range(self.train_v_iters):
            value = self.actor_critic.get_value(states).squeeze(-1)
            value_loss = F.mse_loss(value, returns)
            
            self.optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()


########## STANDARD AGENT ##########

#TODO create a standard agent when the PPO agent is complete


########## ADENT FUNCTIONS ##########

##### initialize agent data #####

def initialize_agent_data():
    """Initialize only the agent data structures without reinitializing PPO policy"""
    
    logging.debug("(training.py): Initializing agent data structures...\n")
    
    # Initialize per-agent data
    num_robots = config.MULTI_ROBOT_CONFIG['num_robots']
    agent_data = {}
    
    for robot_id in range(num_robots):
        agent_data[robot_id] = {
            'episode_counter': 0,
            'episode_reward': 0.0,
            'episode_states': [],
            'episode_actions': [],
            'episode_rewards': [],
            'episode_values': [],
            'episode_log_probs': [],
            'episode_dones': [],
            'episode_scores': [],
            'average_score': 0.0,
            'episode_needs_reset': False
        }
    
    logging.debug(f"Agent data initialized for {num_robots} robots\n")

    return agent_data
