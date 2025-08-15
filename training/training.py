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
import os
import logging
from collections import deque
import random
from scipy.spatial.transform import Rotation

##### reinforcement learning libraries #####

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

##### movement functions #####

from movement.movement_coordinator import neutral_position

##### import config #####

import utilities.config as config

########## CREATE DEPENDENCIES ##########

##### global TD3 instance and replay buffer #####

td3_policy = None
replay_buffer = None
episode_step = 0
episode_reward = 0.0
episode_counter = 0
last_state = None
last_action = None
total_steps = 0

# Track average episode scores for model naming
episode_scores = []  # List of all episode final scores
average_score = 0.0  # Running average of episode scores

# COMPLETELY GUTTED - No more forward movement tracking
# You now have complete control over movement detection and rewards

##### training config ##### TODO move me to config.py when ready

TRAINING_CONFIG = {
    'max_episodes': 1000000,
    'max_steps_per_episode': 750,  # GPT-5 recommendation: 600-1200 steps (~10-20 seconds)
    'save_frequency': 20000,  # Save model every 20,000 steps (more frequent saves)
    'training_frequency': 2,  # Train every 2 steps (GPT-5: more frequent training)
    'batch_size': 64,  # GPT-5 recommendation: standard batch size
    'learning_rate': 3e-4,  # Back to standard learning rate
    'gamma': 0.99,  # Discount factor
    'tau': 0.005,  # Standard target network update rate
    'exploration_noise': 0.1,  # Standard exploration noise
    'max_action': 1.0
}

##### models derectory #####

models_dir = "/home/matthewthomasbeck/Projects/Robot_Dog/model"





##################################################
############### ISAAC SIM TRAINING ###############
##################################################


########## BLIND RL AGENT INTERFACES ##########

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

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        if len(replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).reshape(-1, 1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).reshape(-1, 1)

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
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)


########## MAIN LOOP ##########

##### initialize training #####

def initialize_training():
    """Initialize the complete training system"""
    global td3_policy, replay_buffer, episode_counter, total_steps

    print("Initializing training system...")
    os.makedirs(models_dir, exist_ok=True)

    # Initialize TD3
    state_dim = 21  # 12 joints + 8 commands + 1 intensity
    action_dim = 24  # 12 mid + 12 target angles
    max_action = TRAINING_CONFIG['max_action']

    td3_policy = TD3(state_dim, action_dim, max_action)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=100000)

    # Try to load the latest saved model to continue training
    latest_model = find_latest_model()
    if latest_model:
        print(f"🔄 Loading latest model: {latest_model}")
        if load_model(latest_model):
            print(f"✅ Successfully loaded model from step {total_steps}, episode {episode_counter}")
        else:
            print(f"❌ Failed to load model, starting fresh")
            episode_counter = 0
            total_steps = 0
    else:
        print(f"🆕 No saved models found, starting fresh training")
        episode_counter = 0
        total_steps = 0

    print(f"Training system initialized:")
    print(f"  - State dimension: {state_dim}")
    print(f"  - Action dimension: {action_dim}")
    print(f"  - Models directory: {models_dir}")
    print(f"  - TD3 policy created: {td3_policy is not None}")
    print(f"  - Replay buffer created: {replay_buffer is not None}")
    print(f"  - Starting from episode: {episode_counter}")
    print(f"  - Starting from step: {total_steps}")


def find_latest_model():
    """Find the latest saved model file"""
    import glob
    import re
    
    # Look for model files in the models directory
    model_pattern = os.path.join(models_dir, "td3_steps_*_episode_*_reward_*.pth")
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        return None
    
    # Extract step numbers and find the latest
    latest_model = None
    latest_steps = 0
    
    for model_file in model_files:
        # Extract step number from filename like "td3_steps_40000_episode_103_reward_74.27.pth"
        match = re.search(r'td3_steps_(\d+)_episode_', os.path.basename(model_file))
        if match:
            steps = int(match.group(1))
            if steps > latest_steps:
                latest_steps = steps
                latest_model = model_file
    
    return latest_model


##### integrate with main loop #####

def integrate_with_main_loop():
    """
    This function should be called from your main loop to integrate episode management.
    Call this function after each simulation step to check for episode resets.

    Usage in main loop:
    if config.USE_SIMULATION and config.USE_ISAAC_SIM:
        from training.training import integrate_with_main_loop
        integrate_with_main_loop()
    """
    global episode_step, episode_counter, total_steps

    import utilities.config as config

    # Check if worker thread has signaled that episode needs reset
    if hasattr(config, 'EPISODE_NEEDS_RESET') and config.EPISODE_NEEDS_RESET:
        # Clear the signal
        config.EPISODE_NEEDS_RESET = False

        # Perform the actual reset in the main thread (safe for Isaac Sim)
        print(f"Main thread: Episode reset signal received, performing reset...")
        reset_episode()
        return True

    # Also check directly if episode should be reset (backup check)
    if check_and_reset_episode_if_needed():
        # Episode was reset, log the event
        import logging
        logging.info(f"(training.py): Episode {episode_counter} reset in main loop integration\n")
        return True

    return False


########## AGENT FUNCTIONS ##########

##### summon standard agent #####

def get_rl_action_standard(state, commands, intensity, frame):  # will eventually take camera frame as input
    pass


##### summon blind agent #####

def get_rl_action_blind(current_angles, commands, intensity):
    """
    TD3 RL agent that takes current joint angles, commands, and intensity as state
    and outputs 24 values (12 mid angles + 12 target angles)

    NOTE: This function runs in worker threads, so it CANNOT call Isaac Sim reset functions.
    It only signals that a reset is needed - the main thread handles actual resets.
    """
    global td3_policy, replay_buffer, episode_step, episode_reward, episode_counter
    global last_state, last_action, total_steps

    # Initialize training system if not done yet
    if td3_policy is None:
        initialize_training()
        start_episode()

    # CRITICAL: Check episode completion but DON'T reset from worker thread
    # Just signal that a reset is needed - main thread will handle it
    episode_needs_reset = False

    # COMPLETELY GUTTED - No more automatic episode termination due to falling
    # You now control when episodes end through your reward system

    if episode_step >= TRAINING_CONFIG['max_steps_per_episode']:
        episode_needs_reset = True
        print(f"Episode {episode_counter} completed after {episode_step} steps! (signaling main thread)")

    # Store reset signal for main thread to check
    config.EPISODE_NEEDS_RESET = episode_needs_reset

    # Log episode progress every 100 steps
    if episode_step % 100 == 0:
        print(f"Episode {episode_counter}, Step {episode_step}, Total Steps: {total_steps}")
    
    # Track orientation every 50 steps to understand robot facing direction
    if episode_step % 50 == 0:
        track_orientation()

    # Build state vector
    state = []

    # 1. Extract joint angles (12D)
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        for joint_name in ['hip', 'upper', 'lower']:
            angle = current_angles[leg_id][joint_name]
            state.append(float(angle))

    # 2. Encode commands (8D one-hot)
    if isinstance(commands, list):
        command_list = commands
    elif isinstance(commands, str):
        command_list = commands.split('+') if commands else []
    else:
        command_list = []

    command_encoding = [
        1.0 if 'w' in command_list else 0.0,
        1.0 if 's' in command_list else 0.0,
        1.0 if 'a' in command_list else 0.0,
        1.0 if 'd' in command_list else 0.0,
        1.0 if 'arrowleft' in command_list else 0.0,
        1.0 if 'arrowright' in command_list else 0.0,
        1.0 if 'arrowup' in command_list else 0.0,
        1.0 if 'arrowdown' in command_list else 0.0
    ]
    state.extend(command_encoding)

    # 3. Normalize intensity (1D)
    intensity_normalized = float(intensity) / 10.0
    state.append(intensity_normalized)
    state = np.array(state, dtype=np.float32)

    # Get action from TD3 (add exploration noise in early episodes)
    add_noise = episode_counter < 100  # Add noise for first 100 episodes
    
    # COMPLETELY GUTTED - No more automatic fall tracking and exploration forcing
    # You now control exploration and fall recovery through your reward system
    
    if add_noise:
        action = td3_policy.select_action(state)
        noise = np.random.normal(0, TRAINING_CONFIG['exploration_noise'], size=action.shape)
        action = np.clip(action + noise, -TRAINING_CONFIG['max_action'], TRAINING_CONFIG['max_action'])
    else:
        action = td3_policy.select_action(state)
    
    # COMPLETELY GUTTED - No more automatic fall tracking
    # You now control fall recovery through your reward system

    # Store experience for training
    if last_state is not None and last_action is not None:
        # Calculate reward using the dedicated reward function
        reward = calculate_step_reward(current_angles, commands, intensity)

        # Update episode reward for tracking
        episode_reward += reward

        # COMPLETELY GUTTED - No more automatic episode termination due to falling
        # You now control when episodes end through your reward system
        done = episode_step >= TRAINING_CONFIG['max_steps_per_episode']

        # CRITICAL: Add to replay buffer BEFORE checking for episode reset
        # This ensures the falling experience is captured
        replay_buffer.add(last_state, last_action, reward, state, done)
        
        # Log experience collection for debugging
        if done:
            print(f"💾 Experience collected: State={len(last_state)}D, Action={len(last_action)}D, Reward={reward:.3f}, Done={done}")
            print(f"   📊 Replay buffer size: {len(replay_buffer)}")

        # Train TD3 periodically
        if total_steps % TRAINING_CONFIG['training_frequency'] == 0 and len(replay_buffer) >= TRAINING_CONFIG[
            'batch_size']:
            # Train the agent and log the experience (reduced frequency to avoid spam)
            if total_steps % 50 == 0:  # Only log every 50 steps
                print(f"🧠 Training TD3 at step {total_steps}, buffer size: {len(replay_buffer)}")
            td3_policy.train(replay_buffer, TRAINING_CONFIG['batch_size'])
            
            # COMPLETELY GUTTED - No more automatic fall-related logging
            # You now control what gets logged through your reward system

        # Save model periodically based on total steps
        if total_steps % TRAINING_CONFIG['save_frequency'] == 0 and td3_policy is not None:
            save_model(
                f"/home/matthewthomasbeck/Projects/Robot_Dog/model/td3_steps_{total_steps}_episode_{episode_counter}_reward_{episode_reward:.2f}.pth")
            print(f"💾 Model saved: steps_{total_steps}, episode_{episode_counter}")

    # Update tracking variables
    last_state = state.copy()
    last_action = action.copy()
    episode_step += 1
    total_steps += 1

    # Convert action (-1 to 1) to joint angles
    target_angles = {}
    mid_angles = {}
    movement_rates = {}

    action_idx = 0

    for leg_id in ['FL', 'FR', 'BL', 'BR']:

        target_angles[leg_id] = {}
        mid_angles[leg_id] = {}
        movement_rates[leg_id] = {'speed': 1.0, 'acceleration': 0.5}

        for joint_name in ['hip', 'upper', 'lower']:

            # Get joint limits from config
            servo_data = config.SERVO_CONFIG[leg_id][joint_name]
            min_angle = servo_data['FULL_BACK_ANGLE']
            max_angle = servo_data['FULL_FRONT_ANGLE']

            # Ensure correct order
            if min_angle > max_angle:
                min_angle, max_angle = max_angle, min_angle

            # Convert mid action (-1 to 1) to joint angle
            mid_action = action[action_idx]
            mid_angle = min_angle + (mid_action + 1.0) * 0.5 * (max_angle - min_angle)
            mid_angle = np.clip(mid_angle, min_angle, max_angle)
            mid_angles[leg_id][joint_name] = float(mid_angle)

            # Convert target action (-1 to 1) to joint angle
            target_action = action[action_idx + 12]  # Target angles are second half
            target_angle = min_angle + (target_action + 1.0) * 0.5 * (max_angle - min_angle)
            target_angle = np.clip(target_angle, min_angle, max_angle)
            target_angles[leg_id][joint_name] = float(target_angle)

            action_idx += 1

    return target_angles, mid_angles, movement_rates


########## MOVEMENT DETECTION FUNCTIONS ##########

##### reward function #####

def calculate_step_reward(current_angles, commands, intensity):
    """
    Reward system that rewards balance, proper height, and correct movement execution.
    Falls result in massive penalty and episode termination.
    
    This function is called after get_rl_action_blind() in the while True loop in control_logic.
    
    Args:
        current_angles: Current joint angles for all legs
        commands: Movement commands (e.g., 'w', 's', 'a', 'd', 'w+a+arrowleft')
        intensity: Movement intensity (1-10) (ignored for now)

    Returns:
        float: Reward value for this step
    """
    # Get fresh position and orientation data
    center_pos, facing_deg = track_orientation()

    if center_pos is None:
        return 0.0  # Can't calculate reward without position data
    
    # Initialize reward and perfect execution tracker
    reward = 0.0
    was_perfect = True  # Start assuming perfect execution, set to False if any issues found

    # 0. FALL DETECTION: Massive penalty and episode termination if robot falls
    # Check if robot has fallen based on height and balance
    current_height = center_pos[2]
    
    # Get actual off_balance from track_orientation data
    if hasattr(track_orientation, 'last_off_balance'):
        current_balance = track_orientation.last_off_balance
    else:
        current_balance = 0.0
    
    # Robot has fallen if:
    # - Height is too low (lying on ground)
    # - Balance is too extreme (tipped over)
    has_fallen = current_balance > 90.0

    # 1. BALANCE REWARD: Reward near 0°, punish near 90°, ignore middle ground
    # Target: 0° (perfect balance), Bad: 90° (tipped over)
    # 33% tolerance: ±30° from 0° and ±30° from 90°
    
    if current_balance < 30.0:  # Within 33% of perfect balance (0° to 30°)
        # Reward being close to perfect balance
        if current_balance < 3.0:  # Within 10% of perfect (0° ± 3°)
            balance_reward = 1.0
            print(f"🎯 PERFECT BALANCE! +1.0 reward - Balance: {current_balance:.1f}°")
        else:  # Within 33% of perfect (3° to 30°)
            # Logarithmic scaling: 0.1 at 33% error, 1.0 at 10% error
            balance_progress = 1.0 - (current_balance / 30.0) ** 2
            balance_reward = 0.1 + 0.9 * balance_progress
            print(f"🎯 GOOD BALANCE: +{balance_reward:.2f} reward - Balance: {current_balance:.1f}°")
            was_perfect = False
    
    elif current_balance > 60.0:  # Within 33% of bad balance (60° to 90°)
        # Punish being close to tipped over
        if current_balance > 87.0:  # Within 10% of bad (87° to 90°)
            balance_reward = -1.0  # Maximum penalty for being tipped over
            print(f"❌ TIPPED OVER: {balance_reward:.1f} penalty - Balance: {current_balance:.1f}°")
            was_perfect = False
        else:  # Within 33% of bad (60° to 87°)
            # Logarithmic scaling: -0.1 at 33% from bad, -1.0 at 10% from bad
            balance_progress = 1.0 - ((90.0 - current_balance) / 30.0) ** 2
            balance_reward = -0.1 - 0.9 * balance_progress
            print(f"❌ POOR BALANCE: {balance_reward:.2f} penalty - Balance: {current_balance:.1f}°")
            was_perfect = False
    
    else:  # Middle ground (30° to 60°) - no reward, no penalty
        balance_reward = 0.0
        print(f"📊 MIDDLE BALANCE: No reward/penalty - Balance: {current_balance:.1f}°")
        was_perfect = False
    
    reward += balance_reward
    
    # 2. HEIGHT REWARD: Reward being at the correct height (0.129m), punish being too low
    # Target: 0.129m (perfect height), Bad: 0.0m (lying on ground)
    # 33% zones: Good (0.129m to 0.086m), Neutral (0.086m to 0.043m), Bad (0.043m to 0.0m)
    
    if current_height > 0.086:  # Within 33% of perfect height (0.086m to 0.129m)
        # Reward being close to perfect height
        if current_height > 0.126:  # Within 10% of perfect (0.126m to 0.129m)
            height_reward = 1.0
            print(f"�� PERFECT HEIGHT! +1.0 reward - Height: {current_height:.3f}m")
        else:  # Within 33% of perfect (0.086m to 0.126m)
            # Logarithmic scaling: 0.1 at 33% error, 1.0 at 10% error
            height_progress = 1.0 - ((0.129 - current_height) / 0.043) ** 2
            height_reward = 0.1 + 0.9 * height_progress
            print(f"🎯 GOOD HEIGHT: +{height_reward:.2f} reward - Height: {current_height:.3f}m")
            was_perfect = False
    
    elif current_height < 0.043:  # Within 33% of bad height (0.0m to 0.043m)
        # Punish being close to lying on ground
        if current_height < 0.003:  # Within 10% of bad (0.0m to 0.003m)
            height_reward = -1.0  # Maximum penalty for lying on ground
            print(f"❌ LYING ON GROUND: {height_reward:.1f} penalty - Height: {current_height:.3f}m")
            was_perfect = False
        else:  # Within 33% of bad (0.003m to 0.043m)
            # Logarithmic scaling: -0.1 at 33% from bad, -1.0 at 10% from bad
            height_progress = 1.0 - (current_height / 0.043) ** 2
            height_reward = -0.1 - 0.9 * height_progress
            print(f"❌ POOR HEIGHT: {height_reward:.2f} penalty - Height: {current_height:.3f}m")
            was_perfect = False
    
    else:  # Middle ground (0.043m to 0.086m) - no reward, no penalty
        height_reward = 0.0
        print(f"📊 MIDDLE HEIGHT: No reward/penalty - Height: {current_height:.3f}m")
        was_perfect = False
    
    reward += height_reward
    
    # 3. MOVEMENT REWARD: Reward moving in the correct direction, punish ALL wrong directions
    if commands:
        # Parse complex commands like 'w+a+arrowleft'
        command_list = commands.split('+') if isinstance(commands, str) else commands
        
        # Get movement data from the last call to track_orientation
        try:
            if hasattr(track_orientation, 'last_movement_data'):
                movement_data = track_orientation.last_movement_data
                
                # Check rotation first - must be minimal to get any movement reward
                rotation_during_movement = abs(track_orientation.last_rotation) if hasattr(track_orientation, 'last_rotation') else 0
                
                if rotation_during_movement < 2.0:  # Acceptable rotation for all directions
                    # Check each movement command and reward correct execution, punish wrong directions
                    for cmd in command_list:
                        if cmd == 'w':  # Forward movement commanded
                            forward_movement = movement_data.get('w', 0)
                            left_movement = movement_data.get('a', 0)
                            right_movement = movement_data.get('d', 0)
                            backward_movement = movement_data.get('s', 0)
                            
                            # Reward forward movement (commanded direction)
                            if forward_movement > 3.33:  # Within 33% of target (3.33cm to 5cm)
                                if forward_movement > 4.5:  # Within 10% of target (4.5cm to 5cm)
                                    movement_reward = 3.0
                                    print(f"�� PERFECT FORWARD: +{movement_reward:.1f} reward - Forward: {forward_movement:.3f}m")
                                else:  # Within 33% of target (3.33cm to 4.5cm)
                                    movement_progress = 1.0 - ((5.0 - forward_movement) / 1.67) ** 2
                                    movement_reward = 0.3 + 2.7 * movement_progress
                                    print(f"🎯 GOOD FORWARD: +{movement_reward:.2f} reward - Forward: {forward_movement:.3f}m")
                                    was_perfect = False
                            elif forward_movement < 0.0:  # Moving backward (wrong direction)
                                if forward_movement < -4.5:  # Within 10% of bad (-4.5cm to -5cm)
                                    movement_reward = -3.0
                                    print(f"❌ PERFECT BACKWARD: {movement_reward:.1f} penalty - Forward: {forward_movement:.3f}m")
                                    was_perfect = False
                                else:  # Within 33% of bad (0cm to -4.5cm)
                                    movement_progress = max(0.0, min(1.0, 1.0 - ((forward_movement + 5.0) / 4.5) ** 2))
                                    movement_reward = -0.3 - 2.7 * movement_progress
                                    print(f"❌ POOR FORWARD: {movement_reward:.2f} penalty - Forward: {forward_movement:.3f}m")
                                    was_perfect = False
                            else:  # Middle ground (0cm to 3.33cm) - no reward, no penalty
                                movement_reward = 0.0
                                print(f"�� MIDDLE FORWARD: No reward/penalty - Forward: {forward_movement:.3f}m")
                                was_perfect = False
                            
                            reward += movement_reward
                            
                            # PUNISH movement in wrong directions (even if not commanded)
                            if abs(left_movement) > 0.5:  # Moving left when should go forward
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving LEFT: {left_movement:.3f}m when FORWARD commanded")
                                was_perfect = False
                            
                            if abs(right_movement) > 0.5:  # Moving right when should go forward
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving RIGHT: {right_movement:.3f}m when FORWARD commanded")
                                was_perfect = False
                            
                            if abs(backward_movement) > 0.5:  # Moving backward when should go forward
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving BACKWARD: {backward_movement:.3f}m when FORWARD commanded")
                                was_perfect = False
                        
                        elif cmd == 's':  # Backward movement commanded
                            # Similar logic for backward - reward backward, punish forward/left/right
                            backward_movement = movement_data.get('s', 0)
                            forward_movement = movement_data.get('w', 0)
                            left_movement = movement_data.get('a', 0)
                            right_movement = movement_data.get('d', 0)
                            
                            # Reward backward movement (commanded direction)
                            if backward_movement > 3.33:  # Within 33% of target (3.33cm to 5cm)
                                if backward_movement > 4.5:  # Within 10% of target (4.5cm to 5cm)
                                    movement_reward = 3.0
                                    print(f"🎯 PERFECT BACKWARD: +{movement_reward:.1f} reward - Backward: {backward_movement:.3f}m")
                                else:  # Within 33% of target (3.33cm to 4.5cm)
                                    movement_progress = 1.0 - ((5.0 - backward_movement) / 1.67) ** 2
                                    movement_reward = 0.3 + 2.7 * movement_progress
                                    print(f"🎯 GOOD BACKWARD: +{movement_reward:.2f} reward - Backward: {backward_movement:.3f}m")
                                    was_perfect = False
                            elif backward_movement < 0.0:  # Moving forward (wrong direction)
                                if backward_movement < -4.5:  # Within 10% of bad (-4.5cm to -5cm)
                                    movement_reward = -3.0
                                    print(f"❌ PERFECT FORWARD: {movement_reward:.1f} penalty - Backward: {backward_movement:.3f}m")
                                    was_perfect = False
                                else:  # Within 33% of bad (0cm to -4.5cm)
                                    movement_progress = max(0.0, min(1.0, 1.0 - ((backward_movement + 5.0) / 4.5) ** 2))
                                    movement_reward = -0.3 - 2.7 * movement_progress
                                    print(f"❌ POOR BACKWARD: {movement_reward:.2f} penalty - Backward: {backward_movement:.3f}m")
                                    was_perfect = False
                            else:  # Middle ground (0cm to 3.33cm) - no reward, no penalty
                                movement_reward = 0.0
                                print(f"📊 MIDDLE BACKWARD: No reward/penalty - Backward: {backward_movement:.3f}m")
                                was_perfect = False
                            
                            reward += movement_reward
                            
                            # PUNISH movement in wrong directions
                            if abs(forward_movement) > 0.5:  # Moving forward when should go backward
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving FORWARD: {forward_movement:.3f}m when BACKWARD commanded")
                                was_perfect = False
                            
                            if abs(left_movement) > 0.5:  # Moving left when should go backward
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving LEFT: {left_movement:.3f}m when BACKWARD commanded")
                                was_perfect = False
                            
                            if abs(right_movement) > 0.5:  # Moving right when should go backward
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving RIGHT: {right_movement:.3f}m when BACKWARD commanded")
                                was_perfect = False
                        
                        elif cmd == 'a':  # Left movement commanded
                            # Similar logic for left - reward left, punish forward/backward/right
                            left_movement = movement_data.get('a', 0)
                            forward_movement = movement_data.get('w', 0)
                            backward_movement = movement_data.get('s', 0)
                            right_movement = movement_data.get('d', 0)
                            
                            # Reward left movement (commanded direction)
                            if left_movement > 3.33:  # Within 33% of target (3.33cm to 5cm)
                                if left_movement > 4.5:  # Within 10% of target (4.5cm to 5cm)
                                    movement_reward = 3.0
                                    print(f"🎯 PERFECT LEFT: +{movement_reward:.1f} reward - Left: {left_movement:.3f}m")
                                else:  # Within 33% of target (3.33cm to 4.5cm)
                                    movement_progress = 1.0 - ((5.0 - left_movement) / 1.67) ** 2
                                    movement_reward = 0.3 + 2.7 * movement_progress
                                    print(f"🎯 GOOD LEFT: +{movement_reward:.2f} reward - Left: {left_movement:.3f}m")
                                    was_perfect = False
                            elif left_movement < 0.0:  # Moving right (wrong direction)
                                if left_movement < -4.5:  # Within 10% of bad (-4.5cm to -5cm)
                                    movement_reward = -3.0
                                    print(f"❌ PERFECT RIGHT: {movement_reward:.1f} penalty - Left: {left_movement:.3f}m")
                                    was_perfect = False
                                else:  # Within 33% of bad (0cm to -4.5cm)
                                    movement_progress = max(0.0, min(1.0, 1.0 - ((left_movement + 5.0) / 4.5) ** 2))
                                    movement_reward = -0.3 - 2.7 * movement_progress
                                    print(f"❌ POOR LEFT: {movement_reward:.2f} penalty - Left: {left_movement:.3f}m")
                                    was_perfect = False
                            else:  # Middle ground (0cm to 3.33cm) - no reward, no penalty
                                movement_reward = 0.0
                                print(f"📊 MIDDLE LEFT: No reward/penalty - Left: {left_movement:.3f}m")
                                was_perfect = False
                            
                            reward += movement_reward
                            
                            # PUNISH movement in wrong directions
                            if abs(forward_movement) > 0.5:  # Moving forward when should go left
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving FORWARD: {forward_movement:.3f}m when LEFT commanded")
                                was_perfect = False
                            
                            if abs(backward_movement) > 0.5:  # Moving backward when should go left
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving BACKWARD: {backward_movement:.3f}m when LEFT commanded")
                                was_perfect = False
                            
                            if abs(right_movement) > 0.5:  # Moving right when should go left
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving RIGHT: {right_movement:.3f}m when LEFT commanded")
                                was_perfect = False
                        
                        elif cmd == 'd':  # Right movement commanded
                            # Similar logic for right - reward right, punish forward/backward/left
                            right_movement = movement_data.get('d', 0)
                            forward_movement = movement_data.get('w', 0)
                            backward_movement = movement_data.get('s', 0)
                            left_movement = movement_data.get('a', 0)
                            
                            # Reward right movement (commanded direction)
                            if right_movement > 3.33:  # Within 33% of target (3.33cm to 5cm)
                                if right_movement > 4.5:  # Within 10% of target (4.5cm to 5cm)
                                    movement_reward = 3.0
                                    print(f"🎯 PERFECT RIGHT: +{movement_reward:.1f} reward - Right: {right_movement:.3f}m")
                                else:  # Within 33% of target (3.33cm to 4.5cm)
                                    movement_progress = 1.0 - ((5.0 - right_movement) / 1.67) ** 2
                                    movement_reward = 0.3 + 2.7 * movement_progress
                                    print(f"🎯 GOOD RIGHT: +{movement_reward:.2f} reward - Right: {right_movement:.3f}m")
                                    was_perfect = False
                            elif right_movement < 0.0:  # Moving left (wrong direction)
                                if right_movement < -4.5:  # Within 10% of bad (-4.5cm to -5cm)
                                    movement_reward = -3.0
                                    print(f"❌ PERFECT LEFT: {movement_reward:.1f} penalty - Right: {right_movement:.3f}m")
                                    was_perfect = False
                                else:  # Within 33% of bad (0cm to -4.5cm)
                                    movement_progress = max(0.0, min(1.0, 1.0 - ((right_movement + 5.0) / 4.5) ** 2))
                                    movement_reward = -0.3 - 2.7 * movement_progress
                                    print(f"❌ POOR RIGHT: {movement_reward:.2f} penalty - Right: {right_movement:.3f}m")
                                    was_perfect = False
                            else:  # Middle ground (0cm to 3.33cm) - no reward, no penalty
                                movement_reward = 0.0
                                print(f"📊 MIDDLE RIGHT: No reward/penalty - Right: {right_movement:.3f}m")
                                was_perfect = False
                            
                            reward += movement_reward
                            
                            # PUNISH movement in wrong directions
                            if abs(forward_movement) > 0.5:  # Moving forward when should go right
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving FORWARD: {forward_movement:.3f}m when RIGHT commanded")
                                was_perfect = False
                            
                            if abs(backward_movement) > 0.5:  # Moving backward when should go right
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving BACKWARD: {backward_movement:.3f}m when RIGHT commanded")
                                was_perfect = False
                            
                            if abs(left_movement) > 0.5:  # Moving left when should go right
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving LEFT: {left_movement:.3f}m when RIGHT commanded")
                                was_perfect = False
                        
                        elif cmd in ['arrowleft', 'arrowright']:  # Rotation commands - handled in rotation section
                            pass  # Skip here, handled in rotation reward section
                
                else:  # Too much rotation during movement
                    reward -= 0.1
                    print(f"❌ EXCESSIVE ROTATION: -0.1 penalty - Rotation: {rotation_during_movement:.1f}°")
                    was_perfect = False
            
        except Exception as e:
            # If movement analysis fails, give neutral reward
            reward += 0.0
    
    # 4. ROTATION CONTROL REWARD: Reward maintaining stable orientation or executing rotation commands
    if hasattr(track_orientation, 'last_rotation'):
        rotation_magnitude = abs(track_orientation.last_rotation)
        rotation_commanded = any(cmd in ['arrowleft', 'arrowright'] for cmd in command_list)
        
        if not rotation_commanded:
            # NO ROTATION COMMANDED: Reward staying stable (minimal rotation)
            # Target: 0° rotation (perfect stability), Bad: 30° rotation (excessive rotation)
            # 33% zones: Good (0° to 10°), Neutral (10° to 20°), Bad (20° to 30°)
            
            if rotation_magnitude < 10.0:  # Within 33% of perfect stability (0° to 10°)
                # Reward being close to perfect stability
                if rotation_magnitude < 1.0:  # Within 10% of perfect (0° to 1°)
                    rotation_reward = 2.0  # Perfect stability
                    print(f"�� PERFECT STABILITY: +1.0 reward - Rotation: {rotation_magnitude:.1f}°")
                else:  # Within 33% of perfect (1° to 10°)
                    # Logarithmic scaling: 0.2 at 33% error, 2.0 at 10% error
                    rotation_progress = 1.0 - (rotation_magnitude / 10.0) ** 2
                    rotation_reward = 0.2 + 1.8 * rotation_progress
                    print(f"�� GOOD STABILITY: +{rotation_reward:.2f} reward - Rotation: {rotation_magnitude:.1f}°")
                    was_perfect = False
            
            elif rotation_magnitude > 20.0:  # Within 33% of bad rotation (20° to 30°)
                # Punish being close to excessive rotation
                if rotation_magnitude > 29.0:  # Within 10% of bad (29° to 30°)
                    rotation_reward = -2.0  # Maximum penalty for excessive rotation
                    print(f"❌ EXCESSIVE ROTATION: {rotation_reward:.1f} penalty - Rotation: {rotation_magnitude:.1f}°")
                    was_perfect = False
                else:  # Within 33% of bad (20° to 29°)
                    # Logarithmic scaling: -0.2 at 33% from bad, -2.0 at 10% from bad
                    rotation_progress = 1.0 - ((30.0 - rotation_magnitude) / 10.0) ** 2
                    rotation_reward = -0.2 - 1.8 * rotation_progress
                    print(f"❌ POOR STABILITY: {rotation_reward:.2f} penalty - Rotation: {rotation_magnitude:.1f}°")
                    was_perfect = False
            
            else:  # Middle ground (10° to 20°) - no reward, no penalty
                rotation_reward = 0.0
                print(f"📊 MIDDLE STABILITY: No reward/penalty - Rotation: {rotation_magnitude:.1f}°")
                was_perfect = False
            
            reward += rotation_reward
            
        else:
            # ROTATION COMMANDED: Reward executing rotation commands properly
            # Target: 30° rotation (rapid rotation), Bad: 0° rotation (no rotation)
            # 33% zones: Good (20° to 30°), Neutral (10° to 20°), Bad (0° to 10°)
            
            if rotation_magnitude > 20.0:  # Within 33% of target rotation (20° to 30°)
                # Reward being close to target rotation
                if rotation_magnitude > 29.0:  # Within 10% of target (29° to 30°)
                    rotation_reward = 2.0  # Perfect rotation execution
                    print(f"🎯 PERFECT ROTATION: +1.0 reward - Rotation: {rotation_magnitude:.1f}°")
                else:  # Within 33% of target (20° to 29°)
                    # Logarithmic scaling: 0.2 at 33% error, 2.0 at 10% error
                    rotation_progress = 1.0 - ((30.0 - rotation_magnitude) / 10.0) ** 2
                    rotation_reward = 0.2 + 1.8 * rotation_progress
                    print(f"🎯 GOOD ROTATION: +{rotation_reward:.2f} reward - Rotation: {rotation_magnitude:.1f}°")
                    was_perfect = False
            
            elif rotation_magnitude < 10.0:  # Within 33% of bad rotation (0° to 10°)
                # Punish being close to no rotation
                if rotation_magnitude < 1.0:  # Within 10% of bad (0° to 1°)
                    rotation_reward = -2.0  # Maximum penalty for no rotation
                    print(f"❌ NO ROTATION: {rotation_reward:.1f} penalty - Rotation: {rotation_magnitude:.1f}°")
                    was_perfect = False
                else:  # Within 33% of bad (1° to 10°)
                    # Logarithmic scaling: -0.2 at 33% from bad, -2.0 at 10% from bad
                    rotation_progress = 1.0 - (rotation_magnitude / 10.0) ** 2
                    rotation_reward = -0.2 - 1.8 * rotation_progress
                    print(f"❌ POOR ROTATION: {rotation_reward:.2f} penalty - Rotation: {rotation_magnitude:.1f}°")
                    was_perfect = False
            
            else:  # Middle ground (10° to 20°) - no reward, no penalty
                rotation_reward = 0.0
                print(f"📊 MIDDLE ROTATION: No reward/penalty - Rotation: {rotation_magnitude:.1f}°")
                was_perfect = False

            reward += rotation_reward
    
    # 5. PERFECT PERFORMANCE BONUS: Massive reward for doing exactly what we want
    # Simple boolean approach: was_perfect starts True, gets set False if any issues found
    
    if was_perfect and commands:  # Only give bonus if we had commands and were perfect
        perfect_bonus = 10.0  # Much bigger than individual movement rewards
        reward += perfect_bonus
        print(f"🏆 PERFECT EXECUTION! +{perfect_bonus:.1f} MASSIVE BONUS - All commands executed flawlessly!")
    elif commands:
        print(f"📊 Good execution, but not perfect - no bonus this time")

    if has_fallen:
        # MASSIVE PENALTY: Standard -100 penalty for falling, episode ends
        fall_penalty = -100
        print(f"EPISODE FAILURE -100 points (robot fell over)")
        
        # Signal that episode should end due to falling
        global episode_step
        episode_step = TRAINING_CONFIG['max_steps_per_episode']  # Force episode end
        
        return fall_penalty
    
    # 5. CLAMP REWARD: Prevent extreme values (but allow fall penalty)
    elif not has_fallen:
        reward = max(-1.0, min(1.0, reward))
    
    return reward

##### orientation tracking #####

def track_orientation():
    """
    Track the robot's position and orientation (facing direction).
    Robot spawns facing -Y (forward), so we measure rotation from that direction.
    """
    global episode_step
    
    try:
        positions, rotations = config.ISAAC_ROBOT.get_world_poses()
        center_pos = positions[0]
        rotation = rotations[0]
        
        # Convert quaternion to euler angles to get yaw (rotation around Z-axis)
        from scipy.spatial.transform import Rotation
        quat_wxyz = [rotation[0], rotation[1], rotation[2], rotation[3]]
        r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        euler_angles = r.as_euler('xyz', degrees=True)
        yaw_deg = euler_angles[2]  # Yaw is rotation around Z-axis
        
        # Robot spawns facing -Y (forward), so we need to calculate relative rotation
        # -Y direction = 0 degrees
        # Left rotation = positive degrees (10° = facing slightly left of -Y)
        # Right rotation = negative degrees, but we convert to 360° system
        
        # Convert to 0-360 degree system where 0° = facing -Y (forward)
        if yaw_deg < 0:
            facing_deg = 360 + yaw_deg
        else:
            facing_deg = yaw_deg
        
        # Calculate off-balance (combined pitch and roll deviation from vertical)
        roll_deg = abs(euler_angles[0])  # Roll around X-axis
        pitch_deg = abs(euler_angles[1])  # Pitch around Y-axis
        
        # Combined off-balance is the sum of roll and pitch deviations from vertical
        # Perfectly upright = 0°, maximum tilt = 180° (though robot would fall before then)
        off_balance = roll_deg + pitch_deg
        
        # Height off the ground (Z coordinate)
        height = center_pos[2]
        
        # Store current facing direction for reward function (even on first call)
        track_orientation.last_facing_deg = facing_deg
        
        # Calculate current directions relative to robot's current facing (WASD format)
        # These change as the robot rotates
        curr_w = facing_deg  # Current forward direction (W key)
        curr_s = (facing_deg + 180) % 360  # Opposite of forward (S key)
        curr_a = (facing_deg + 90) % 360  # 90° left of forward (A key)
        curr_d = (facing_deg + 270) % 360  # 90° right of forward (D key)
        
        # Print position, facing direction, balance, height, and current WASD directions
        # print(f"center: x {center_pos[0]:.3f}, y {center_pos[1]:.3f}, z {center_pos[2]:.3f} facing(deg): {facing_deg:.0f} off_balance(deg): {off_balance:.1f} height(m): {height:.3f} curr_w(deg): {curr_w:.0f} curr_s(deg): {curr_s:.0f} curr_a(deg): {curr_a:.0f} curr_d(deg): {curr_d:.0f}")
        
        # Check if position is changing (robot is actually moving)
        if not hasattr(track_orientation, 'last_position'):
            track_orientation.last_position = center_pos
            track_orientation.static_count = 0
        else:
            # Calculate horizontal distance moved (ignore Z-axis bouncing)
            dx = center_pos[0] - track_orientation.last_position[0]  # X movement
            dy = center_pos[1] - track_orientation.last_position[1]  # Y movement
            
            horizontal_distance = np.sqrt(dx**2 + dy**2)
            
            if horizontal_distance < 0.001:  # Less than 1mm horizontal movement
                track_orientation.static_count += 1
                if track_orientation.static_count > 10:
                    print(f"   ⚠️  Robot hasn't moved horizontally in {track_orientation.static_count} steps!")
                
                # Store current facing direction even when not moving (for reward function)
                track_orientation.last_facing_deg = facing_deg
            else:
                track_orientation.static_count = 0
                
                # Calculate directional movement components relative to robot's current facing (WASD format)
                facing_rad = np.radians(facing_deg)
                
                # Project movement onto robot's current WASD directions
                # W = forward, S = backward, A = left, D = right
                w_movement = -dy * np.cos(facing_rad) - dx * np.sin(facing_rad)  # W direction (forward)
                s_movement = -w_movement  # S direction (backward) - opposite of W
                a_movement = -dx * np.cos(facing_rad) + dy * np.sin(facing_rad)  # A direction (left)
                d_movement = -a_movement  # D direction (right) - opposite of A
                
                # Determine movement direction string (WASD format)
                movement_direction = ""
                if abs(w_movement) > 0.001:
                    if w_movement > 0:
                        movement_direction += "w"
                    else:
                        movement_direction += "s"
                
                if abs(a_movement) > 0.001:
                    if a_movement > 0:
                        movement_direction += "a"
                    else:
                        movement_direction += "d"
                
                if not movement_direction:
                    movement_direction = "n"  # No significant movement
                
                # Calculate rotation (change in facing direction)
                if not hasattr(track_orientation, 'last_facing'):
                    track_orientation.last_facing = facing_deg
                    rotation_deg = 0.0
                else:
                    # Calculate rotation as change in facing direction
                    rotation_change = facing_deg - track_orientation.last_facing
                    
                    # Handle wrapping around 360° boundary
                    if rotation_change > 180:
                        rotation_change -= 360
                    elif rotation_change < -180:
                        rotation_change += 360
                    
                    rotation_deg = rotation_change
                    track_orientation.last_facing = facing_deg
                
                # print(f"   moved(m): {horizontal_distance:.3f} ({movement_direction}) w: {w_movement:.3f} s: {s_movement:.3f} a: {a_movement:.3f} d: {d_movement:.3f}")
                # print(f"   rotated(deg): {rotation_deg:.1f}")
                
                # Store movement data for reward function to access
                track_orientation.last_movement_data = {
                    'w': w_movement,
                    's': s_movement,
                    'a': a_movement,
                    'd': d_movement,
                    'movement_direction': movement_direction,
                    'horizontal_distance': horizontal_distance
                }
                track_orientation.last_rotation = rotation_deg
                track_orientation.last_off_balance = off_balance
                track_orientation.last_facing_deg = facing_deg  # Store current facing direction for strict movement detection
                
            track_orientation.last_position = center_pos
        
        return center_pos, facing_deg

    except Exception as e:
        print(f"❌ Failed to track orientation: {e}")
        return None, None


########## EPISODE MANAGEMENT ##########

##### start episode #####

def start_episode():
    """Start a new training episode"""
    global episode_counter, episode_step, episode_reward, last_state, last_action

    episode_counter += 1
    episode_step = 0
    episode_reward = 0.0
    last_state = None
    last_action = None

    # COMPLETELY GUTTED - No more movement tracking
    # You now control movement rewards through your reward function

    print(f"🚀 Starting episode {episode_counter}")
    print(f"   🎯 Episode started - implement your own movement tracking if desired")
    # Show initial orientation at episode start
    track_orientation()


##### end episode #####

def end_episode():
    """End current episode and save progress"""
    global episode_counter, episode_reward, episode_scores, average_score

    print(f"🎯 Episode {episode_counter} ended:")
    print(f"   📊 Steps: {episode_step}")
    print(f"   📊 Final Reward: {episode_reward:.3f}")

    # Track episode scores for average calculation
    episode_scores.append(episode_reward)
    if len(episode_scores) > 100:  # Keep only last 100 episodes for recent average
        episode_scores.pop(0)
    
    # Calculate running average
    average_score = sum(episode_scores) / len(episode_scores)
    print(f"   📊 Average Score (last {len(episode_scores)} episodes): {average_score:.3f}")

    # Save model periodically based on total steps (only if TD3 policy is initialized)
    if total_steps % TRAINING_CONFIG['save_frequency'] == 0 and td3_policy is not None:
        save_model(
            f"/home/matthewthomasbeck/Projects/Robot_Dog/model/td3_steps_{total_steps}_episode_{episode_counter}_avg_{average_score:.2f}.pth")
        print(f"💾 Model saved: steps_{total_steps}, episode_{episode_counter}, avg_score_{average_score:.2f}")
    elif td3_policy is None:
        print(f"⚠️  Warning: TD3 policy not initialized yet, skipping model save for episode {episode_counter}")
    else:
        print(f"📝 Episode {episode_counter} completed but not at save frequency ({TRAINING_CONFIG['save_frequency']} steps)")


##### check/reset episode #####

def check_and_reset_episode_if_needed():  # TODO compare with reset_episode()
    """
    Check if episode should be reset and trigger reset if needed.
    This function should be called from the main thread to integrate episode management.

    Returns:
        bool: True if episode was reset, False otherwise
    """
    global episode_step, episode_counter, episode_reward

    import utilities.config as config

    # COMPLETELY GUTTED - No more automatic episode termination due to falling
    # You now control when episodes end through your reward system
    # The robot can fall and recover without ending the episode

    # COMPLETELY GUTTED - You now have complete control over warning systems
    # Add your custom warning logic here if desired

    # Check if episode has reached max steps
    if episode_step >= TRAINING_CONFIG['max_steps_per_episode']:
        print(f"🎯 Episode {episode_counter} completed after {episode_step} steps!")
        # Save model before resetting
        end_episode()
        
        # Monitor learning progress before resetting
        monitor_learning_progress()
        
        reset_episode()
        return True

    return False


##### reset episode ##### TODO compare with check_and_reset_episode_if_needed()

def reset_episode():
    """
    Reset the current episode by resetting Isaac Sim world and moving robot to neutral position.
    This is the critical function that was working in working_robot_reset.py
    """
    global episode_step, episode_reward, last_state, last_action, episode_counter, total_steps

    try:
        logging.info(
            f"(training.py): Episode {episode_counter} ending - Episode complete, resetting Isaac Sim world.\n")

        # CRITICAL: Save the model before resetting (if episode had any progress)
        if episode_step > 0:
            end_episode()

        # CRITICAL: Small delay to ensure all experiences are processed
        # This gives the agent time to learn from the falling experience
        import time
        time.sleep(0.2)  # 200ms delay for experience processing
        
        # CRITICAL: Reset Isaac Sim world (position, velocity, physics state)
        if config.USE_SIMULATION and config.USE_ISAAC_SIM:
            # Reset the world - this resets robot position, velocities, and physics state
            config.ISAAC_WORLD.reset()

            # Give Isaac Sim a few steps to stabilize after world reset
            for _ in range(5):
                config.ISAAC_WORLD.step(render=True)

        # CRITICAL: Move robot to neutral position (joint angles)
        neutral_position(10)  # High intensity for quick reset

        # Give Isaac Sim more steps to stabilize after neutral position
        if config.USE_SIMULATION and config.USE_ISAAC_SIM:
            for _ in range(5):
                config.ISAAC_WORLD.step(render=True)

        # Reset Python tracking variables
        episode_step = 0
        episode_reward = 0.0
        last_state = None
        last_action = None

        # COMPLETELY GUTTED - No more movement tracking reset
        # You now control movement tracking through your reward function

        # Increment episode counter
        episode_counter += 1

        logging.info(f"(training.py): Episode {episode_counter} reset complete - World and robot state reset.\n")

        # Log learning progress every 10 episodes
        if episode_counter % 10 == 0:
            print(f"🎯 Training Progress: {episode_counter} episodes completed")
            if replay_buffer is not None:
                print(f"   📊 Replay buffer size: {len(replay_buffer)}")
            if td3_policy is not None:
                print(f"   🧠 TD3 policy ready for training")

    except Exception as e:
        logging.error(f"(training.py): Failed to reset episode: {e}\n")
        # Don't crash - try to continue with next episode
        episode_counter += 1
        episode_step = 0
        episode_reward = 0.0


########## MODEL FUNCTIONS ##########

##### save trained model #####

def save_model(filepath):
    """Save the current TD3 model"""
    if td3_policy:

        print(f"💾 Saving TD3 model to: {filepath}")
        print(f"   📊 Current episode: {episode_counter}")
        print(f"   📊 Current step: {episode_step}")
        print(f"   📊 Total steps: {total_steps}")
        print(f"   📊 Episode reward: {episode_reward:.4f}")

        # Create the checkpoint data
        checkpoint = {
            'actor_state_dict': td3_policy.actor.state_dict(),
            'critic_1_state_dict': td3_policy.critic_1.state_dict(),
            'critic_2_state_dict': td3_policy.critic_2.state_dict(),
            'actor_target_state_dict': td3_policy.actor_target.state_dict(),
            'critic_1_target_state_dict': td3_policy.critic_1_target.state_dict(),
            'critic_2_target_state_dict': td3_policy.critic_2_target.state_dict(),
            'actor_optimizer_state_dict': td3_policy.actor_optimizer.state_dict(),
            'critic_1_optimizer_state_dict': td3_policy.critic_1_optimizer.state_dict(),
            'critic_2_optimizer_state_dict': td3_policy.critic_2_optimizer.state_dict(),
            'episode_counter': episode_counter,
            'total_steps': total_steps,
            'episode_reward': episode_reward,
        }

        # Verify checkpoint data before saving
        print(f"   🔍 Checkpoint contains {len(checkpoint)} keys:")
        for key, value in checkpoint.items():
            if 'state_dict' in key:
                if isinstance(value, dict):
                    print(f"      ✅ {key}: {len(value)} layers")
                else:
                    print(f"      ❌ {key}: Invalid type {type(value)}")
            else:
                print(f"      📊 {key}: {value}")

        # Save the model
        torch.save(checkpoint, filepath)

        # Verify the file was created and has content
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   ✅ Model saved successfully! File size: {file_size:.1f} MB")
        else:
            print(f"   ❌ Failed to save model - file not created")

        print(f"Model saved to: {filepath}")
        print(f"   📊 Current average score: {average_score:.3f}")
    else:
        print(f"❌ Cannot save model - TD3 policy not initialized")


##### load the model #####

def load_model(filepath):  # TODO find out how this can be used???
    """Load a TD3 model from file"""
    global td3_policy, episode_counter, total_steps, episode_reward

    if td3_policy and os.path.exists(filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        td3_policy.actor.load_state_dict(checkpoint['actor_state_dict'])
        td3_policy.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        td3_policy.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        td3_policy.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        td3_policy.critic_1_target.load_state_dict(checkpoint['critic_1_target_state_dict'])
        td3_policy.critic_2_target.load_state_dict(checkpoint['critic_2_target_state_dict'])
        td3_policy.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        td3_policy.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
        td3_policy.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])

        # Restore training state
        episode_counter = checkpoint.get('episode_counter', 0)
        total_steps = checkpoint.get('total_steps', 0)
        episode_reward = checkpoint.get('episode_reward', 0.0)

        print(f"Model loaded from: {filepath}")
        print(f"  - Episode: {episode_counter}")
        print(f"  - Total steps: {total_steps}")
        print(f"  - Episode reward: {episode_reward:.4f}")
        return True
    return False


def monitor_learning_progress():
    """
    Monitor the agent's learning progress and detect potential issues.
    This helps identify if the agent is stuck in a loop of falling behaviors.
    """
    global episode_counter, episode_reward, replay_buffer
    
    if replay_buffer is None or len(replay_buffer) == 0:
        return
    
    # Calculate average reward over last few episodes
    recent_rewards = []
    if hasattr(monitor_learning_progress, 'episode_rewards'):
        recent_rewards = monitor_learning_progress.episode_rewards[-5:]  # Last 5 episodes
    
    # Store current episode reward
    if not hasattr(monitor_learning_progress, 'episode_rewards'):
        monitor_learning_progress.episode_rewards = []
    
    monitor_learning_progress.episode_rewards.append(episode_reward)
    
    # Keep only last 20 episodes
    if len(monitor_learning_progress.episode_rewards) > 20:
        monitor_learning_progress.episode_rewards.pop(0)
    
    # Analyze learning progress
    if len(recent_rewards) >= 3:
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        min_reward = min(recent_rewards)
        
        print(f"📊 Learning Progress Analysis:")
        print(f"   🎯 Recent episodes: {len(recent_rewards)}")
        print(f"   📈 Average reward: {avg_reward:.3f}")
        print(f"   📉 Worst reward: {min_reward:.3f}")
        print(f"   🧠 Replay buffer: {len(replay_buffer)} experiences")
        
        # COMPLETELY GUTTED - No more automatic fall loop detection
        # You now control what constitutes problematic behavior through your reward system
        
        # Detect if agent is improving
        if len(monitor_learning_progress.episode_rewards) >= 10:
            first_half = monitor_learning_progress.episode_rewards[:10]
            second_half = monitor_learning_progress.episode_rewards[-10:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg > first_avg:
                print(f"✅ Agent is improving! First 10: {first_avg:.3f}, Last 10: {second_avg:.3f}")
            else:
                print(f"❌ Agent not improving. First 10: {first_avg:.3f}, Last 10: {second_avg:.3f}")
