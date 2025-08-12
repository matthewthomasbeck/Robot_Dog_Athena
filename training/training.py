##################################################################################
# Copyright (c) 2024 Matthew Thomas Beck                                         #
#                                                                                #
# All rights reserved. This code and its associated files may not be reproduced, #
# modified, distributed, or otherwise used, in part or in whole, by any person   #
# or entity without the express written permission of the copyright holder,      #
# Matthew Thomas Beck.                                                           #
##################################################################################

"""
INTEGRATION WITH MAIN LOOP:

To use this training system, add this to your main loop after each simulation step:

if config.USE_SIMULATION and config.USE_ISAAC_SIM:
    from training.training import integrate_with_main_loop
    integrate_with_main_loop()

This will automatically:
1. Check if robot has fallen
2. Reset Isaac Sim world when episodes end
3. Move robot back to neutral position
4. Reset episode tracking variables
5. Continue training seamlessly

The key difference from your old system is that this properly resets the Isaac Sim world
and robot position, not just Python variables.
"""


############################################################
############### IMPORT / CREATE DEPENDENCIES ###############
############################################################


import utilities.config as config





##################################################
############### TRAINING FUNCTIONS ###############
##################################################


########## SIMULATION VARIABLES ##########

def set_simulation_variables(robot_id, joint_map): # for pybullet, ignore me
    """
    Set global simulation variables for PyBullet.
    Args:
        robot_id: PyBullet robot body ID
        joint_map: Dictionary mapping joint names to indices
    """
    global ROBOT_ID, JOINT_MAP
    ROBOT_ID = robot_id
    JOINT_MAP = joint_map


########## FALLEN ROBOT ##########

def is_robot_fallen():  # TODO this function does a good job at telling when the robot has fallen, DO NOT TOUCH
    """
    Check if robot has tilted more than 45 degrees from upright.
    Returns:
        bool: True if robot has fallen over
    """
    import numpy as np
    import utilities.config as config
    from scipy.spatial.transform import Rotation

    try:
        positions, rotations = config.ISAAC_ROBOT.get_world_poses()
        # get_world_poses() returns arrays - get first element (index 0)
        rotation = rotations[0]  # First robot rotation

        # Convert quaternion to rotation object
        quat_wxyz = [rotation[0], rotation[1], rotation[2], rotation[3]]  # (w, x, y, z)
        r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # scipy wants (x, y, z, w)

        # Robot's up direction is local +Z
        local_up = np.array([0.0, 0.0, 1.0])  # Local +Z
        robot_up = r.apply(local_up)
        world_up = np.array([0.0, 0.0, 1.0])  # World +Z

        # Calculate angle between robot up and world up
        dot_product = np.dot(robot_up, world_up)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure valid range
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        return angle_deg > 45.0  # Fallen if tilted more than 45 degrees

    except Exception as e:
        import logging
        logging.error(f"(training.py): Failed to check if robot fallen: {e}\n")
        return False  # Assume upright if error


########## STANDARD RL AGENT INTERFACE ##########

def get_rl_action_standard(state, commands, intensity, frame): # will eventually take camera frame as input
    pass


########## BLIND RL AGENT INTERFACE ##########

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# TD3 Networks
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

# Global TD3 instance and replay buffer
td3_policy = None
replay_buffer = None
episode_step = 0
episode_reward = 0.0
episode_counter = 0
last_state = None
last_action = None
total_steps = 0

# Training configuration
TRAINING_CONFIG = {
    'max_episodes': 1000000,
    'max_steps_per_episode': 500,  # ~8.3 seconds at 60 FPS
    'save_frequency': 1,  # Save model every episode (for testing)
    'training_frequency': 10,  # Train every 10 steps
    'batch_size': 64,
    'learning_rate': 3e-4,
    'gamma': 0.99,  # Discount factor
    'tau': 0.005,   # Target network update rate
    'exploration_noise': 0.1,
    'max_action': 1.0
}

def initialize_training():
    """Initialize the complete training system"""
    global td3_policy, replay_buffer
    
    print("Initializing training system...")
    
    # Create models directory
    import os
    models_dir = "/home/matthewthomasbeck/Projects/Robot_Dog/model"
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize TD3
    state_dim = 21  # 12 joints + 8 commands + 1 intensity
    action_dim = 24  # 12 mid + 12 target angles
    max_action = TRAINING_CONFIG['max_action']
    
    td3_policy = TD3(state_dim, action_dim, max_action)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=100000)
    
    print(f"Training system initialized:")
    print(f"  - State dimension: {state_dim}")
    print(f"  - Action dimension: {action_dim}")
    print(f"  - Models directory: {models_dir}")
    print(f"  - TD3 policy created: {td3_policy is not None}")
    print(f"  - Replay buffer created: {replay_buffer is not None}")

def start_episode():
    """Start a new training episode"""
    global episode_counter, episode_step, episode_reward, last_state, last_action
    
    episode_counter += 1
    episode_step = 0
    episode_reward = 0.0
    last_state = None
    last_action = None
    
    print(f"Starting episode {episode_counter}")

def end_episode():
    """End current episode and save progress"""
    global episode_counter, episode_reward
    
    print(f"Episode {episode_counter} ended:")
    print(f"  - Steps: {episode_step}")
    print(f"  - Reward: {episode_reward:.4f}")
    
    # Save model periodically (only if TD3 policy is initialized)
    if episode_counter % TRAINING_CONFIG['save_frequency'] == 0 and td3_policy is not None:
        save_model(f"/home/matthewthomasbeck/Projects/Robot_Dog/model/td3_episode_{episode_counter}_reward_{episode_reward:.2f}.pth")
        print(f"Model saved: episode_{episode_counter}")
    elif td3_policy is None:
        print(f"Warning: TD3 policy not initialized yet, skipping model save for episode {episode_counter}")
    else:
        print(f"Episode {episode_counter} completed but not at save frequency ({TRAINING_CONFIG['save_frequency']})")

def save_model(filepath):
    """Save the current TD3 model"""
    if td3_policy:
        import torch
        torch.save({
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
        }, filepath)
        print(f"Model saved to: {filepath}")

def load_model(filepath):
    """Load a TD3 model from file"""
    global td3_policy, episode_counter, total_steps, episode_reward
    import os
    import torch
    
    if td3_policy and os.path.exists(filepath):
        checkpoint = torch.load(filepath)
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


def get_neutral_action():
    """
    Return neutral joint angles when episodes are reset.
    This gives the robot time to stabilize after reset.
    """
    import utilities.config as config
    
    target_angles = {}
    mid_angles = {}
    movement_rates = {}
    
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        target_angles[leg_id] = {}
        mid_angles[leg_id] = {}
        movement_rates[leg_id] = {'speed': 0.1, 'acceleration': 0.5}  # Very slow for stability
        
        for joint_name in ['hip', 'upper', 'lower']:
            # Get neutral angles from config
            servo_data = config.SERVO_CONFIG[leg_id][joint_name]
            neutral_angle = servo_data.get('NEUTRAL_ANGLE', 0.0)  # Default to 0 if not defined
            
            target_angles[leg_id][joint_name] = neutral_angle
            mid_angles[leg_id][joint_name] = neutral_angle
    
    return target_angles, mid_angles, movement_rates


def check_and_reset_episode_if_needed():
    """
    Check if episode should be reset and trigger reset if needed.
    This function should be called from the main thread to integrate episode management.
    
    Returns:
        bool: True if episode was reset, False otherwise
    """
    global episode_step, episode_counter
    
    # Check if robot has fallen (episode should end)
    if is_robot_fallen():
        print(f"Robot fallen! Ending episode {episode_counter} at step {episode_step}")
        # Save model before resetting
        if episode_step > 0:
            end_episode()
        reset_episode()
        return True
    
    # Check if episode has reached max steps
    if episode_step >= TRAINING_CONFIG['max_steps_per_episode']:
        print(f"Episode {episode_counter} completed after {episode_step} steps!")
        # Save model before resetting
        end_episode()
        reset_episode()
        return True
    
    return False


def get_rl_action_blind(current_angles, commands, intensity):
    """
    TD3 RL agent that takes current joint angles, commands, and intensity as state
    and outputs 24 values (12 mid angles + 12 target angles)
    
    NOTE: This function runs in worker threads, so it CANNOT call Isaac Sim reset functions.
    It only signals that a reset is needed - the main thread handles actual resets.
    """
    global td3_policy, replay_buffer, episode_step, episode_reward, episode_counter
    global last_state, last_action, total_steps
    
    # Import config at the top to avoid importing in loops
    import utilities.config as config

    # Initialize training system if not done yet
    if td3_policy is None:
        initialize_training()
        start_episode()
    
    # CRITICAL: Check episode completion but DON'T reset from worker thread
    # Just signal that a reset is needed - main thread will handle it
    episode_needs_reset = False
    
    if is_robot_fallen():
        episode_needs_reset = True
        print(f"Robot fallen! Episode {episode_counter} needs reset (signaling main thread)")
    
    if episode_step >= TRAINING_CONFIG['max_steps_per_episode']:
        episode_needs_reset = True
        print(f"Episode {episode_counter} completed after {episode_step} steps! (signaling main thread)")
    
    # Store reset signal for main thread to check
    config.EPISODE_NEEDS_RESET = episode_needs_reset
    
    # Log episode progress every 100 steps
    if episode_step % 100 == 0:
        print(f"Episode {episode_counter}, Step {episode_step}, Total Steps: {total_steps}")
    
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
    if add_noise:
        action = td3_policy.select_action(state)
        noise = np.random.normal(0, TRAINING_CONFIG['exploration_noise'], size=action.shape)
        action = np.clip(action + noise, -TRAINING_CONFIG['max_action'], TRAINING_CONFIG['max_action'])
    else:
        action = td3_policy.select_action(state)
    
    # Store experience for training
    if last_state is not None and last_action is not None:
        # Calculate reward based on robot state
        if is_robot_fallen():
            reward = -10.0  # Heavy penalty for falling
            print(f"Robot fell! Applying penalty reward: {reward}")
        else:
            reward = 0.1  # Small positive reward for staying upright
        
        # Check if episode ended due to fall or completion
        done = is_robot_fallen() or episode_step >= TRAINING_CONFIG['max_steps_per_episode']
        
        # Add to replay buffer
        replay_buffer.add(last_state, last_action, reward, state, done)
        
        # Train TD3 periodically
        if total_steps % TRAINING_CONFIG['training_frequency'] == 0 and len(replay_buffer) >= TRAINING_CONFIG['batch_size']:
            print(f"Training TD3 at step {total_steps}, buffer size: {len(replay_buffer)}")
            td3_policy.train(replay_buffer, TRAINING_CONFIG['batch_size'])
    
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


def update_reward(reward):
    """Update the current episode reward (called from external reward calculation)"""
    global episode_reward
    episode_reward += reward

def reset_episode():
    """
    Reset the current episode by resetting Isaac Sim world and moving robot to neutral position.
    This is the critical function that was working in working_robot_reset.py
    """
    global episode_step, episode_reward, last_state, last_action, episode_counter, total_steps
    
    import logging
    from movement.fundamental_movement import neutral_position
    import utilities.config as config
    
    try:
        logging.info(f"(training.py): Episode {episode_counter} ending - Robot fallen or episode complete, resetting Isaac Sim world.\n")
        
        # CRITICAL: Save the model before resetting (if episode had any progress)
        if episode_step > 0:
            end_episode()
        
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
        
        # Increment episode counter
        episode_counter += 1
        
        logging.info(f"(training.py): Episode {episode_counter} reset complete - World and robot state reset.\n")
        
        # Log learning progress every 10 episodes
        if episode_counter % 10 == 0:
            logging.info(f"(training.py): LEARNING PROGRESS - Episodes: {episode_counter}, Total Steps: {total_steps}")
        
    except Exception as e:
        logging.error(f"(training.py): Failed to reset episode: {e}\n")
        # Fallback: just reset Python variables if Isaac Sim reset fails
        episode_step = 0
        episode_reward = 0.0
        last_state = None
        last_action = None
        episode_counter += 1

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


def get_training_status():
    """Get current training status for monitoring"""
    return {
        'episode': episode_counter,
        'episode_step': episode_step,
        'episode_reward': episode_reward,
        'total_steps': total_steps,
        'replay_buffer_size': len(replay_buffer) if replay_buffer else 0,
        'model_initialized': td3_policy is not None
    }