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

from movement.fundamental_movement import neutral_position

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

# COMPLETELY GUTTED - No more forward movement tracking
# You now have complete control over movement detection and rewards

##### training config ##### TODO move me to config.py when ready

TRAINING_CONFIG = {
    'max_episodes': 1000000,
    'max_steps_per_episode': 1000,  # GPT-5 recommendation: 600-1200 steps (~10-20 seconds)
    'save_frequency': 10000,  # Save model every 10,000 steps (instead of every 10 episodes)
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

# COMPLETELY GUTTED - No more desired pitch tracking
# You now control pitch behavior through your reward function





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
    global td3_policy, replay_buffer

    print("Initializing training system...")
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
    COMPLETELY GUTTED - You now have complete control to implement your own reward system from scratch.
    
    This function is called every step to calculate the reward for the current state.
    
    Args:
        current_angles: Current joint angles for all legs
        commands: Movement commands (e.g., 'w', 's', 'a', 'd')
        intensity: Movement intensity (1-10)

    Returns:
        float: Your custom reward value for this step
    """
    # TODO: Implement your custom reward system here
    # You have complete control over what gets rewarded and punished
    
    # Example placeholder - replace with your own logic:
    reward = 0.0
    
    # Add your reward logic here:
    # - What behaviors do you want to encourage?
    # - What behaviors do you want to discourage?
    # - How do you want to measure success?
    # - What are your training objectives?
    
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
        
        # Calculate current directions relative to robot's current facing (WASD format)
        # These change as the robot rotates
        curr_w = facing_deg  # Current forward direction (W key)
        curr_s = (facing_deg + 180) % 360  # Opposite of forward (S key)
        curr_a = (facing_deg + 90) % 360  # 90° left of forward (A key)
        curr_d = (facing_deg + 270) % 360  # 90° right of forward (D key)
        
        # Print position, facing direction, balance, height, and current WASD directions
        print(f"center: x {center_pos[0]:.3f}, y {center_pos[1]:.3f}, z {center_pos[2]:.3f} facing(deg): {facing_deg:.0f} off_balance(deg): {off_balance:.1f} height(m): {height:.3f} curr_w(deg): {curr_w:.0f} curr_s(deg): {curr_s:.0f} curr_a(deg): {curr_a:.0f} curr_d(deg): {curr_d:.0f}")
        
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
                
                print(f"   moved(m): {horizontal_distance:.3f} ({movement_direction}) w: {w_movement:.3f} s: {s_movement:.3f} a: {a_movement:.3f} d: {d_movement:.3f}")
                print(f"   rotated(deg): {rotation_deg:.1f}")
            
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
    global episode_counter, episode_reward

    print(f"🎯 Episode {episode_counter} ended:")
    print(f"   📊 Steps: {episode_step}")
    print(f"   📊 Final Reward: {episode_reward:.3f}")

    # Save model periodically based on total steps (only if TD3 policy is initialized)
    if total_steps % TRAINING_CONFIG['save_frequency'] == 0 and td3_policy is not None:
        save_model(
            f"/home/matthewthomasbeck/Projects/Robot_Dog/model/td3_steps_{total_steps}_episode_{episode_counter}_reward_{episode_reward:.2f}.pth")
        print(f"💾 Model saved: steps_{total_steps}, episode_{episode_counter}")
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
    else:
        print(f"❌ Cannot save model - TD3 policy not initialized")


##### load the model #####

def load_model(filepath):  # TODO find out how this can be used???
    """Load a TD3 model from file"""
    global td3_policy, episode_counter, total_steps, episode_reward

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
