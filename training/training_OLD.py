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
import os
import logging
import random

##### import necessary functions #####

from movement.isaac_joints import neutral_position_isaac
from training.agents import *
import training.rewards as rewards
from training.orientation import track_orientation


########## CREATE DEPENDENCIES ##########

##### global PPO instance and episode data #####

ppo_policy = None
episode_reward = 0.0
episode_counter = 0
episode_states = []
episode_actions = []
episode_rewards = []
episode_values = []
episode_log_probs = []
episode_dones = []
total_steps = 0
episode_scores = []
average_score = 0.0

##### random command and intensity #####

previous_command = None
previous_intensity = None





##################################################
############### ISAAC SIM TRAINING ###############
##################################################


########## MAIN LOOP ##########

##### initialize training #####

def initialize_training():
    """Initialize the complete training system"""
    global ppo_policy, episode_counter, total_steps

    logging.debug("(training.py): Initializing training system...\n")
    os.makedirs(config.MODELS_DIRECTORY, exist_ok=True)

    # Initialize PPO
    state_dim = 19  # 12 joints + 6 commands + 1 intensity
    action_dim = 36  # 12 mid + 12 target angles + 12 velocity values
    max_action = config.TRAINING_CONFIG['max_action']

    ppo_policy = PPO(state_dim, action_dim, max_action)

    # Try to load the latest saved model to continue training
    latest_model = find_latest_model()
    if latest_model:
        logging.debug(f"🔄 Loading latest model: {latest_model}...\n")
        if load_model(latest_model):
            logging.info(f"✅ Successfully loaded model from step {total_steps}, episode {episode_counter}\n")
        else:
            logging.warning(f"❌ Failed to load model, starting fresh.\n")
            episode_counter = 0
            total_steps = 0
    else:
        logging.warning(f"🆕 No saved models found, starting fresh training.\n")
        episode_counter = 0
        total_steps = 0

    logging.info(f"Training system initialized:")
    logging.info(f"  - State dimension: {state_dim}")
    logging.info(f"  - Action dimension: {action_dim}")
    logging.info(f"  - Models directory: {config.MODELS_DIRECTORY}")
    logging.info(f"  - PPO policy created: {ppo_policy is not None}")
    logging.info(f"  - Starting from episode: {episode_counter}")
    logging.info(f"  - Starting from step: {total_steps}\n")


def find_latest_model():
    """Find the latest saved model file"""
    import glob
    import re
    
    # Look for model files in the models directory
    model_pattern = os.path.join(config.MODELS_DIRECTORY, "ppo_steps_*_episode_*_reward_*.pth")
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        return None
    
    # Extract step numbers and find the latest
    latest_model = None
    latest_steps = 0
    
    for model_file in model_files:
        # Extract step number from filename like "ppo_steps_700000_episode_47474_avg_-109.76.pth"
        match = re.search(r'ppo_steps_(\d+)_episode_', os.path.basename(model_file))
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
    global episode_counter, total_steps

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
    PPO RL agent that takes current joint angles, commands, and intensity as state
    and outputs 24 values (12 mid angles + 12 target angles)

    NOTE: This function runs in worker threads, so it CANNOT call Isaac Sim reset functions.
    It only signals that a reset is needed - the main thread handles actual resets.
    """
    global ppo_policy, episode_reward, episode_counter
    global episode_states, episode_actions, episode_rewards, episode_values, episode_log_probs, episode_dones
    global total_steps

    # Initialize training system if not done yet
    if ppo_policy is None:
        initialize_training()
        start_episode()

    # CRITICAL: Check episode completion but DON'T reset from worker thread
    # Just signal that a reset is needed - main thread will handle it
    episode_needs_reset = False

    # COMPLETELY GUTTED - No more automatic episode termination due to falling
    # You now control when episodes end through your reward system

    if rewards.EPISODE_STEP >= config.TRAINING_CONFIG['max_steps_per_episode']:
        episode_needs_reset = True
        print(f"Episode {episode_counter} completed after {rewards.EPISODE_STEP} steps! (signaling main thread)")

    # Store reset signal for main thread to check
    config.EPISODE_NEEDS_RESET = episode_needs_reset

    # Log episode progress every 100 steps
    if rewards.EPISODE_STEP % 100 == 0:
        print(f"Episode {episode_counter}, Step {rewards.EPISODE_STEP}, Total Steps: {total_steps}")
    
    # Track orientation every 50 steps to understand robot facing direction
    if rewards.EPISODE_STEP % 50 == 0:
        track_orientation()

    # Build state vector
    state = []

    # 1. Extract and normalize joint angles (12D)
    # Normalize angles to [-1, 1] range for better training stability with Adam optimizer
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        for joint_name in ['hip', 'upper', 'lower']:
            angle = current_angles[leg_id][joint_name]
            
            # Get joint limits for normalization
            servo_data = config.SERVO_CONFIG[leg_id][joint_name]
            min_angle = servo_data['FULL_BACK_ANGLE']
            max_angle = servo_data['FULL_FRONT_ANGLE']
            
            # Ensure correct order
            if min_angle > max_angle:
                min_angle, max_angle = max_angle, min_angle
            
            # Normalize to [-1, 1] range
            angle_range = max_angle - min_angle
            if angle_range > 0:
                normalized_angle = 2.0 * (float(angle) - min_angle) / angle_range - 1.0
                normalized_angle = np.clip(normalized_angle, -1.0, 1.0)
            else:
                normalized_angle = 0.0  # Fallback if range is zero
                
            state.append(normalized_angle)

    # 2. Encode commands (6D one-hot for movement commands only)
    # Note: arrowup/arrowdown are filtered out by move_direction, so we only need 6 dimensions
    if isinstance(commands, list):
        command_list = commands
    elif isinstance(commands, str):
        command_list = commands.split('+') if commands else []
    else:
        command_list = []

    # CRITICAL: Always create exactly 6D command encoding regardless of input list length
    # This ensures consistent state size and prevents dynamic shapes
    command_encoding = [
        1.0 if 'w' in command_list else 0.0,
        1.0 if 's' in command_list else 0.0,
        1.0 if 'a' in command_list else 0.0,
        1.0 if 'd' in command_list else 0.0,
        1.0 if 'arrowleft' in command_list else 0.0,
        1.0 if 'arrowright' in command_list else 0.0
    ]
    
    # Validate command encoding size
    if len(command_encoding) != 6:
        raise ValueError(f"Command encoding must be exactly 6D, got {len(command_encoding)}D")
    
    state.extend(command_encoding)

    # 3. Normalize intensity (1D) - Adam optimizer prefers values roughly in [-1, 1] range
    # Map intensity 1-10 to range [-1.0, 1.0] preserving all 10 distinct levels
    intensity_normalized = (float(intensity) - 5.5) / 4.5  # Maps 1->-1.0, 10->1.0
    # No clipping needed - this preserves all 10 distinct intensity levels
    state.append(intensity_normalized)
    
    # Convert to numpy array and validate state size
    state = np.array(state, dtype=np.float32)
    
    # Validate state size: 12 (joints) + 6 (commands) + 1 (intensity) = 19
    expected_state_size = 19
    if len(state) != expected_state_size:
        raise ValueError(f"State size mismatch: expected {expected_state_size}, got {len(state)}")
    
    # Log state composition for debugging
    if rewards.EPISODE_STEP % 100 == 0:  # Log every 100 steps to avoid spam
        print(f"State composition: {len(state)}D total")
        print(f"  - Joint angles: 12D (normalized to [-1, 1])")
        print(f"  - Commands: 6D (one-hot for w,s,a,d,arrowleft,arrowright)")
        print(f"  - Intensity: 1D (normalized to [-1.0, 1.0])")

    # Get action from PPO (stochastic during training, deterministic for deployment)
    # PPO handles exploration through action distributions, no manual noise needed
    action = ppo_policy.select_action(state, deterministic=False)  # Stochastic for training
    
    # COMPLETELY GUTTED - No more automatic fall tracking
    # You now control fall recovery through your reward system

    # Store experience for training
    if episode_states and episode_actions: # Only add if previous step had data
        # Calculate reward using the dedicated reward function
        reward = rewards.calculate_step_reward(current_angles, commands, intensity)

        # Update episode reward for tracking
        episode_reward += reward

        # COMPLETELY GUTTED - No more automatic episode termination due to falling
        # You now control when episodes end through your reward system
        done = rewards.EPISODE_STEP >= config.TRAINING_CONFIG['max_steps_per_episode']

        # Add to episode data lists
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_values.append(None) # Value is not available here, will be calculated later
        episode_log_probs.append(None) # Log prob is not available here, will be calculated later
        episode_dones.append(done)
        
        # Log experience collection for debugging
        if done:
            print(f"💾 Experience collected: State={len(state)}D, Action={len(action)}D, Reward={reward:.3f}, Done={done}")
            print(f"   📊 Episode data size: {len(episode_states)}")

        # Train PPO at episode end (PPO processes full episodes)
        if done and len(episode_states) >= config.TRAINING_CONFIG['batch_size']:
            print(f"🧠 Training PPO at episode end, episode data size: {len(episode_states)}")
            
            # Convert to tensors
            states = torch.FloatTensor(episode_states)
            actions = torch.FloatTensor(episode_actions)
            rewards_tensor = torch.FloatTensor(episode_rewards)
            dones = torch.FloatTensor(episode_dones)
            
            # Calculate log probabilities for all actions in the episode
            with torch.no_grad():
                _, log_probs, _, _ = ppo_policy.actor_critic.get_action_and_value(states, actions)
                log_probs = log_probs.squeeze(-1)  # Remove extra dimension
            
            # Get final value estimate for GAE calculation
            with torch.no_grad():
                final_value = ppo_policy.actor_critic.get_value(states[-1:]).item()
            
            # Train PPO on episode data
            ppo_policy.train(states, actions, log_probs, rewards_tensor, dones, final_value)
            
            # Reset episode data lists
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_values = []
            episode_log_probs = []
            episode_dones = []
            
            # COMPLETELY GUTTED - No more automatic fall-related logging
            # You now control what gets logged through your reward system

        # Save model periodically based on total steps
        if total_steps % config.TRAINING_CONFIG['save_frequency'] == 0 and ppo_policy is not None:
            save_model(
                f"/home/matthewthomasbeck/Projects/Robot_Dog/model/ppo_steps_{total_steps}_episode_{episode_counter}_reward_{episode_reward:.2f}.pth")
            print(f"💾 Model saved: steps_{total_steps}, episode_{episode_counter}")

    # Update tracking variables
    episode_states.append(state) # Append current state for the next step
    episode_actions.append(action) # Append current action for the next step
    
    # Get value and log_prob for PPO training (needed for next step)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        _, log_prob, _, value = ppo_policy.actor_critic.get_action_and_value(state_tensor, torch.FloatTensor(action).unsqueeze(0))
        episode_values.append(value.item())
        episode_log_probs.append(log_prob.item())
    
    rewards.EPISODE_STEP += 1
    total_steps += 1

    # Convert 36D action vector to joint angles and velocities:
    # action[0:11] = mid angles (12 joints)
    # action[12:23] = target angles (12 joints) 
    # action[24:35] = velocity values in radians/second (12 joints)
    target_angles = {}
    mid_angles = {}
    movement_rates = {}

    action_idx = 0

    for leg_id in ['FL', 'FR', 'BL', 'BR']:

        target_angles[leg_id] = {}
        mid_angles[leg_id] = {}
        movement_rates[leg_id] = {}

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

            # Convert velocity action (-1 to 1) to movement rate in radians/second
            # Velocity actions are the last 12 dimensions (indices 24-35)
            velocity_action = action[action_idx + 24]
            
            # Convert from [-1, 1] to [0, 9.5] radians/second for Isaac Sim
            # This maps the neural network output to Isaac Sim's expected velocity range
            joint_speed = (velocity_action + 1.0) * 4.75  # Maps [-1,1] to [0,9.5]
            joint_speed = np.clip(joint_speed, 0.0, 9.5)  # Ensure within Isaac Sim limits
            
            movement_rates[leg_id][joint_name] = float(joint_speed)

            action_idx += 1

    return target_angles, mid_angles, movement_rates


########## EPISODE MANAGEMENT ##########

##### start episode #####

def start_episode():
    """Start a new training episode"""
    global episode_counter, episode_reward, episode_states, episode_actions, episode_rewards, episode_values, episode_log_probs, episode_dones

    episode_counter += 1
    rewards.EPISODE_STEP = 0
    episode_reward = 0.0
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_values = []
    episode_log_probs = []
    episode_dones = []

    # COMPLETELY GUTTED - No more movement tracking
    # You now control movement rewards through your reward function

    logging.debug(f"🚀 Starting episode {episode_counter}\n")
    # Show initial orientation at episode start
    track_orientation()


##### end episode #####

def end_episode():
    """End current episode and save progress"""
    global episode_counter, episode_reward, episode_scores, average_score

    logging.debug(f"🎯 Episode {episode_counter} ended:")
    logging.debug(f"   📊 Steps: {rewards.EPISODE_STEP}")
    logging.debug(f"   📊 Final Reward: {episode_reward:.3f}\n")

    # Track episode scores for average calculation
    episode_scores.append(episode_reward)
    if len(episode_scores) > 100:  # Keep only last 100 episodes for recent average
        episode_scores.pop(0)
    
    # Calculate running average
    average_score = sum(episode_scores) / len(episode_scores)
    logging.info(f"   📊 Average Score (last {len(episode_scores)} episodes): {average_score:.3f}\n")

    # Save model periodically based on total steps (only if PPO policy is initialized)
    if total_steps % config.TRAINING_CONFIG['save_frequency'] == 0 and ppo_policy is not None:
        save_model(
            f"/home/matthewthomasbeck/Projects/Robot_Dog/model/ppo_steps_{total_steps}_episode_{episode_counter}_avg_{average_score:.2f}.pth")
        logging.info(f"💾 Model saved: steps_{total_steps}, episode_{episode_counter}, avg_score_{average_score:.2f}\n")
    elif ppo_policy is None:
        logging.warning(f"⚠️  Warning: PPO policy not initialized yet, skipping model save for episode {episode_counter}\n")
    else:
        logging.warning(f"📝 Episode {episode_counter} completed but not at save frequency ({config.TRAINING_CONFIG['save_frequency']} steps)\n")


##### check/reset episode #####

def check_and_reset_episode_if_needed():  # TODO compare with reset_episode()
    """
    Check if episode should be reset and trigger reset if needed.
    This function should be called from the main thread to integrate episode management.

    Returns:
        bool: True if episode was reset, False otherwise
    """
    global episode_counter, episode_reward

    import utilities.config as config

    # COMPLETELY GUTTED - No more automatic episode termination due to falling
    # You now control when episodes end through your reward system
    # The robot can fall and recover without ending the episode

    # COMPLETELY GUTTED - You now have complete control over warning systems
    # Add your custom warning logic here if desired

    # Check if episode has reached max steps
    if rewards.EPISODE_STEP >= config.TRAINING_CONFIG['max_steps_per_episode']:
        logging.debug(f"🎯 Episode {episode_counter} completed after {rewards.EPISODE_STEP} steps.\n")
        # Save model before resetting
        end_episode()
        
        # Monitor learning progress before resetting
        monitor_learning_progress()
        
        reset_episode()
        return True

    return False


##### reset episode #####

def reset_episode():
    """
    Reset the current episode by resetting Isaac Sim world and moving robot to neutral position.
    This is the critical function that was working in working_robot_reset.py
    """
    global episode_reward, episode_states, episode_actions, episode_counter, total_steps

    try:
        logging.info(
            f"(training.py): Episode {episode_counter} ending - Episode complete, resetting Isaac Sim world.\n")

        # CRITICAL: Save the model before resetting (if episode had any progress)
        if rewards.EPISODE_STEP > 0:
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
        neutral_position_isaac()  # Move to neutral position in Isaac Sim

        # Give Isaac Sim more steps to stabilize after neutral position
        if config.USE_SIMULATION and config.USE_ISAAC_SIM:
            for _ in range(5):
                config.ISAAC_WORLD.step(render=True)

        # Reset Python tracking variables
        rewards.EPISODE_STEP = 0
        episode_reward = 0.0
        episode_states = []
        episode_actions = []

        # COMPLETELY GUTTED - No more movement tracking reset
        # You now control movement tracking through your reward function

        # Increment episode counter
        episode_counter += 1

        logging.info(f"(training.py): Episode {episode_counter} reset complete - World and robot state reset.\n")

        # Log learning progress every 10 episodes
        if episode_counter % 10 == 0:
            logging.debug(f"🎯 Training Progress: {episode_counter} episodes completed.\n")
            if episode_states: # Check if episode_states is not empty
                logging.debug(f"   📊 Episode data size: {len(episode_states)}\n")
            if ppo_policy is not None:
                logging.debug(f"   🧠 PPO policy ready for training.\n")

    except Exception as e:
        logging.error(f"(training.py): Failed to reset episode: {e}\n")
        # Don't crash - try to continue with next episode
        episode_counter += 1
        rewards.EPISODE_STEP = 0
        episode_reward = 0.0


########## MODEL FUNCTIONS ##########

##### save trained model #####

def save_model(filepath):
    """Save the current PPO model"""
    if ppo_policy:

        logging.info(f"💾 Saving PPO model to: {filepath}")
        logging.info(f"   📊 Current episode: {episode_counter}")
        logging.info(f"   📊 Current step: {rewards.EPISODE_STEP}")
        logging.info(f"   📊 Total steps: {total_steps}")
        logging.info(f"   📊 Episode reward: {episode_reward:.4f}\n")

        # Create the checkpoint data
        checkpoint = {
            'actor_critic_state_dict': ppo_policy.actor_critic.state_dict(),
            'optimizer_state_dict': ppo_policy.optimizer.state_dict(),
            'episode_counter': episode_counter,
            'total_steps': total_steps,
            'episode_reward': episode_reward,
        }

        # Verify checkpoint data before saving
        logging.info(f"   🔍 Checkpoint contains {len(checkpoint)} keys:")
        for key, value in checkpoint.items():
            if 'state_dict' in key:
                if isinstance(value, dict):
                    logging.info(f"      ✅ {key}: {len(value)} layers\n")
                else:
                    logging.warning(f"      ❌ {key}: Invalid type {type(value)}\n")
            else:
                logging.info(f"      📊 {key}: {value}\n")

        # Save the model
        torch.save(checkpoint, filepath)

        # Verify the file was created and has content
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            logging.info(f"   ✅ Model saved successfully! File size: {file_size:.1f} MB\n")
        else:
            logging.error(f"   ❌ Failed to save model - file not created.\n")

        logging.info(f"Model saved to: {filepath}")
        logging.info(f"   📊 Current average score: {average_score:.3f}\n")
    else:
        logging.error(f"❌ Cannot save model - PPO policy not initialized.\n")


##### load the model #####

def load_model(filepath):
    """Load a PPO model from file"""
    global ppo_policy, episode_counter, total_steps, episode_reward

    if ppo_policy and os.path.exists(filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        ppo_policy.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        ppo_policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore training state
        episode_counter = checkpoint.get('episode_counter', 0)
        total_steps = checkpoint.get('total_steps', 0)
        episode_reward = checkpoint.get('episode_reward', 0.0)

        logging.info(f"Model loaded from: {filepath}")
        logging.info(f"  - Episode: {episode_counter}")
        logging.info(f"  - Total steps: {total_steps}")
        logging.info(f"  - Episode reward: {episode_reward:.4f}\n")
        return True
    return False

##### monitor learning progress #####

def monitor_learning_progress():
    """
    Monitor the agent's learning progress and detect potential issues.
    This helps identify if the agent is stuck in a loop of falling behaviors.
    """
    global episode_counter, episode_reward, episode_states
    
    if not episode_states:
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
        
        logging.info(f"📊 Learning Progress Analysis:")
        logging.info(f"   🎯 Recent episodes: {len(recent_rewards)}")
        logging.info(f"   📈 Average reward: {avg_reward:.3f}")
        logging.info(f"   📉 Worst reward: {min_reward:.3f}")
        logging.info(f"   🧠 PPO policy: Ready for training.\n")
        
        # COMPLETELY GUTTED - No more automatic fall loop detection
        # You now control what constitutes problematic behavior through your reward system
        
        # Detect if agent is improving
        if len(monitor_learning_progress.episode_rewards) >= 10:
            first_half = monitor_learning_progress.episode_rewards[:10]
            second_half = monitor_learning_progress.episode_rewards[-10:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg > first_avg:
                logging.info(f"✅ Agent is improving! First 10: {first_avg:.3f}, Last 10: {second_avg:.3f}\n")
            else:
                logging.info(f"❌ Agent not improving. First 10: {first_avg:.3f}, Last 10: {second_avg:.3f}\n")


########## RANDOM COMMANDS AND INTENSITIES ##########

##### random commands #####

def get_random_command(phase=1): # returns semirandom, realistic command combinations based on phase
    global previous_command

    # Phase 1: Basic forward movement and turning only
    if phase == 1:
        command_combinations = [
            # Single movements only
            'w', 'arrowleft', 'arrowright'
        ]
        
        # Simple transition weights for phase 1
        command_weights = {
            'w': {
                'w': 0.6, 'arrowleft': 0.2, 'arrowright': 0.2
            },
            'arrowleft': {
                'w': 0.4, 'arrowleft': 0.4, 'arrowright': 0.2
            },
            'arrowright': {
                'w': 0.4, 'arrowleft': 0.2, 'arrowright': 0.4
            }
        }
    
    # Phase 2: Full movement + rotation combinations
    elif phase == 2:
        command_combinations = [
            # Single movements
            'w', 's', 'a', 'd',
            # Movement + rotation
            'w+arrowleft', 'w+arrowright', 's+arrowleft', 's+arrowright',
            'a+arrowleft', 'a+arrowright', 'd+arrowleft', 'd+arrowright'
        ]
        
        # Transition weights for phase 2
        command_weights = {
            # Single forward movement
            'w': {
                'w': 0.3, 'w+arrowleft': 0.2, 'w+arrowright': 0.2, 's': 0.1, 'a': 0.1, 'd': 0.1
            },
            # Single backward movement  
            's': {
                's': 0.3, 's+arrowleft': 0.2, 's+arrowright': 0.2, 'w': 0.1, 'a': 0.1, 'd': 0.1
            },
            # Single left movement
            'a': {
                'a': 0.3, 'a+arrowleft': 0.2, 'a+arrowright': 0.2, 'w': 0.1, 's': 0.1, 'd': 0.1
            },
            # Single right movement
            'd': {
                'd': 0.3, 'd+arrowleft': 0.2, 'd+arrowright': 0.2, 'w': 0.1, 's': 0.1, 'a': 0.1
            },
            # Movement + rotation combinations
            'w+arrowleft': {'w': 0.3, 'w+arrowleft': 0.3, 'arrowleft': 0.2, 'a': 0.1, 's': 0.1},
            'w+arrowright': {'w': 0.3, 'w+arrowright': 0.3, 'arrowright': 0.2, 'd': 0.1, 's': 0.1},
            's+arrowleft': {'s': 0.3, 's+arrowleft': 0.3, 'arrowleft': 0.2, 'a': 0.1, 'w': 0.1},
            's+arrowright': {'s': 0.3, 's+arrowright': 0.3, 'arrowright': 0.2, 'd': 0.1, 'w': 0.1},
            'a+arrowleft': {'a': 0.3, 'a+arrowleft': 0.3, 'arrowleft': 0.2, 'w': 0.1, 's': 0.1},
            'a+arrowright': {'a': 0.3, 'a+arrowright': 0.3, 'arrowright': 0.2, 'w': 0.1, 's': 0.1},
            'd+arrowleft': {'d': 0.3, 'd+arrowleft': 0.3, 'arrowleft': 0.2, 'w': 0.1, 's': 0.1},
            'd+arrowright': {'d': 0.3, 'd+arrowright': 0.3, 'arrowright': 0.2, 'w': 0.1, 's': 0.1}
        }
    
    # Phase 3: Full complexity with diagonals and complex combinations
    elif phase == 3:
        command_combinations = [
            # Single movements
            'w', 's', 'a', 'd',
            # Diagonals
            'w+a', 'w+d', 's+a', 's+d',
            # Movement + rotation
            'w+arrowleft', 'w+arrowright', 's+arrowleft', 's+arrowright',
            'a+arrowleft', 'a+arrowright', 'd+arrowleft', 'd+arrowright',
            # Complex combinations (diagonal + rotation)
            'w+a+arrowleft', 'w+a+arrowright', 'w+d+arrowleft', 'w+d+arrowright',
            's+a+arrowleft', 's+a+arrowright', 's+d+arrowleft', 's+d+arrowright'
        ]
        
        # Full transition weights for phase 3
        command_weights = {
            # Single forward movement
            'w': {
                'w': 0.3, 'w+a': 0.15, 'w+d': 0.15, 'w+arrowleft': 0.1, 'w+arrowright': 0.1,
                'a': 0.05, 'd': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Single backward movement  
            's': {
                's': 0.3, 's+a': 0.15, 's+d': 0.15, 's+arrowleft': 0.1, 's+arrowright': 0.1,
                'a': 0.05, 'd': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Single left movement
            'a': {
                'a': 0.3, 'w+a': 0.15, 's+a': 0.15, 'a+arrowleft': 0.1, 'a+arrowright': 0.1,
                'w': 0.05, 's': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Single right movement
            'd': {
                'd': 0.3, 'w+d': 0.15, 's+d': 0.15, 'd+arrowleft': 0.1, 'd+arrowright': 0.1,
                'w': 0.05, 's': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Single left rotation
            'arrowleft': {
                'arrowleft': 0.25, 'w+arrowleft': 0.15, 's+arrowleft': 0.15, 'a+arrowleft': 0.1, 'd+arrowleft': 0.1,
                'w': 0.05, 's': 0.05, 'a': 0.05, 'd': 0.05, 'arrowright': 0.05
            },
            # Single right rotation
            'arrowright': {
                'arrowright': 0.25, 'w+arrowright': 0.15, 's+arrowright': 0.15, 'a+arrowright': 0.1, 'd+arrowright': 0.1,
                'w': 0.05, 's': 0.05, 'a': 0.05, 'd': 0.05, 'arrowleft': 0.05
            },
            # Diagonal forward-left
            'w+a': {
                'w+a': 0.25, 'w': 0.15, 'a': 0.15, 'w+a+arrowleft': 0.1, 'w+a+arrowright': 0.1,
                'w+arrowleft': 0.05, 'w+arrowright': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Diagonal forward-right
            'w+d': {
                'w+d': 0.25, 'w': 0.15, 'd': 0.15, 'w+d+arrowleft': 0.1, 'w+d+arrowright': 0.1,
                'w+arrowleft': 0.05, 'w+arrowright': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Diagonal backward-left
            's+a': {
                's+a': 0.25, 's': 0.15, 'a': 0.15, 's+a+arrowleft': 0.1, 's+a+arrowright': 0.1,
                's+arrowleft': 0.05, 's+arrowright': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Diagonal backward-right
            's+d': {
                's+d': 0.25, 's': 0.15, 'd': 0.15, 's+d+arrowleft': 0.1, 's+d+arrowright': 0.1,
                's+arrowleft': 0.05, 's+arrowright': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Movement + rotation combinations
            'w+arrowleft': {'w': 0.2, 'w+a': 0.15, 'w+arrowleft': 0.2, 'w+a+arrowleft': 0.1, 'a': 0.1, 'arrowleft': 0.1, 'w+d': 0.05, 'w+arrowright': 0.05, 's': 0.05},
            'w+arrowright': {'w': 0.2, 'w+d': 0.15, 'w+arrowright': 0.2, 'w+d+arrowright': 0.1, 'd': 0.1, 'arrowright': 0.1, 'w+a': 0.05, 'w+arrowleft': 0.05, 's': 0.05},
            's+arrowleft': {'s': 0.2, 's+a': 0.15, 's+arrowleft': 0.2, 's+a+arrowleft': 0.1, 'a': 0.1, 'arrowleft': 0.1, 's+d': 0.05, 's+arrowright': 0.05, 'w': 0.05},
            's+arrowright': {'s': 0.2, 's+d': 0.15, 's+arrowright': 0.2, 's+d+arrowright': 0.1, 'd': 0.1, 'arrowright': 0.1, 's+a': 0.05, 's+arrowleft': 0.05, 'w': 0.05},
            'a+arrowleft': {'a': 0.2, 'w+a': 0.15, 'a+arrowleft': 0.2, 'w+a+arrowleft': 0.1, 'w': 0.1, 'arrowleft': 0.1, 's+a': 0.05, 'a+arrowright': 0.05, 's': 0.05},
            'a+arrowright': {'a': 0.2, 's+a': 0.15, 'a+arrowright': 0.2, 's+a+arrowright': 0.1, 's': 0.1, 'arrowright': 0.1, 'w+a': 0.05, 'a+arrowleft': 0.05, 'w': 0.05},
            'd+arrowleft': {'d': 0.2, 's+d': 0.15, 'd+arrowleft': 0.2, 's+d+arrowleft': 0.1, 's': 0.1, 'arrowleft': 0.1, 'w+d': 0.05, 'd+arrowright': 0.05, 'w': 0.05},
            'd+arrowright': {'d': 0.2, 'w+d': 0.15, 'd+arrowright': 0.2, 'w+d+arrowright': 0.1, 'w': 0.1, 'arrowright': 0.1, 's+d': 0.05, 'd+arrowleft': 0.05, 's': 0.05},
            # Complex combinations
            'w+a+arrowleft': {'w+a': 0.2, 'w+a+arrowleft': 0.25, 'w': 0.15, 'a': 0.1, 'arrowleft': 0.1, 'w+arrowleft': 0.1, 'w+arrowright': 0.05, 's': 0.05},
            'w+a+arrowright': {'w+a': 0.2, 'w+a+arrowright': 0.25, 'w': 0.15, 'a': 0.1, 'arrowright': 0.1, 'w+arrowright': 0.1, 'w+arrowleft': 0.05, 's': 0.05},
            'w+d+arrowleft': {'w+d': 0.2, 'w+d+arrowleft': 0.25, 'w': 0.15, 'd': 0.1, 'arrowleft': 0.1, 'w+arrowleft': 0.1, 'w+arrowright': 0.05, 's': 0.05},
            'w+d+arrowright': {'w+d': 0.2, 'w+d+arrowright': 0.25, 'w': 0.15, 'd': 0.1, 'arrowright': 0.1, 'w+arrowright': 0.1, 'w+arrowleft': 0.05, 's': 0.05},
            's+a+arrowleft': {'s+a': 0.2, 's+a+arrowleft': 0.25, 's': 0.15, 'a': 0.1, 'arrowleft': 0.1, 's+arrowleft': 0.1, 's+arrowright': 0.05, 'w': 0.05},
            's+a+arrowright': {'s+a': 0.2, 's+a+arrowright': 0.25, 's': 0.15, 'a': 0.1, 'arrowright': 0.1, 's+arrowright': 0.1, 's+arrowleft': 0.05, 'w': 0.05},
            's+d+arrowleft': {'s+d': 0.2, 's+d+arrowleft': 0.25, 's': 0.15, 'd': 0.1, 'arrowleft': 0.1, 's+arrowleft': 0.1, 's+arrowright': 0.05, 'w': 0.05},
            's+d+arrowright': {'s+d': 0.2, 's+d+arrowright': 0.25, 's': 0.15, 'd': 0.1, 'arrowright': 0.1, 's+arrowright': 0.1, 's+arrowleft': 0.05, 'w': 0.05}
        }
    
    # Default fallback to phase 1
    else:
        command_combinations = ['w', 'arrowleft', 'arrowright']
        command_weights = {
            'w': {'w': 0.6, 'arrowleft': 0.2, 'arrowright': 0.2},
            'arrowleft': {'w': 0.4, 'arrowleft': 0.4, 'arrowright': 0.2},
            'arrowright': {'w': 0.4, 'arrowleft': 0.2, 'arrowright': 0.4}
        }
    
    # If this is the first command, choose randomly from current phase options
    if previous_command is None:
        command = random.choice(command_combinations)
        previous_command = command
        return command
    
    # Get weights for current previous command
    if previous_command in command_weights:
        weights = command_weights[previous_command]
    else:
        # Fallback: if previous command not in weights, choose randomly
        command = random.choice(command_combinations)
        previous_command = command
        return command
    
    # Convert weights to list for random.choices
    commands = list(weights.keys())
    weight_values = list(weights.values())
    
    # Choose command based on weighted probabilities
    command = random.choices(commands, weights=weight_values, k=1)[0]
    
    # Update previous command
    previous_command = command
    
    return command

##### random intensity #####

def get_random_intensity(phase=1): # returns intensity based on phase
    global previous_intensity

    # Phase 1 & 2: Intensity locked at 10 for stability
    if phase in [1, 2]:
        return 10
    
    # Phase 3: Full intensity range 1-10 with realistic transitions
    elif phase == 3:
        # If this is the first intensity, start with moderate
        if previous_intensity is None:
            intensity = random.choice([4, 5, 6, 7])
            previous_intensity = intensity
            return intensity
        
        # Define intensity change probabilities for realistic movement
        # Higher chance of staying close to previous intensity, with gradual changes
        intensity_change = random.choices(
            [-3, -2, -1, 0, 1, 2, 3],  # Possible changes
            weights=[0.05, 0.1, 0.25, 0.2, 0.25, 0.1, 0.05],  # Weights favoring small changes
            k=1
        )[0]
        
        # Calculate new intensity
        new_intensity = previous_intensity + intensity_change
        
        # Clamp to valid range 1-10
        new_intensity = max(1, min(10, new_intensity))
        
        # Update previous intensity
        previous_intensity = new_intensity
        
        return new_intensity
    
    # Default fallback to phase 1 (intensity 10)
    else:
        return 10
    