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

##### global TD3 instance and replay buffer #####

td3_policy = None
replay_buffer = None
episode_reward = 0.0
episode_counter = 0
last_state = None
last_action = None
total_steps = 0

# Track average episode scores for model naming
episode_scores = []  # List of all episode final scores
average_score = 0.0  # Running average of episode scores

##### models derectory #####

models_dir = "/home/matthewthomasbeck/Projects/Robot_Dog/model"





##################################################
############### ISAAC SIM TRAINING ###############
##################################################


########## MAIN LOOP ##########

##### initialize training #####

def initialize_training():
    """Initialize the complete training system"""
    global td3_policy, replay_buffer, episode_counter, total_steps

    logging.debug("Initializing training system...\n")
    os.makedirs(models_dir, exist_ok=True)

    # Initialize TD3
    state_dim = 21  # 12 joints + 8 commands + 1 intensity
    action_dim = 24  # 12 mid + 12 target angles
    max_action = config.TRAINING_CONFIG['max_action']

    td3_policy = TD3(state_dim, action_dim, max_action)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=100000)

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
    logging.info(f"  - Models directory: {models_dir}")
    logging.info(f"  - TD3 policy created: {td3_policy is not None}")
    logging.info(f"  - Replay buffer created: {replay_buffer is not None}")
    logging.info(f"  - Starting from episode: {episode_counter}")
    logging.info(f"  - Starting from step: {total_steps}\n")


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
    TD3 RL agent that takes current joint angles, commands, and intensity as state
    and outputs 24 values (12 mid angles + 12 target angles)

    NOTE: This function runs in worker threads, so it CANNOT call Isaac Sim reset functions.
    It only signals that a reset is needed - the main thread handles actual resets.
    """
    global td3_policy, replay_buffer, episode_reward, episode_counter
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
        noise = np.random.normal(0, config.TRAINING_CONFIG['exploration_noise'], size=action.shape)
        action = np.clip(action + noise, -config.TRAINING_CONFIG['max_action'], config.TRAINING_CONFIG['max_action'])
    else:
        action = td3_policy.select_action(state)
    
    # COMPLETELY GUTTED - No more automatic fall tracking
    # You now control fall recovery through your reward system

    # Store experience for training
    if last_state is not None and last_action is not None:
        # Calculate reward using the dedicated reward function
        reward = rewards.calculate_step_reward(current_angles, commands, intensity)

        # Update episode reward for tracking
        episode_reward += reward

        # COMPLETELY GUTTED - No more automatic episode termination due to falling
        # You now control when episodes end through your reward system
        done = rewards.EPISODE_STEP >= config.TRAINING_CONFIG['max_steps_per_episode']

        # CRITICAL: Add to replay buffer BEFORE checking for episode reset
        # This ensures the falling experience is captured
        replay_buffer.add(last_state, last_action, reward, state, done)
        
        # Log experience collection for debugging
        if done:
            print(f"💾 Experience collected: State={len(last_state)}D, Action={len(last_action)}D, Reward={reward:.3f}, Done={done}")
            print(f"   📊 Replay buffer size: {len(replay_buffer)}")

        # Train TD3 periodically
        if total_steps % config.TRAINING_CONFIG['training_frequency'] == 0 and len(replay_buffer) >= config.TRAINING_CONFIG[
            'batch_size']:
            # Train the agent and log the experience (reduced frequency to avoid spam)
            if total_steps % 50 == 0:  # Only log every 50 steps
                print(f"🧠 Training TD3 at step {total_steps}, buffer size: {len(replay_buffer)}")
            td3_policy.train(replay_buffer, config.TRAINING_CONFIG['batch_size'])
            
            # COMPLETELY GUTTED - No more automatic fall-related logging
            # You now control what gets logged through your reward system

        # Save model periodically based on total steps
        if total_steps % config.TRAINING_CONFIG['save_frequency'] == 0 and td3_policy is not None:
            save_model(
                f"/home/matthewthomasbeck/Projects/Robot_Dog/model/td3_steps_{total_steps}_episode_{episode_counter}_reward_{episode_reward:.2f}.pth")
            print(f"💾 Model saved: steps_{total_steps}, episode_{episode_counter}")

    # Update tracking variables
    last_state = state.copy()
    last_action = action.copy()
    rewards.EPISODE_STEP += 1
    total_steps += 1

    # Convert action (-1 to 1) to joint angles
    target_angles = {}
    mid_angles = {}
    movement_rates = {}

    action_idx = 0

    for leg_id in ['FL', 'FR', 'BL', 'BR']:

        target_angles[leg_id] = {}
        mid_angles[leg_id] = {}
        movement_rates[leg_id] = {'speed': 9.52}

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


########## EPISODE MANAGEMENT ##########

##### start episode #####

def start_episode():
    """Start a new training episode"""
    global episode_counter, episode_reward, last_state, last_action

    episode_counter += 1
    rewards.EPISODE_STEP = 0
    episode_reward = 0.0
    last_state = None
    last_action = None

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

    # Save model periodically based on total steps (only if TD3 policy is initialized)
    if total_steps % config.TRAINING_CONFIG['save_frequency'] == 0 and td3_policy is not None:
        save_model(
            f"/home/matthewthomasbeck/Projects/Robot_Dog/model/td3_steps_{total_steps}_episode_{episode_counter}_avg_{average_score:.2f}.pth")
        logging.info(f"💾 Model saved: steps_{total_steps}, episode_{episode_counter}, avg_score_{average_score:.2f}\n")
    elif td3_policy is None:
        logging.warning(f"⚠️  Warning: TD3 policy not initialized yet, skipping model save for episode {episode_counter}\n")
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


##### reset episode ##### TODO compare with check_and_reset_episode_if_needed()

def reset_episode():
    """
    Reset the current episode by resetting Isaac Sim world and moving robot to neutral position.
    This is the critical function that was working in working_robot_reset.py
    """
    global episode_reward, last_state, last_action, episode_counter, total_steps

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
        last_state = None
        last_action = None

        # COMPLETELY GUTTED - No more movement tracking reset
        # You now control movement tracking through your reward function

        # Increment episode counter
        episode_counter += 1

        logging.info(f"(training.py): Episode {episode_counter} reset complete - World and robot state reset.\n")

        # Log learning progress every 10 episodes
        if episode_counter % 10 == 0:
            logging.debug(f"🎯 Training Progress: {episode_counter} episodes completed.\n")
            if replay_buffer is not None:
                logging.debug(f"   📊 Replay buffer size: {len(replay_buffer)}\n")
            if td3_policy is not None:
                logging.debug(f"   🧠 TD3 policy ready for training.\n")

    except Exception as e:
        logging.error(f"(training.py): Failed to reset episode: {e}\n")
        # Don't crash - try to continue with next episode
        episode_counter += 1
        rewards.EPISODE_STEP = 0
        episode_reward = 0.0


########## MODEL FUNCTIONS ##########

##### save trained model #####

def save_model(filepath):
    """Save the current TD3 model"""
    if td3_policy:

        logging.info(f"💾 Saving TD3 model to: {filepath}")
        logging.info(f"   📊 Current episode: {episode_counter}")
        logging.info(f"   📊 Current step: {rewards.EPISODE_STEP}")
        logging.info(f"   📊 Total steps: {total_steps}")
        logging.info(f"   📊 Episode reward: {episode_reward:.4f}\n")

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
        logging.error(f"❌ Cannot save model - TD3 policy not initialized.\n")


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

        logging.info(f"Model loaded from: {filepath}")
        logging.info(f"  - Episode: {episode_counter}")
        logging.info(f"  - Total steps: {total_steps}")
        logging.info(f"  - Episode reward: {episode_reward:.4f}\n")
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
        
        logging.info(f"📊 Learning Progress Analysis:")
        logging.info(f"   🎯 Recent episodes: {len(recent_rewards)}")
        logging.info(f"   📈 Average reward: {avg_reward:.3f}")
        logging.info(f"   📉 Worst reward: {min_reward:.3f}")
        logging.info(f"   🧠 Replay buffer: {len(replay_buffer)} experiences\n")
        
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
