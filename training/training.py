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
import torch
import concurrent.futures
import time

# Optimize PyTorch for performance
torch.set_num_threads(24)  # Use all 24 CPU cores
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

##### import necessary functions #####

from movement.isaac_joints import neutral_position_isaac
from training.agents import *
import training.rewards as rewards
from training.orientation import track_orientation


########## CREATE DEPENDENCIES ##########

##### global PPO instance and episode data #####

ppo_policies = []  # Array of PPO policies for each robot
episode_rewards = []  # Array of episode rewards for each robot
episode_states = []  # Array of episode states for each robot
episode_actions = []  # Array of episode actions for each robot
episode_rewards_list = []  # Array of episode rewards lists for each robot
episode_values = []  # Array of episode values for each robot
episode_log_probs = []  # Array of episode log probs for each robot
episode_dones = []  # Array of episode dones for each robot
total_steps_list = []  # Array of total steps for each robot
episode_scores = []  # Array of episode scores for each robot
average_scores = []  # Array of average scores for each robot

##### random command and intensity #####

previous_command = None
previous_intensity = None





##################################################
############### ISAAC SIM TRAINING ###############
##################################################


########## MAIN LOOP ##########

##### initialize training #####

def initialize_training():
    """Initialize the complete training system for multiple robots"""
    global ppo_policies, episode_rewards, episode_states, episode_actions, episode_rewards_list, episode_values, episode_log_probs, episode_dones, total_steps_list, episode_scores, average_scores

    logging.debug("(training.py): Initializing multi-robot training system...\n")
    os.makedirs(config.MODELS_DIRECTORY, exist_ok=True)

    # Initialize PPO for each robot
    state_dim = 19  # 12 joints + 6 commands + 1 intensity
    action_dim = 36  # 12 mid + 12 target angles + 12 velocity values
    max_action = config.TRAINING_CONFIG['max_action']
    
    num_robots = config.MULTI_ROBOT_CONFIG['num_robots']
    
    # Clear and reinitialize arrays for each robot
    ppo_policies = []
    episode_rewards = []
    episode_states = []
    episode_actions = []
    episode_rewards_list = []
    episode_values = []
    episode_log_probs = []
    episode_dones = []
    total_steps_list = []
    episode_scores = []
    average_scores = []
    
    for robot_idx in range(num_robots):
        # Create PPO policy for this robot
        ppo_policy = PPO(state_dim, action_dim, max_action)
        ppo_policies.append(ppo_policy)
        
        # Initialize tracking variables for this robot
        episode_rewards.append(0.0)
        episode_states.append([])
        episode_actions.append([])
        episode_rewards_list.append([])
        episode_values.append([])
        episode_log_probs.append([])
        episode_dones.append([])
        total_steps_list.append(0)
        episode_scores.append([])
        average_scores.append(0.0)

    # Try to load the latest saved model to continue training
    latest_model = find_latest_model()
    if latest_model:
        logging.debug(f"🔄 Loading latest model: {latest_model}...\n")
        if load_model(latest_model):
            logging.info(f"✅ Successfully loaded model from step {sum(total_steps_list)}, episode {episode_counter}\n")
        else:
            logging.warning(f"❌ Failed to load model, starting fresh.\n")
            episode_counter = 0
            total_steps_list = [0] * num_robots
    else:
        logging.warning(f"🆕 No saved models found, starting fresh training.\n")
        episode_counter = 0
        total_steps_list = [0] * num_robots

    logging.info(f"Multi-robot training system initialized:")
    logging.info(f"  - Number of robots: {num_robots}")
    logging.info(f"  - State dimension: {state_dim}")
    logging.info(f"  - Action dimension: {action_dim}")
    logging.info(f"  - Models directory: {config.MODELS_DIRECTORY}")
    logging.info(f"  - PPO policies created: {len(ppo_policies)}")
    logging.info(f"  - Starting from episode: {episode_counter}")
    logging.info(f"  - Starting from step: {sum(total_steps_list)}\n")


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
    global total_steps

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
        logging.info(f"(training.py): Episode reset in main loop integration\n")
        return True

    return False


########## AGENT FUNCTIONS ##########

##### summon standard agent #####

def get_rl_action_standard(state, commands, intensity, frame):  # will eventually take camera frame as input
    pass


##### summon blind agent #####

def get_single_robot_action(robot_idx, current_angles, commands, intensity):
    """
    Get action for a single robot - used for parallel processing
    """
    import time
    single_start = time.time()
    global ppo_policies, episode_rewards, episode_states, episode_actions, episode_rewards_list, episode_values, episode_log_probs, episode_dones, total_steps_list

    # STEP 1 COMPLETE: Safety checks already done in main function
    # Get the PPO policy for this specific robot
    ppo_policy = ppo_policies[robot_idx]
    
    # Get robot-specific tracking variables
    episode_reward = episode_rewards[robot_idx]
    episode_states_robot = episode_states[robot_idx]
    episode_actions_robot = episode_actions[robot_idx]
    episode_rewards_robot = episode_rewards_list[robot_idx]
    episode_values_robot = episode_values[robot_idx]
    episode_log_probs_robot = episode_log_probs[robot_idx]
    episode_dones_robot = episode_dones[robot_idx]
    total_steps = total_steps_list[robot_idx]

    # Build state vector for this robot
    state = []
    state_creation_start = time.time()

    # 1. Extract and normalize joint angles (12D) for this robot
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        for joint_name in ['hip', 'upper', 'lower']:
            angle = current_angles[leg_id][joint_name]
            
            # Get joint limits for normalization from static config (same for all robots)
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
                normalized_angle = 0.0
                
            state.append(normalized_angle)

    # 2. Encode commands (6D one-hot for movement commands only) - SAME FOR ALL ROBOTS
    command_list = commands

    command_encoding = [
        1.0 if 'w' in command_list else 0.0,
        1.0 if 's' in command_list else 0.0,
        1.0 if 'a' in command_list else 0.0,
        1.0 if 'd' in command_list else 0.0,
        1.0 if 'arrowleft' in command_list else 0.0,
        1.0 if 'arrowright' in command_list else 0.0
    ]
    
    state.extend(command_encoding)

    # 3. Normalize intensity (1D) - SAME FOR ALL ROBOTS
    intensity_normalized = (float(intensity) - 5.5) / 4.5
    state.append(intensity_normalized)
    
    # Convert to numpy array and validate state size
    state = np.array(state, dtype=np.float32)
    
    # Validate state size: 12 (joints) + 6 (commands) + 1 (intensity) = 19
    expected_state_size = 19
    if len(state) != expected_state_size:
        raise ValueError(f"State size mismatch: expected {expected_state_size}, got {len(state)}")

    state_creation_time = time.time() - state_creation_start

    # Get action from PPO for this robot (stochastic during training, deterministic for deployment)
    inference_start = time.time()
    action = ppo_policy.select_action(state, deterministic=False)
    inference_time = time.time() - inference_start

    # Store experience for training
    if episode_states_robot and episode_actions_robot:
        # Calculate reward using the dedicated reward function for this robot
        reward = rewards.calculate_step_reward(robot_idx, current_angles, commands, intensity)

        # Update episode reward for tracking
        episode_reward += reward

        done = rewards.EPISODE_STEP >= config.TRAINING_CONFIG['max_steps_per_episode']

        # Add to episode data lists
        episode_states_robot.append(state)
        episode_actions_robot.append(action)
        episode_rewards_robot.append(reward)
        episode_values_robot.append(None)
        episode_log_probs_robot.append(None)
        episode_dones_robot.append(done)
        
        # Train PPO at episode end
        if done and len(episode_states_robot) >= config.TRAINING_CONFIG['batch_size']:
            print(f"Robot {robot_idx}: Training PPO at episode end, episode data size: {len(episode_states_robot)}")
            
            # Convert to tensors
            states = torch.FloatTensor(episode_states_robot)
            actions = torch.FloatTensor(episode_actions_robot)
            rewards_tensor = torch.FloatTensor(episode_rewards_robot)
            dones = torch.FloatTensor(episode_dones_robot)
            
            # Calculate log probabilities for all actions in the episode
            with torch.no_grad():
                _, log_probs, _, _ = ppo_policy.actor_critic.get_action_and_value(states, actions)
                log_probs = log_probs.squeeze(-1)
            
            # Get final value estimate for GAE calculation
            with torch.no_grad():
                final_value = ppo_policy.actor_critic.get_value(states[-1:]).item()
            
            # Train PPO on episode data
            ppo_policy.train(states, actions, log_probs, rewards_tensor, dones, final_value)
            
            # Reset episode data lists for this robot
            episode_states[robot_idx] = []
            episode_actions[robot_idx] = []
            episode_rewards_list[robot_idx] = []
            episode_values[robot_idx] = []
            episode_log_probs[robot_idx] = []
            episode_dones[robot_idx] = []

        # Save model periodically based on total steps
        if total_steps % config.TRAINING_CONFIG['save_frequency'] == 0 and ppo_policy is not None:
            save_model(
                f"/home/matthewthomasbeck/Projects/Robot_Dog/model/ppo_robot_{robot_idx}_steps_{total_steps}_reward_{episode_reward:.2f}.pth")
            print(f"Robot {robot_idx}: Model saved: steps_{total_steps}")

    # Update tracking variables for this robot
    episode_states[robot_idx].append(state)
    episode_actions[robot_idx].append(action)
    
    # Get value and log_prob for PPO training
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        _, log_prob, _, value = ppo_policy.actor_critic.get_action_and_value(state_tensor, torch.FloatTensor(action).unsqueeze(0))
        episode_values[robot_idx].append(value.item())
        episode_log_probs[robot_idx].append(log_prob.item())
    
    total_steps_list[robot_idx] += 1

    # Convert 36D action vector to joint angles and velocities for this robot
    conversion_start = time.time()
    target_angles = {}
    mid_angles = {}
    movement_rates = {}

    action_idx = 0

    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        target_angles[leg_id] = {}
        mid_angles[leg_id] = {}
        movement_rates[leg_id] = {}

        for joint_name in ['hip', 'upper', 'lower']:
            # Get joint limits from static config (same for all robots)
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
            mid_angles[leg_id][joint_name] = round(float(mid_angle), 4)  # Round to 4 decimal places

            # Convert target action (-1 to 1) to joint angle
            target_action = action[action_idx + 12]
            target_angle = min_angle + (target_action + 1.0) * 0.5 * (max_angle - min_angle)
            target_angle = np.clip(target_angle, min_angle, max_angle)
            target_angles[leg_id][joint_name] = round(float(target_angle), 4)  # Round to 4 decimal places

            # Convert velocity action (-1 to 1) to movement rate in radians/second
            velocity_action = action[action_idx + 24]
            joint_speed = (velocity_action + 1.0) * 4.75
            joint_speed = np.clip(joint_speed, 0.0, 9.5)
            
            movement_rates[leg_id][joint_name] = round(float(joint_speed), 3)  # Round to 3 decimal places

            action_idx += 1

    conversion_time = time.time() - conversion_start
    total_single_time = time.time() - single_start
    
    if robot_idx == 0:  # Only log first robot to avoid spam
        print(f"  Robot {robot_idx} - State creation: {state_creation_time:.4f}s, Model inference: {inference_time:.4f}s, Action conversion: {conversion_time:.4f}s, Total: {total_single_time:.4f}s")

    return target_angles, mid_angles, movement_rates

def get_rl_action_blind(all_current_angles, commands, intensity):
    """
    PPO RL agent that takes current joint angles for ALL robots, commands, and intensity as state
    and outputs predictions for ALL robots using parallel processing.
    
    Args:
        all_current_angles: List of current joint angles for each robot
        commands: Movement commands
        intensity: Movement intensity
    
    Returns:
        all_target_angles: List of target joint angles for each robot
        all_mid_angles: List of mid joint angles for each robot  
        all_movement_rates: List of movement rates for each robot
    """
    import time
    start_time = time.time()
    global ppo_policies, episode_rewards, episode_states, episode_actions, episode_rewards_list, episode_values, episode_log_probs, episode_dones, total_steps_list

    # Initialize training system if not done yet
    if not ppo_policies or len(ppo_policies) == 0:
        initialize_training()
        start_episode()

    num_robots = config.MULTI_ROBOT_CONFIG['num_robots']
    
    # STEP 1: STREAMLINED SAFETY CHECKS - Do all validation upfront, cache results
    if num_robots > len(ppo_policies):
        logging.error(f"(training.py): Not enough PPO policies ({len(ppo_policies)}) for {num_robots} robots")
        num_robots = len(ppo_policies)  # Limit to available policies
        
    if num_robots > len(episode_rewards):
        logging.error(f"(training.py): Not enough tracking arrays ({len(episode_rewards)}) for {num_robots} robots")
        num_robots = len(episode_rewards)  # Limit to available tracking arrays
        
    # No more config validation needed - we use static SERVO_CONFIG for all robots
    
    # CRITICAL: Check episode completion but DON'T reset from worker thread
    episode_needs_reset = False
    if rewards.EPISODE_STEP >= config.TRAINING_CONFIG['max_steps_per_episode']:
        episode_needs_reset = True
        print(f"Episode completed after {rewards.EPISODE_STEP} steps! (signaling main thread)")

    # Store reset signal for main thread to check
    config.EPISODE_NEEDS_RESET = episode_needs_reset

    # Log episode progress every 100 steps
    if rewards.EPISODE_STEP % 100 == 0:
        print(f"Step {rewards.EPISODE_STEP}, Total Steps: {sum(total_steps_list)}")
    
    if rewards.EPISODE_STEP % 50 == 0:
        # Track orientation for all robots
        for robot_idx in range(num_robots):
            track_orientation(robot_idx)
    
    # Direct processing - collect results immediately like the OLD function
    all_target_angles = []
    all_mid_angles = []
    all_movement_rates = []
    
    robot_processing_start = time.time()
    for robot_idx in range(num_robots):
        # Process each robot directly - no thread pool overhead
        single_robot_start = time.time()
        target_angles, mid_angles, movement_rates = get_single_robot_action(robot_idx, all_current_angles[robot_idx], commands, intensity)
        single_robot_time = time.time() - single_robot_start
        
        # Add results directly to arrays (no future.result() calls)
        if target_angles is not None:
            all_target_angles.append(target_angles)
            all_mid_angles.append(mid_angles)
            all_movement_rates.append(movement_rates)
        
        if robot_idx == 0:  # Only log first robot to avoid spam
            print(f"Robot {robot_idx} processing time: {single_robot_time:.4f}s")
    
    robot_processing_time = time.time() - robot_processing_start
    total_time = time.time() - start_time
    
    # CRITICAL: Only increment episode step ONCE per call, not per robot
    rewards.EPISODE_STEP += 1

    print(f"get_rl_action_blind total time: {total_time:.4f}s (robot processing: {robot_processing_time:.4f}s)")
    
    # Return predictions for ALL robots
    return all_target_angles, all_mid_angles, all_movement_rates


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
    track_orientation(robot_idx)


##### end episode #####

def end_episode():
    """End current episode - just track progress, don't save models"""
    global episode_reward, episode_scores, average_score
    
    # Track episode scores for average calculation
    episode_scores.append(episode_reward)
    if len(episode_scores) > 100:
        episode_scores.pop(0)
    
    # Calculate running average
    average_score = sum(episode_scores) / len(episode_scores)
    logging.info(f"   📊 Average Score (last {len(episode_scores)} episodes): {average_score:.3f}\n")

    # TODO difference here- no longer save model at end of episode


##### check/reset episode #####

def check_and_reset_episode_if_needed():  # TODO compare with reset_episode()
    """
    Check if episode should be reset and trigger reset if needed.
    This function should be called from the main thread to integrate episode management.

    Returns:
        bool: True if episode was reset, False otherwise
    """
    global episode_reward

    import utilities.config as config

    # COMPLETELY GUTTED - No more automatic episode termination due to falling
    # You now control when episodes end through your reward system
    # The robot can fall and recover without ending the episode

    # COMPLETELY GUTTED - You now have complete control over warning systems
    # Add your custom warning logic here if desired

    # Check if episode has reached max steps
    if rewards.EPISODE_STEP >= config.TRAINING_CONFIG['max_steps_per_episode']:
        logging.debug(f"🎯 Episode completed after {rewards.EPISODE_STEP} steps.\n")
        # Save model before resetting
        end_episode()

        # TODO difference here- no longer monitor learning progress

        reset_episode()
        return True

    return False


##### reset episode #####

def reset_episode():
    """
    Reset the current episode by resetting Isaac Sim world and moving robot to neutral position.
    This is the critical function that was working in working_robot_reset.py
    """
    global episode_reward, episode_states, episode_actions, total_steps, episode_counter

    try:
        logging.info(
            f"(training.py): Episode ending - Episode complete, resetting Isaac Sim world.\n")

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

        episode_counter += 1

        logging.info(f"(training.py): Episode reset complete - World and robot state reset.\n")

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
        logging.info(f"   📊 Current step: {rewards.EPISODE_STEP}")
        logging.info(f"   📊 Total steps: {total_steps}")
        logging.info(f"   📊 Episode reward: {episode_reward:.4f}\n")

        # TODO difference here- no longer save episode counter in the checkpoint
        checkpoint = {
            'actor_critic_state_dict': ppo_policy.actor_critic.state_dict(),
            'optimizer_state_dict': ppo_policy.optimizer.state_dict(),
            'total_steps': total_steps,
            'episode_reward': episode_reward
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
        episode_counter = 0
        total_steps = checkpoint.get('total_steps', 0)
        episode_reward = checkpoint.get('episode_reward', 0.0)

        logging.info(f"Model loaded from: {filepath}")
        logging.info(f"  - Total steps: {total_steps}")
        logging.info(f"  - Episode reward: {episode_reward:.4f}\n")
        return True
    return False


# TODO difference here- monitor learning progress function deleted


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
                'w': 0.9, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            'arrowleft': {
                'w': 0.05, 'arrowleft': 0.9, 'arrowright': 0.05
            },
            'arrowright': {
                'w': 0.05, 'arrowleft': 0.05, 'arrowright': 0.9
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
    