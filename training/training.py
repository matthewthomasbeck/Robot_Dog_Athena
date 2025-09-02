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

##### import necessary functions #####

from movement.isaac_joints import neutral_position_isaac
from training.agents import *
import training.rewards as rewards
from training.orientation import track_orientation
import training.episodes as episodes


########## CREATE DEPENDENCIES ##########

##### global PPO instance and episode data #####

ppo_policy = None
total_steps = 0

##### multi-agent training data #####

# Shared experience buffer for all agents
shared_experience_buffer = {
    'states': [],
    'actions': [],
    'rewards': [],
    'values': [],
    'log_probs': [],
    'dones': [],
    'agent_ids': []  # Track which agent generated each experience
}

# Per-agent data
agent_data = {}  # Will be initialized with num_robots agents

##### random command and intensity #####

previous_command = None
previous_intensity = None





##################################################
############### ISAAC SIM TRAINING ###############
##################################################


########## TRAINING FUNCTIONS ##########

##### initialize training #####

def initialize_training():
    """Initialize the complete multi-agent training system"""
    global ppo_policy, total_steps, agent_data

    logging.debug("(training.py): Initializing multi-agent training system...\n")
    os.makedirs(config.MODELS_DIRECTORY, exist_ok=True)

    # Initialize PPO (shared across all agents)
    state_dim = 19  # 12 joints + 6 commands + 1 intensity
    action_dim = 36  # 12 mid + 12 target angles + 12 velocity values
    max_action = config.TRAINING_CONFIG['max_action']

    ppo_policy = PPO(state_dim, action_dim, max_action)

    # Initialize per-agent data
    agent_data = initialize_agent_data()

    # Try to load the latest saved model to continue training
    latest_model = find_latest_model()
    if latest_model:
        logging.debug(f"🔄 Loading latest model: {latest_model}...\n")
        if load_model(latest_model):
            logging.info(f"✅ Successfully loaded model from step {total_steps}\n")
        else:
            logging.warning(f"❌ Failed to load model, starting fresh.\n")
            total_steps = 0
    else:
        logging.warning(f"🆕 No saved models found, starting fresh training.\n")
        total_steps = 0

    num_robots = config.MULTI_ROBOT_CONFIG['num_robots']
    logging.info(f"Multi-agent training system initialized:")
    logging.info(f"  - Number of agents: {num_robots}")
    logging.info(f"  - State dimension: {state_dim}")
    logging.info(f"  - Action dimension: {action_dim}")
    logging.info(f"  - Models directory: {config.MODELS_DIRECTORY}")
    logging.info(f"  - PPO policy created: {ppo_policy is not None}")
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
    This function should be called from your main loop to integrate multi-agent episode management.
    Call this function after each simulation step to check for episode resets.

    Usage in main loop:
    if config.USE_SIMULATION and config.USE_ISAAC_SIM:
        from training.training import integrate_with_main_loop
        integrate_with_main_loop()
    """
    global total_steps, agent_data

    import utilities.config as config

    # Safety check: ensure training system is initialized
    if not agent_data:
        return False

    # Check each agent for episode reset signals
    num_robots = config.MULTI_ROBOT_CONFIG['num_robots']
    any_reset_occurred = False
    
    for robot_id in range(num_robots):
        # Safety check: ensure agent data exists for this robot
        if robot_id not in agent_data:
            continue
            
        # Check if this agent needs a reset
        if agent_data[robot_id]['episode_needs_reset']:
            # Clear the signal
            agent_data[robot_id]['episode_needs_reset'] = False
            
            # Perform the actual reset in the main thread (safe for Isaac Sim)
            print(f"Main thread: Episode reset signal received for robot {robot_id}, performing reset...")
            reset_episode(agent_data, robot_id)
            any_reset_occurred = True

        # Also check directly if episode should be reset (backup check)
        if episodes.check_and_reset_episode_if_needed(agent_data, robot_id):
            # Episode was reset, log the event
            import logging
            logging.info(f"(training.py): Robot {robot_id} episode {agent_data[robot_id]['episode_counter']} reset in main loop integration\n")
            any_reset_occurred = True

    return any_reset_occurred


########## AGENT FUNCTIONS ##########

##### summon standard agent #####

def get_rl_action_standard(state, commands, intensity, frame):  # will eventually take camera frame as input
    pass


##### summon blind agent #####

def get_rl_action_blind(all_current_angles, commands, intensity):
    """
    Multi-agent PPO RL system that processes all robots in parallel.
    
    Args:
        all_current_angles: List of current joint angles for each robot
        commands: Commands for all robots (same for all)
        intensity: Intensity for all robots (same for all)
    
    Returns:
        Tuple of (all_target_angles, all_mid_angles, all_movement_rates) for all robots
    """
    global ppo_policy, total_steps, agent_data, shared_experience_buffer

    try:
        # Initialize training system if not done yet
        if ppo_policy is None:
            logging.debug("Initializing training system...")
            initialize_training()
            logging.debug("Training system initialized successfully")
        # CRITICAL: Always ensure agent_data is initialized, even if model was loaded
        elif not agent_data or len(agent_data) != config.MULTI_ROBOT_CONFIG['num_robots']:
            logging.debug("Agent data not properly initialized, initializing now...")
            agent_data = initialize_agent_data()
            logging.debug("Agent data initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize training system: {e}")
        return [], [], []

    num_robots = config.MULTI_ROBOT_CONFIG['num_robots']
    
    # Safety check: ensure agent_data is properly initialized
    if not agent_data or len(agent_data) != num_robots:
        logging.error(f"Agent data not properly initialized. Expected {num_robots} agents, got {len(agent_data) if agent_data else 0}")
        return [], [], []
    
    all_target_angles = []
    all_mid_angles = []
    all_movement_rates = []

    # Process each robot
    for robot_id in range(num_robots):
        try:
            # Safety check: ensure we have data for this robot
            if robot_id >= len(all_current_angles):
                logging.error(f"Robot {robot_id} current angles not available. Available robots: {len(all_current_angles)}")
                continue
                
            current_angles = all_current_angles[robot_id]
            agent = agent_data[robot_id]
            
            # Check if this agent's episode needs reset
            episode_needs_reset = False
            
            # Check episode completion
            if hasattr(rewards, f'EPISODE_STEP_{robot_id}'):
                episode_step = getattr(rewards, f'EPISODE_STEP_{robot_id}')
            else:
                episode_step = 0
            
            # Check for fall detection (immediate reset)
            if episode_step >= config.TRAINING_CONFIG['max_steps_per_episode']:
                episode_needs_reset = True
                if episode_step == config.TRAINING_CONFIG['max_steps_per_episode'] and episode_step > 0:
                    print(f"Robot {robot_id} FELL OVER! Episode {agent['episode_counter']} failed - immediate reset!")
                else:
                    print(f"Robot {robot_id} Episode {agent['episode_counter']} completed after {episode_step} steps! (signaling main thread)")

            # Store reset signal for main thread to check
            agent['episode_needs_reset'] = episode_needs_reset
            
            # Skip processing this robot if it needs a reset (to avoid joint angle errors)
            if episode_needs_reset:
                # Add empty data for this robot to maintain consistency
                all_target_angles.append({})
                all_mid_angles.append({})
                all_movement_rates.append({})
                continue

            # Log episode progress every 100 steps
            if episode_step % 100 == 0:
                print(f"Robot {robot_id} Episode {agent['episode_counter']}, Step {episode_step}, Total Steps: {total_steps}")
            
            # Track orientation every 50 steps to understand robot facing direction
            if episode_step % 50 == 0:
                track_orientation(robot_id)

            # Build state vector for this robot
            state = []

            # 1. Extract and normalize joint angles (12D)
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
                        normalized_angle = 0.0
                        
                    state.append(normalized_angle)

            # 2. Encode commands (6D one-hot for movement commands only)
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
                1.0 if 'arrowright' in command_list else 0.0
            ]
            
            state.extend(command_encoding)

            # 3. Normalize intensity (1D)
            intensity_normalized = (float(intensity) - 5.5) / 4.5
            state.append(intensity_normalized)
            
            # Convert to numpy array and validate state size
            state = np.array(state, dtype=np.float32)
            
            # Validate state size: 12 (joints) + 6 (commands) + 1 (intensity) = 19
            expected_state_size = 19
            if len(state) != expected_state_size:
                raise ValueError(f"Robot {robot_id} state size mismatch: expected {expected_state_size}, got {len(state)}")

            # Get action from PPO (stochastic during training)
            action = ppo_policy.select_action(state, deterministic=False)

            # Store experience for training (only if previous step had data)
            if agent['episode_states'] and agent['episode_actions']:
                # Calculate reward using the dedicated reward function
                reward = rewards.calculate_step_reward(current_angles, commands, intensity, robot_id)

                # Update episode reward for tracking
                agent['episode_reward'] += reward

                # Check if episode is done
                done = episode_step >= config.TRAINING_CONFIG['max_steps_per_episode']

                # Add to shared experience buffer
                shared_experience_buffer['states'].append(state)
                shared_experience_buffer['actions'].append(action)
                shared_experience_buffer['rewards'].append(reward)
                shared_experience_buffer['values'].append(None)  # Will be calculated later
                shared_experience_buffer['log_probs'].append(None)  # Will be calculated later
                shared_experience_buffer['dones'].append(done)
                shared_experience_buffer['agent_ids'].append(robot_id)

                # Log experience collection for debugging
                if done:
                    print(f"Robot {robot_id} 💾 Experience collected: State={len(state)}D, Action={len(action)}D, Reward={reward:.3f}, Done={done}")
                    print(f"   📊 Shared buffer size: {len(shared_experience_buffer['states'])}")

            # Update agent tracking variables
            agent['episode_states'].append(state)
            agent['episode_actions'].append(action)
            
            # Get value and log_prob for PPO training
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                _, log_prob, _, value = ppo_policy.actor_critic.get_action_and_value(state_tensor, torch.FloatTensor(action).unsqueeze(0))
                agent['episode_values'].append(value.item())
                agent['episode_log_probs'].append(log_prob.item())
            
            # Update episode step counter
            if not hasattr(rewards, f'EPISODE_STEP_{robot_id}'):
                setattr(rewards, f'EPISODE_STEP_{robot_id}', 0)
            setattr(rewards, f'EPISODE_STEP_{robot_id}', episode_step + 1)
            total_steps += 1

            # Convert 36D action vector to joint angles and velocities for this robot
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
                    target_action = action[action_idx + 12]
                    target_angle = min_angle + (target_action + 1.0) * 0.5 * (max_angle - min_angle)
                    target_angle = np.clip(target_angle, min_angle, max_angle)
                    target_angles[leg_id][joint_name] = float(target_angle)

                    # Convert velocity action (-1 to 1) to movement rate
                    velocity_action = action[action_idx + 24]
                    joint_speed = (velocity_action + 1.0) * 4.75
                    joint_speed = np.clip(joint_speed, 0.0, 9.5)
                    movement_rates[leg_id][joint_name] = float(joint_speed)

                    action_idx += 1

            all_target_angles.append(target_angles)
            all_mid_angles.append(mid_angles)
            all_movement_rates.append(movement_rates)

        except Exception as e:
            logging.error(f"Error processing robot {robot_id}: {e}")
            # Add empty data for this robot to maintain consistency
            all_target_angles.append({})
            all_mid_angles.append({})
            all_movement_rates.append({})
            continue

    # Train PPO when shared buffer has enough experiences
    if len(shared_experience_buffer['states']) >= config.TRAINING_CONFIG['batch_size']:
        train_shared_ppo()

    # Save model periodically based on total steps
    if total_steps % config.TRAINING_CONFIG['save_frequency'] == 0 and ppo_policy is not None:
        save_model(f"/home/matthewthomasbeck/Projects/Robot_Dog/model/ppo_steps_{total_steps}_multi_agent.pth")
        print(f"💾 Multi-agent model saved: steps_{total_steps}")

    return all_target_angles, all_mid_angles, all_movement_rates


########## SHARED PPO TRAINING ##########

def train_shared_ppo():
    """Train PPO using shared experience buffer from all agents"""
    global ppo_policy, shared_experience_buffer
    
    if len(shared_experience_buffer['states']) < config.TRAINING_CONFIG['batch_size']:
        return
    
    print(f"🧠 Training PPO with shared buffer, buffer size: {len(shared_experience_buffer['states'])}")
    
    # Convert to tensors
    states = torch.FloatTensor(shared_experience_buffer['states'])
    actions = torch.FloatTensor(shared_experience_buffer['actions'])
    rewards_tensor = torch.FloatTensor(shared_experience_buffer['rewards'])
    dones = torch.FloatTensor(shared_experience_buffer['dones'])
    
    # Calculate log probabilities for all actions in the buffer
    with torch.no_grad():
        _, log_probs, _, _ = ppo_policy.actor_critic.get_action_and_value(states, actions)
        log_probs = log_probs.squeeze(-1)  # Remove extra dimension
    
    # Get final value estimate for GAE calculation
    with torch.no_grad():
        final_value = ppo_policy.actor_critic.get_value(states[-1:]).item()
    
    # Train PPO on shared buffer data
    ppo_policy.train(states, actions, log_probs, rewards_tensor, dones, final_value)
    
    # Clear the shared buffer after training
    shared_experience_buffer = {
        'states': [],
        'actions': [],
        'rewards': [],
        'values': [],
        'log_probs': [],
        'dones': [],
        'agent_ids': []
    }
    
    print(f"✅ PPO training completed, shared buffer cleared")


########## MODEL FUNCTIONS ##########

##### save trained model #####

def save_model(filepath):
    """Save the current PPO model"""
    if ppo_policy:

        logging.info(f"💾 Saving multi-agent PPO model to: {filepath}")
        logging.info(f"   📊 Total steps: {total_steps}")
        logging.info(f"   📊 Number of agents: {len(agent_data)}")
        logging.info(f"   📊 Current average score: {episodes.AVERAGE_SCORE:.4f}\n")

        # Create the checkpoint data
        checkpoint = {
            'actor_critic_state_dict': ppo_policy.actor_critic.state_dict(),
            'optimizer_state_dict': ppo_policy.optimizer.state_dict(),
            'total_steps': total_steps,
            'agent_data': agent_data,
            'average_score': episodes.AVERAGE_SCORE,
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

        logging.info(f"Multi-agent model saved to: {filepath}")
        logging.info(f"   📊 Current average score: {episodes.AVERAGE_SCORE:.3f}\n")
    else:
        logging.error(f"❌ Cannot save model - PPO policy not initialized.\n")


##### load the model #####

def load_model(filepath):
    """Load a PPO model from file"""
    global ppo_policy, total_steps, agent_data

    if ppo_policy and os.path.exists(filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        ppo_policy.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        ppo_policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore training state
        total_steps = checkpoint.get('total_steps', 0)
        agent_data = checkpoint.get('agent_data', {})
        episodes.AVERAGE_SCORE = checkpoint.get('average_score', 0.0)

        # CRITICAL: If agent_data is empty or doesn't match expected number of robots, reinitialize it
        num_robots = config.MULTI_ROBOT_CONFIG['num_robots']
        if not agent_data or len(agent_data) != num_robots:
            logging.warning(f"Agent data from checkpoint is invalid (got {len(agent_data)} agents, expected {num_robots}), reinitializing...")
            agent_data = initialize_agent_data()

        logging.info(f"Multi-agent model loaded from: {filepath}")
        logging.info(f"  - Total steps: {total_steps}")
        logging.info(f"  - Number of agents: {len(agent_data)}")
        logging.info(f"  - Average score: {episodes.AVERAGE_SCORE:.4f}\n")
        return True
    return False

##### monitor learning progress #####

def monitor_learning_progress(robot_id=0):
    """
    Monitor the agent's learning progress and detect potential issues for a specific robot.
    This helps identify if the agent is stuck in a loop of falling behaviors.
    """
    global agent_data
    
    agent = agent_data[robot_id]
    if not agent['episode_states']:
        return
    
    # Calculate average reward over last few episodes
    recent_rewards = []
    if hasattr(monitor_learning_progress, f'episode_rewards_{robot_id}'):
        recent_rewards = getattr(monitor_learning_progress, f'episode_rewards_{robot_id}')[-5:]  # Last 5 episodes
    
    # Store current episode reward
    if not hasattr(monitor_learning_progress, f'episode_rewards_{robot_id}'):
        setattr(monitor_learning_progress, f'episode_rewards_{robot_id}', [])
    
    getattr(monitor_learning_progress, f'episode_rewards_{robot_id}').append(agent['episode_reward'])
    
    # Keep only last 20 episodes
    episode_rewards_list = getattr(monitor_learning_progress, f'episode_rewards_{robot_id}')
    if len(episode_rewards_list) > 20:
        episode_rewards_list.pop(0)
    
    # Analyze learning progress
    if len(recent_rewards) >= 3:
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        min_reward = min(recent_rewards)
        
        logging.info(f"📊 Robot {robot_id} Learning Progress Analysis:")
        logging.info(f"   🎯 Recent episodes: {len(recent_rewards)}")
        logging.info(f"   📈 Average reward: {avg_reward:.3f}")
        logging.info(f"   📉 Worst reward: {min_reward:.3f}")
        logging.info(f"   🧠 PPO policy: Ready for training.\n")
        
        # Detect if agent is improving
        if len(episode_rewards_list) >= 10:
            first_half = episode_rewards_list[:10]
            second_half = episode_rewards_list[-10:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg > first_avg:
                logging.info(f"✅ Robot {robot_id} is improving! First 10: {first_avg:.3f}, Last 10: {second_avg:.3f}\n")
            else:
                logging.info(f"❌ Robot {robot_id} not improving. First 10: {first_avg:.3f}, Last 10: {second_avg:.3f}\n")


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
    