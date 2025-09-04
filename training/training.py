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
import torch
import time
from collections import deque

##### import necessary functions #####

from movement.isaac_joints import neutral_position_isaac
from training.agents import *
import training.rewards as rewards
from training.orientation import track_orientation
import training.episodes as episodes
from training.model_manager import save_model, load_model, find_latest_model


########## CREATE DEPENDENCIES ##########

##### global PPO instance and episode data #####

ppo_policy = None
total_steps = 0

# Movement history for each robot - stores last 5 target angle sets (12D each)
# Remove the local movement_history and use config.PREVIOUS_POSITIONS instead
# movement_history = {}  # robot_id -> deque of last 5 target angle arrays

##### multi-agent training data #####

# MASSIVE shared experience buffer for all agents (100k experiences)
shared_experience_buffer = {
    'states': [],
    'actions': [],
    'rewards': [],
    'values': [],
    'log_probs': [],
    'dones': [],
    'agent_ids': []  # Track which agent generated each experience
}

# Buffer size limit to prevent memory overflow
MAX_BUFFER_SIZE = config.TRAINING_CONFIG.get('experience_buffer_size', 100000)

# Per-agent data
agent_data = {}  # Will be initialized with num_robots agents





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
    state_dim = 67  # (12 joints * 5 history) + 6 commands + 1 intensity = 60 + 6 + 1 = 67
    action_dim = 12  # 12 target angles only (no mid angles or velocities)
    max_action = config.TRAINING_CONFIG['max_action']

    ppo_policy = PPO(state_dim, action_dim, max_action)

    # Initialize per-agent data
    agent_data = initialize_agent_data()

    # PREVIOUS_POSITIONS should already be initialized in control_logic.py
    # Just verify it exists and has the right number of robots
    num_robots = config.MULTI_ROBOT_CONFIG['num_robots']
    if not config.PREVIOUS_POSITIONS or len(config.PREVIOUS_POSITIONS) != num_robots:
        logging.warning(f"PREVIOUS_POSITIONS not properly initialized, initializing now...")
        config.PREVIOUS_POSITIONS = []
        for robot_id in range(num_robots):
            robot_history = deque(maxlen=5)
            for _ in range(5):
                robot_history.append(np.zeros(12, dtype=np.float32))
            config.PREVIOUS_POSITIONS.append(robot_history)
        logging.debug(f"PREVIOUS_POSITIONS initialized for {num_robots} robots with zeros")

    # Try to load the latest saved model to continue training
    latest_model = find_latest_model()
    if latest_model:
        logging.debug(f"🔄 Loading latest model: {latest_model}...\n")
        success, loaded_steps, loaded_agent_data = load_model(latest_model, ppo_policy)
        if success:
            total_steps = loaded_steps
            agent_data = loaded_agent_data
            logging.info(f"✅ Successfully loaded model from step {total_steps}\n")
        else:
            logging.warning(f"❌ Failed to load model, starting fresh.\n")
            total_steps = 0
    else:
        logging.warning(f"🆕 No saved models found, starting fresh training.\n")
        total_steps = 0

    logging.info(f"Multi-agent training system initialized:")
    logging.info(f"  - Number of agents: {num_robots}")
    logging.info(f"  - State dimension: {state_dim}")
    logging.info(f"  - Action dimension: {action_dim}")
    logging.info(f"  - Models directory: {config.MODELS_DIRECTORY}")
    logging.info(f"  - PPO policy created: {ppo_policy is not None}")
    logging.info(f"  - Starting from step: {total_steps}")
    logging.info(f"  - PREVIOUS_POSITIONS verified for {num_robots} robots\n")

##### integrate with main loop #####

def integrate_with_main_loop():
    """
    Continuous multi-agent training system - no episodes, just shared step counter.
    Individual robots can be reset independently when they fall.
    
    Usage in main loop:
    if config.USE_SIMULATION and config.USE_ISAAC_SIM:
        from training.training import integrate_with_main_loop
        integrate_with_main_loop()
    """
    global agent_data, total_steps
    
    # Safety check: ensure training system is initialized
    if not agent_data:
        return False

    ##### TRAINING AND MODEL MANAGEMENT #####
    
    # Train PPO when shared buffer has enough experiences
    if len(shared_experience_buffer['states']) >= config.TRAINING_CONFIG['batch_size']:
        train_shared_ppo()

    # Save model every save_frequency steps
    if total_steps % config.TRAINING_CONFIG['save_frequency'] == 0 and ppo_policy is not None and total_steps > 0:
        _save_training_model()

    ##### RESET CHECKS #####
    
    # Check for time-based reset (every max_steps_per_episode steps)
    if total_steps > 0 and total_steps % config.TRAINING_CONFIG.get('max_steps_per_episode', 10000) == 0:
        print(f"⏰ TIME-BASED RESET at step {total_steps}")
        episodes.reset_episode(agent_data)
        print(f"🔄 Time-based reset complete - all robots active")
        return True

    # Check for fall-based reset (any robot falls)
    fallen_robots = _get_fallen_robots()
    if fallen_robots:
        episodes.reset_episode(agent_data)
        return True
    
    return False


def _get_fallen_robots():
    """Helper function to identify fallen robots"""
    fallen_robots = []
    num_robots = config.MULTI_ROBOT_CONFIG['num_robots']

    for robot_id in range(num_robots):
        if robot_id not in agent_data:
            continue

        agent = agent_data[robot_id]
        if not agent.get('is_active', True):
            fallen_robots.append(robot_id)

    return fallen_robots


def _save_training_model():
    """Helper function to save the training model"""
    # Calculate average score across all robots
    total_avg_score = 0.0
    active_robots = 0
    
    for robot_id in range(config.MULTI_ROBOT_CONFIG['num_robots']):
        if robot_id in agent_data:
            robot_avg = agent_data[robot_id].get('average_reward', 0.0)
            total_avg_score += robot_avg
            active_robots += 1
    
    overall_avg_score = total_avg_score / active_robots if active_robots > 0 else 0.0
    
    # Create filename with average score
    filename = f"{total_steps}_{overall_avg_score:.3f}.pth"
    filepath = f"/home/matthewthomasbeck/Projects/Robot_Dog/model/{filename}"
    
    save_model(filepath, ppo_policy, agent_data, total_steps)
    print(f"💾 Continuous learning model saved: {filename}")

##### summon standard agent #####

def get_rl_action_standard(state, commands, intensity, frame):  # will eventually take camera frame as input
    pass

##### summon blind agent #####

def get_rl_action_blind(commands, intensity):
    """
    Multi-agent PPO RL system that processes all robots in parallel.
    Uses movement history instead of current joint angles for state representation.
    
    Args:
        commands: Commands for all robots (same for all)
        intensity: Intensity for all robots (same for all)
    
    Returns:
        List of target_angles for all robots (12D output - only target angles)
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
        return []

    num_robots = config.MULTI_ROBOT_CONFIG['num_robots']
    
    # Safety check: ensure agent_data is properly initialized
    if not agent_data or len(agent_data) != num_robots:
        logging.error(f"Agent data not properly initialized. Expected {num_robots} agents, got {len(agent_data) if agent_data else 0}")
        return []
    
    # PREVIOUS_POSITIONS should already be initialized in control_logic.py
    # Just verify it exists and has the right number of robots
    if not config.PREVIOUS_POSITIONS or len(config.PREVIOUS_POSITIONS) != num_robots:
        logging.error(f"PREVIOUS_POSITIONS not properly initialized. Expected {num_robots} robots, got {len(config.PREVIOUS_POSITIONS) if config.PREVIOUS_POSITIONS else 0}")
        return []
    
    all_target_angles = []
    all_movement_rates = []

    # Process each robot
    for robot_id in range(num_robots):
        try:
            agent = agent_data[robot_id]
            
            # Check if robot is active
            if not agent.get('is_active', True):
                # Robot is fallen/inactive, add empty data to maintain consistency
                all_target_angles.append({})
                all_movement_rates.append({})
                continue

            # Log progress every 100 steps
            if total_steps % 100 == 0:
                steps_since_reset = total_steps - agent.get('last_reset_step', 0)
                #print(f"Robot {robot_id} - Step {total_steps}, Active for {steps_since_reset} steps, Avg Reward: {agent.get('average_reward', 0.0):.3f}")
            
            # Track orientation every 50 steps
            if total_steps % 50 == 0:
                track_orientation(robot_id)

            # Build state vector for this robot
            state = []

            # 1. Add last 5 target angle sets (60D total: 12 * 5) from PREVIOUS_POSITIONS
            for historical_angles in config.PREVIOUS_POSITIONS[robot_id]:
                state.extend(historical_angles)

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
            
            # Validate state size: (12 * 5) + 6 (commands) + 1 (intensity) = 67
            expected_state_size = 67
            if len(state) != expected_state_size:
                raise ValueError(f"Robot {robot_id} state size mismatch: expected {expected_state_size}, got {len(state)}")

            # Get action from PPO (stochastic during training)
            action = ppo_policy.select_action(state, deterministic=False)

            # Calculate reward using the dedicated reward function
            reward = rewards.calculate_step_reward(commands, intensity, robot_id)

            # Update robot's reward tracking
            agent['total_reward'] += reward
            agent['recent_rewards'].append(reward)
            if len(agent['recent_rewards']) > 100:  # Keep only last 100 rewards
                agent['recent_rewards'].pop(0)
            agent['average_reward'] = sum(agent['recent_rewards']) / len(agent['recent_rewards'])

            # Add to shared experience buffer (continuous learning)
            shared_experience_buffer['states'].append(state)
            shared_experience_buffer['actions'].append(action)
            shared_experience_buffer['rewards'].append(reward)
            shared_experience_buffer['values'].append(None)  # Will be calculated later
            shared_experience_buffer['log_probs'].append(None)  # Will be calculated later
            shared_experience_buffer['dones'].append(False)  # No episodes, so never "done"
            shared_experience_buffer['agent_ids'].append(robot_id)
            
            # Get value and log_prob for PPO training
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                _, log_prob, _, value = ppo_policy.actor_critic.get_action_and_value(state_tensor, torch.FloatTensor(action).unsqueeze(0))
                shared_experience_buffer['values'][-1] = value.item()
                shared_experience_buffer['log_probs'][-1] = log_prob.item()
            
            # Manage buffer size to prevent memory overflow
            if len(shared_experience_buffer['states']) > MAX_BUFFER_SIZE:
                # Remove oldest experiences (FIFO)
                for key in shared_experience_buffer:
                    shared_experience_buffer[key].pop(0)
            
            # Increment shared step counter
            total_steps += 1

            # Convert 12D action vector to target joint angles for this robot
            target_angles = {}
            movement_rates = {}

            action_idx = 0
            for leg_id in ['FL', 'FR', 'BL', 'BR']:
                target_angles[leg_id] = {}
                movement_rates[leg_id] = {}

                for joint_name in ['hip', 'upper', 'lower']:
                    # Get joint limits from config
                    servo_data = config.SERVO_CONFIG[leg_id][joint_name]
                    min_angle = servo_data['FULL_BACK_ANGLE']
                    max_angle = servo_data['FULL_FRONT_ANGLE']

                    # Ensure correct order
                    if min_angle > max_angle:
                        min_angle, max_angle = max_angle, min_angle

                    # Convert target action (-1 to 1) to joint angle
                    target_action = action[action_idx]
                    target_angle = min_angle + (target_action + 1.0) * 0.5 * (max_angle - min_angle)
                    target_angle = np.clip(target_angle, min_angle, max_angle)
                    target_angles[leg_id][joint_name] = float(target_angle)

                    movement_rates[leg_id][joint_name] = 1 # legacy support for velocity action dims

                    action_idx += 1

            # Update movement history with current target angles (normalized to [-1, 1])
            current_target_array = np.zeros(12, dtype=np.float32)
            action_idx = 0
            for leg_id in ['FL', 'FR', 'BL', 'BR']:
                for joint_name in ['hip', 'upper', 'lower']:
                    # Store the raw action value (already normalized to [-1, 1])
                    current_target_array[action_idx] = action[action_idx]
                    action_idx += 1
            
            # Add to movement history (deque automatically maintains maxlen=5)
            config.PREVIOUS_POSITIONS[robot_id].append(current_target_array)

            all_target_angles.append(target_angles)
            all_movement_rates.append(movement_rates)

        except Exception as e:
            logging.error(f"Error processing robot {robot_id}: {e}")
            # Add empty data for this robot to maintain consistency
            all_target_angles.append({})
            all_movement_rates.append({})
            continue

    return all_target_angles, all_movement_rates


########## SHARED PPO TRAINING ##########

def train_shared_ppo():
    """Train PPO using shared experience buffer from all agents - SCALED UP VERSION"""
    global ppo_policy, shared_experience_buffer
    
    if len(shared_experience_buffer['states']) < config.TRAINING_CONFIG['batch_size']:
        return
    
    print(f"🧠 Training PPO with MASSIVE batch size: {len(shared_experience_buffer['states'])} experiences")
    
    # Convert to tensors (keep it simple, just bigger)
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
    
    # Train PPO on shared buffer data (same method, just bigger batches)
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
    
    print(f"✅ MASSIVE batch PPO training completed, shared buffer cleared")
