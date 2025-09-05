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
time_steps = 0
robot_steps = 0
loaded_robot_steps = 0  # Track robot steps from loaded model

# Timing variables for performance analysis
training_start_time = None
first_training_time = None

# Episode tracking
current_episode_start_step = 0

# Movement history for each robot - stores last 5 target angle sets (12D each)
# Remove the local movement_history and use config.PREVIOUS_POSITIONS instead
# movement_history = {}  # robot_id -> deque of last 5 target angle arrays

##### multi-agent training data #####

# PPO rollout buffer - collect experiences then train immediately
rollout_buffer = {
    'states': [],
    'actions': [],
    'rewards': [],
    'values': [],
    'log_probs': [],
    'dones': []
}

# Per-agent data
agent_data = {}  # Will be initialized with num_robots agents

last_saved_step = 0





##################################################
############### ISAAC SIM TRAINING ###############
##################################################


########## TRAINING FUNCTIONS ##########

##### integrate with main loop #####

def integrate_with_main_loop():
    
    ##### initialize training #####

    global agent_data, time_steps, last_saved_step, rollout_buffer, ppo_policy, robot_steps, loaded_robot_steps, training_start_time, first_training_time, current_episode_start_step

    try:
        # Initialize training system if not done yet
        if ppo_policy is None:
            logging.debug("(training.py): Initializing training system...\n")
            training_start_time = time.time()
            initialize_training()
            current_episode_start_step = 0  # Initialize episode tracking
            logging.info("(training.py) Training system initialized successfully.\n")
        # CRITICAL: Always ensure agent_data is initialized, even if model was loaded
        elif not agent_data or len(agent_data) != config.MULTI_ROBOT_CONFIG['num_robots']:
            logging.debug("(training.py): Agent data not properly initialized, initializing now...\n")
            agent_data = initialize_agent_data()
            logging.info("(training.py): Agent data initialized successfully.\n")
    except Exception as e:
        logging.error(f"(training.py): Failed to initialize training system: {e}\n")
        return []

    time_steps += 1

    ##### reset episode every x steps #####

    # Check if current episode has reached max duration
    if time_steps >= current_episode_start_step + config.TRAINING_CONFIG['max_steps_per_episode']:
        episodes.reset_episode(agent_data)
        current_episode_start_step = time_steps  # Start new episode from current step
        logging.debug(f"(training.py): Episode reset at step {time_steps}, new episode starts at step {current_episode_start_step}\n")

    ##### reset episode on fall #####

    fallen_robots = _get_fallen_robots()
    if fallen_robots:
        print("(training.py): ROBOT FELL! Resetting episode...")
        episodes.reset_episode(agent_data)
        current_episode_start_step = time_steps  # Start new episode from current step
        logging.debug(f"(training.py): Episode reset due to fall at step {time_steps}, new episode starts at step {current_episode_start_step}\n")
        return True

    ##### train model every x time steps #####

    if time_steps % (config.TRAINING_CONFIG['max_steps_per_episode'] * 2) == 0 and time_steps != 0:
        first_training_time = time.time()
        if training_start_time is not None and time_steps == (config.TRAINING_CONFIG['max_steps_per_episode'] * 2):
            # Only log detailed timing for the first training
            total_time_to_first_training = first_training_time - training_start_time
            logging.info(f"(training.py) TIMING: Time from initialization to first training: {total_time_to_first_training:.3f} seconds")
            logging.info(f"(training.py) TIMING: Steps to first training: {time_steps}")
            logging.info(f"(training.py) TIMING: Robots: {config.MULTI_ROBOT_CONFIG['num_robots']}")
            logging.info(f"(training.py) TIMING: Time per step: {total_time_to_first_training/time_steps:.4f} seconds")
            logging.info(f"(training.py) TIMING: Time per robot per step: {total_time_to_first_training/(time_steps * config.MULTI_ROBOT_CONFIG['num_robots']):.6f} seconds\n")
        train_shared_ppo()
        logging.debug(f"(training.py): Trained model at step {time_steps}\n")

    ##### save model every x robot steps #####

    if robot_steps % config.TRAINING_CONFIG['save_frequency'] == 0 and robot_steps != 0:
        logging.info(
            f"(training.py): Saving model at time_steps {time_steps} and robot_steps {robot_steps}.\n")

        _save_training_model()
        last_saved_step = time_steps
        logging.info(f"(training.py): Saved model at robot_steps {robot_steps}\n")

    ##### safety check #####

    if not agent_data:
        return False

    return False  # return false if training is not done

##### fallen robots helper #####

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

##### save model #####

def _save_training_model():
    """Helper function to save the training model"""
    global robot_steps, loaded_robot_steps
    
    # Calculate average score across all robots
    total_avg_score = 0.0
    active_robots = 0

    for robot_id in range(config.MULTI_ROBOT_CONFIG['num_robots']):
        if robot_id in agent_data:
            robot_avg = agent_data[robot_id].get('average_reward', 0.0)
            total_avg_score += robot_avg
            active_robots += 1

    overall_avg_score = total_avg_score / active_robots if active_robots > 0 else 0.0

    # Create filename with robot steps (including loaded steps) and average score
    total_robot_steps = loaded_robot_steps + robot_steps
    filename = f"{total_robot_steps}_{overall_avg_score:.3f}.pth"
    filepath = f"/home/matthewthomasbeck/Projects/Robot_Dog/model/{filename}"

    save_model(filepath, ppo_policy, agent_data, total_robot_steps)
    logging.info(f"(training.py): Continuous learning model saved: {filename} (total robot steps: {total_robot_steps})\n")

##### initialize training #####

def initialize_training():
    """Initialize the complete multi-agent training system"""
    global ppo_policy, time_steps, agent_data, loaded_robot_steps

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
        logging.warning(f"(training.py): PREVIOUS_POSITIONS not properly initialized, initializing now...\n")
        config.PREVIOUS_POSITIONS = []
        for robot_id in range(num_robots):
            robot_history = deque(maxlen=5)
            for _ in range(5):
                robot_history.append(np.zeros(12, dtype=np.float32))
            config.PREVIOUS_POSITIONS.append(robot_history)
        logging.debug(f"(training.py): PREVIOUS_POSITIONS initialized for {num_robots} robots with zeros.\n")

    # Try to load the latest saved model to continue training
    latest_model = find_latest_model()
    if latest_model:
        logging.debug(f"(training.py): Loading latest model: {latest_model}...\n")
        success, loaded_steps, loaded_agent_data = load_model(latest_model, ppo_policy)
        if success:
            # loaded_steps now represents robot steps from the saved model
            loaded_robot_steps = loaded_steps
            time_steps = 0  # Reset time steps for new training session
            agent_data = loaded_agent_data
            logging.info(f"(training.py): Successfully loaded model with {loaded_robot_steps} robot steps.\n")
        else:
            logging.warning(f"(training.py): Failed to load model, starting fresh...\n")
            loaded_robot_steps = 0
            time_steps = 0
    else:
        logging.warning(f"(training.py): No saved models found, starting fresh training...\n")
        loaded_robot_steps = 0
        time_steps = 0

    logging.info(f"Multi-agent training system initialized:")
    logging.info(f"  - Number of agents: {num_robots}")
    logging.info(f"  - State dimension: {state_dim}")
    logging.info(f"  - Action dimension: {action_dim}")
    logging.info(f"  - Models directory: {config.MODELS_DIRECTORY}")
    logging.info(f"  - PPO policy created: {ppo_policy is not None}")
    logging.info(f"  - Starting from time step: {time_steps}")
    logging.info(f"  - Loaded robot steps: {loaded_robot_steps}")
    logging.info(f"  - PREVIOUS_POSITIONS verified for {num_robots} robots\n")

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
    global ppo_policy, time_steps, agent_data, rollout_buffer, robot_steps

    num_robots = config.MULTI_ROBOT_CONFIG['num_robots']
    
    # Safety check: ensure agent_data is properly initialized
    if not agent_data or len(agent_data) != num_robots:
        logging.error(f"(training.py): Agent data not properly initialized. Expected {num_robots} agents, got {len(agent_data) if agent_data else 0}\n")
        return []
    
    # PREVIOUS_POSITIONS should already be initialized in control_logic.py
    # Just verify it exists and has the right number of robots
    if not config.PREVIOUS_POSITIONS or len(config.PREVIOUS_POSITIONS) != num_robots:
        logging.error(f"(training.py): PREVIOUS_POSITIONS not properly initialized. Expected {num_robots} robots, got {len(config.PREVIOUS_POSITIONS) if config.PREVIOUS_POSITIONS else 0}\n")
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
            if time_steps % 100 == 0:
                steps_since_reset = time_steps - agent.get('last_reset_step', 0)
                #print(f"Robot {robot_id} - Step {total_steps}, Active for {steps_since_reset} steps, Avg Reward: {agent.get('average_reward', 0.0):.3f}")
            
            # Track orientation every 50 steps
            if time_steps % 50 == 0:
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
            if robot_id == 0:
                print(f"(training.py): Calculating reward for robot {robot_id} with commands {commands} and intensity {intensity}...")
            reward = rewards.calculate_step_reward(commands, intensity, robot_id)
            if robot_id == 0:
                print(f"(training.py): Reward for robot {robot_id}: {reward}")

            # Update robot's reward tracking
            agent['total_reward'] += reward
            agent['recent_rewards'].append(reward)
            if len(agent['recent_rewards']) > 100:  # Keep only last 100 rewards
                agent['recent_rewards'].pop(0)
            agent['average_reward'] = sum(agent['recent_rewards']) / len(agent['recent_rewards'])

            # Add to rollout buffer (PPO on-policy learning)
            rollout_buffer['states'].append(state)
            rollout_buffer['actions'].append(action)
            rollout_buffer['rewards'].append(reward)
            rollout_buffer['values'].append(None)  # Will be calculated later
            rollout_buffer['log_probs'].append(None)  # Will be calculated later
            rollout_buffer['dones'].append(False)  # No episodes, so never "done"
            
            # Get value and log_prob for PPO training
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                _, log_prob, _, value = ppo_policy.actor_critic.get_action_and_value(state_tensor, torch.FloatTensor(action).unsqueeze(0))
                rollout_buffer['values'][-1] = value.item()
                rollout_buffer['log_probs'][-1] = log_prob.item()
            
            # Manage buffer size to prevent memory overflow
            if len(rollout_buffer['states']) > config.TRAINING_CONFIG['batch_size']:
                # Remove oldest experiences efficiently (keep only the last batch_size items)
                batch_size = config.TRAINING_CONFIG['batch_size']
                for key in rollout_buffer:
                    rollout_buffer[key] = rollout_buffer[key][-batch_size:]

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

            robot_steps += 1
            print(f"{robot_steps}")

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
    global ppo_policy, rollout_buffer
    
    if len(rollout_buffer['states']) < config.TRAINING_CONFIG['batch_size']:
        return
    
    training_start = time.time()
    logging.debug(f"(training.py): Training PPO with batch size: {len(rollout_buffer['states'])} experiences...\n")
    
    try: # attempt to train PPO

        # Convert to tensors efficiently - convert lists to numpy arrays first
        # Ensure all buffers have exactly batch_size items
        batch_size = config.TRAINING_CONFIG['batch_size']
        if len(rollout_buffer['states']) != batch_size:
            # Trim to exact batch size
            for key in rollout_buffer:
                rollout_buffer[key] = rollout_buffer[key][-batch_size:]

        states = torch.FloatTensor(np.array(rollout_buffer['states']))
        actions = torch.FloatTensor(np.array(rollout_buffer['actions']))
        rewards_tensor = torch.FloatTensor(np.array(rollout_buffer['rewards']))
        dones = torch.FloatTensor(np.array(rollout_buffer['dones']))
        
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
        rollout_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }

    except Exception as e:
        logging.error(f"(training.py): Failed to train PPO: {e}\n")
        return