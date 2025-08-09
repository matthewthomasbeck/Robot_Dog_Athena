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

import numpy

##### import config #####

import utilities.config as config

##### import isaac sim libraries #####

from pxr import Gf





###################################################
############### ISAAC SIM FUNCTIONS ###############
###################################################

########## ISAAC SIM JOINT MAPPING ##########

def build_isaac_joint_index_map(dof_names):

    alias_to_actual = {}

    for name in dof_names:
        parts = name.split("_")
        if len(parts) >= 2:
            leg_id = parts[0]
            if parts[1] in {"hip", "upper", "lower"}:
                joint_type = parts[1]
            elif "femur" in name:
                joint_type = "upper"
            elif "shin" in name:
                joint_type = "lower"
            else:
                continue
            alias = f"{leg_id}_{joint_type}"
            if alias not in alias_to_actual:
                alias_to_actual[alias] = name  # first valid mapping

    joint_index_map = {}
    for alias, actual_name in alias_to_actual.items():
        if actual_name in dof_names:
            joint_index_map[alias] = dof_names.index(actual_name)

    return joint_index_map


########## REWARD FUNCTION ##########

def compute_reward(robot_prim_path, previous_pose, current_pose, command, intensity):

    # position delta
    prev_pos = numpy.array(previous_pose[0])  # (x, y, z)
    curr_pos = numpy.array(current_pose[0])
    delta_pos = curr_pos - prev_pos  # (dx, dy, dz)

    # orientation delta
    prev_rot = Gf.Quatf(*previous_pose[1])  # (w, x, y, z)
    curr_rot = Gf.Quatf(*current_pose[1])
    delta_rot = curr_rot * prev_rot.GetInverse()
    delta_yaw = delta_rot.GetAxisAngle()[0] * delta_rot.GetAxisAngle()[1][2]  # yaw around Z

    reward = 0.0
    command_keys = command.split('+') if isinstance(command, str) else []

    # Translational movements
    if 'w' in command_keys:
        reward += delta_pos[0] * intensity  # +X is forward
    if 's' in command_keys:
        reward -= delta_pos[0] * intensity
    if 'a' in command_keys:
        reward += delta_pos[1] * intensity  # +Y is left
    if 'd' in command_keys:
        reward -= delta_pos[1] * intensity

    # Rotational movements
    if 'arrowleft' in command_keys:
        reward += delta_yaw * intensity
    if 'arrowright' in command_keys:
        reward -= delta_yaw * intensity

    # Tilting (Up/Down): reward changes in pitch via z-axis body angle or delta_z
    delta_z = delta_pos[2]

    if 'arrowup' in command_keys:
        reward += delta_z * intensity  # raise front / tilt up
    if 'arrowdown' in command_keys:
        reward -= delta_z * intensity  # lower front / tilt down

    # Penalty for unintended drift if no command is active
    if not command_keys:
        noise_penalty = numpy.linalg.norm(delta_pos[:2]) * 0.05
        reward -= noise_penalty

    return reward


########## SIMULATION VARIABLES ##########

def set_simulation_variables(robot_id, joint_map):
    """
    Set global simulation variables for PyBullet.
    Args:
        robot_id: PyBullet robot body ID
        joint_map: Dictionary mapping joint names to indices
    """
    global ROBOT_ID, JOINT_MAP
    ROBOT_ID = robot_id
    JOINT_MAP = joint_map


########## RL AGENT ACTION FUNCTIONS ##########

def get_rl_action_standard(state, commands, intensity, frame):
    """
    Placeholder for RL agent's policy WITH image input (standard gait adjustment).
    Args:
        state: The current state of the robot/simulation (to be defined).
        commands: The movement commands.
        intensity: The movement intensity.
        frame: The current image frame (for vision-based agent).
    Returns:
        target_angles: dict of target joint angles for each leg (similar to SERVO_CONFIG structure).
        mid_angles: dict of mid joint angles for each leg (similar to SERVO_CONFIG structure).
        movement_rates: dict of movement rate parameters for each leg.
    TODO: Replace this with a call to your RL agent's policy/model (with image input).
    """
    # For now, just return the current angles as both mid and target
    target_angles = {}
    mid_angles = {}
    movement_rates = {}
    
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        target_angles[leg_id] = {}
        mid_angles[leg_id] = {}
        movement_rates[leg_id] = {'speed': 1000, 'acceleration': 255}
        
        for joint_name in ['hip', 'upper', 'lower']:
            current_angle = config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE']
            target_angles[leg_id][joint_name] = current_angle
            mid_angles[leg_id][joint_name] = current_angle
    
    return target_angles, mid_angles, movement_rates


def get_rl_action_blind(state, commands, intensity):
    """
    Placeholder for RL agent's policy WITHOUT image input (imageless gait adjustment).
    Args:
        state: The current state of the robot/simulation (to be defined).
        commands: The movement commands.
        intensity: The movement intensity.
    Returns:
        target_angles: dict of target joint angles for each leg (similar to SERVO_CONFIG structure).
        mid_angles: dict of mid joint angles for each leg (similar to SERVO_CONFIG structure).
        movement_rates: dict of movement rate parameters for each leg.
    TODO: Replace this with a call to your RL agent's policy/model (no image input).
    """
    # For now, just return the current angles as both mid and target (no movement)
    target_angles = {}
    mid_angles = {}
    movement_rates = {}
    
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        target_angles[leg_id] = {}
        mid_angles[leg_id] = {}
        movement_rates[leg_id] = {'speed': 1000, 'acceleration': 255}
        
        for joint_name in ['hip', 'upper', 'lower']:
            current_angle = config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE']
            target_angles[leg_id][joint_name] = current_angle
            mid_angles[leg_id][joint_name] = current_angle
    
    return target_angles, mid_angles, movement_rates


########## RL COMMAND GENERATION ##########

def generate_rl_command():
    """
    Generate a command string for the RL agent to execute.
    Returns command strings compatible with _execute_keyboard_commands format.
    
    Returns:
        str: Command string like 'w', 'w+d', 'w+d+arrowleft+arrowup', or None for neutral
    
    Examples of possible outputs:
        - 'w' (forward)
        - 's+a' (backward + left)
        - 'w+d+arrowleft' (forward + right + rotate left)
        - 'w+d+arrowleft+arrowup' (forward + right + rotate left + tilt up)
        - None (neutral/no action)
    """
    import random
    
    # Define possible actions with their probabilities
    movement_actions = ['w', 's', 'a', 'd']  # forward, backward, left, right
    rotation_actions = ['arrowleft', 'arrowright']  # rotate left, rotate right
    tilt_actions = ['arrowup', 'arrowdown']  # tilt up, tilt down
    
    # Probability of including each type of action
    movement_prob = 0.7  # 70% chance of movement
    rotation_prob = 0.3  # 30% chance of rotation (can combine with movement)
    tilt_prob = 0.2     # 20% chance of tilt (can combine with movement/rotation)
    neutral_prob = 0.1  # 10% chance of neutral (no action)
    
    # Check for neutral action first
    if random.random() < neutral_prob:
        return None  # Neutral command
    
    command_parts = []
    
    # Add movement action
    if random.random() < movement_prob:
        # Decide between single movement or diagonal
        if random.random() < 0.3:  # 30% chance of diagonal movement
            # Choose two non-contradictory movements
            forward_back = random.choice(['w', 's'])
            left_right = random.choice(['a', 'd'])
            command_parts.extend([forward_back, left_right])
        else:
            # Single movement
            command_parts.append(random.choice(movement_actions))
    
    # Add rotation action (can combine with movement)
    if random.random() < rotation_prob:
        command_parts.append(random.choice(rotation_actions))
    
    # Add tilt action (can combine with movement and rotation)
    if random.random() < tilt_prob:
        command_parts.append(random.choice(tilt_actions))
    
    # If no actions were selected, return neutral
    if not command_parts:
        return None
    
    # Join parts with '+' to create command string
    return '+'.join(command_parts)


def get_rl_command_with_intensity():
    """
    Generate both a command and intensity for the RL agent.
    
    Returns:
        tuple: (command_string, intensity_value)
            command_string: str or None (compatible with _execute_keyboard_commands)
            intensity_value: int between 1 and 10 (always 10 for neutral commands)
    """
    import random
    
    command = generate_rl_command()
    
    # Special case: neutral commands always have intensity 10 (fast return to neutral)
    if command is None:
        return command, 10
    
    # Generate intensity between 1 and 10 (integers only) with varied randomness for movement commands
    rand_value = random.random()
    
    if rand_value < 0.15:  # 15% chance of low intensity (1-3)
        intensity = random.randint(1, 3)
    elif rand_value < 0.70:  # 55% chance of moderate intensity (4-7)
        intensity = random.randint(4, 7)
    else:  # 30% chance of high intensity (8-10)
        intensity = random.randint(8, 10)
    
    return command, intensity


def inject_rl_command_into_queue(rl_command_queue, command, intensity):
    """
    Inject an RL-generated command into the RL command queue.
    This follows the same pattern as web commands being injected into COMMAND_QUEUE.
    
    Args:
        rl_command_queue: Queue object for RL commands
        command: str or None - Command string like 'w+d+arrowleft' or None for neutral
        intensity: int - Movement intensity (1 to 10)
    """
    if rl_command_queue is not None:
        # Package command with intensity (similar to web command format)
        command_data = {
            'command': command,
            'intensity': intensity
        }
        rl_command_queue.put(command_data)




def run_rl_episode_step(rl_command_queue):
    """
    Execute one step of RL training by generating a command and putting it in the queue.
    This follows the same pattern as web commands.
    
    Args:
        rl_command_queue: Queue object for RL commands
    
    Returns:
        dict: Episode step information including:
            - command: The generated command
            - intensity: The command intensity
    """
    # Generate RL command and intensity
    command, intensity = get_rl_command_with_intensity()
    
    # Put command into queue (same as web command flow)
    inject_rl_command_into_queue(rl_command_queue, command, intensity)
    
    # Return step information for RL training
    return {
        'command': command,
        'intensity': intensity
    }


########## ISAAC SIM QUEUE PROCESSING ##########

def process_isaac_movement_queue():
    """
    Process queued movements for Isaac Sim in the main thread to avoid PhysX violations.
    This function should be called from the main loop after each simulation step.
    """
    if not config.USE_SIMULATION or not config.USE_ISAAC_SIM:
        return
    
    # Import here to avoid circular imports
    from movement.fundamental_movement import ISAAC_MOVEMENT_QUEUE, ISAAC_CALIBRATION_COMPLETE, _apply_single_joint_position_isaac
    import queue
    
    # Process any new movement data
    while not ISAAC_MOVEMENT_QUEUE.empty():
        try:
            movement_data = ISAAC_MOVEMENT_QUEUE.get_nowait()
            
            # Check if this is calibration data
            if isinstance(movement_data, dict) and movement_data.get('type') == 'calibration':
                # Apply the joint position directly
                joint_name = movement_data['joint_name']
                angle_rad = movement_data['angle_rad']
                velocity = movement_data.get('velocity', 0.5)  # Default to 0.5 if not specified
                _apply_single_joint_position_isaac(joint_name, angle_rad, velocity)
                
                # Signal that this calibration movement is complete
                ISAAC_CALIBRATION_COMPLETE.set()
                
        except queue.Empty:
            break
