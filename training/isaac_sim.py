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
    import random
    
    target_angles = {}
    mid_angles = {}
    movement_rates = {}
    
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        target_angles[leg_id] = {}
        mid_angles[leg_id] = {}
        movement_rates[leg_id] = {'speed': 1.0, 'acceleration': 0.5}  # 1 rad/s, 0.5 rad/s²
        
        for joint_name in ['hip', 'upper', 'lower']:
            servo_data = config.SERVO_CONFIG[leg_id][joint_name]
            
            # Get the valid range for this joint
            full_back_angle = servo_data['FULL_BACK_ANGLE']  # Already in radians
            full_front_angle = servo_data['FULL_FRONT_ANGLE']  # Already in radians
            
            # Ensure we have the correct order (back < front)
            min_angle = min(full_back_angle, full_front_angle)
            max_angle = max(full_back_angle, full_front_angle)
            
            # Generate random angles within the valid range
            target_angle = random.uniform(min_angle, max_angle)
            mid_angle = random.uniform(min_angle, max_angle)
            
            target_angles[leg_id][joint_name] = target_angle
            mid_angles[leg_id][joint_name] = mid_angle
    
    return target_angles, mid_angles, movement_rates


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


########## RL STATE TRACKING ##########

def get_robot_state():
    """
    Get the current state of the robot for RL training.
    Returns:
        dict: Robot state including position, orientation, velocity, joint angles, etc.
    """
    try:
        # Get robot position and orientation
        robot_position, robot_orientation = config.ISAAC_ROBOT.get_world_pose()
        
        # Get robot velocities
        linear_velocity = config.ISAAC_ROBOT.get_linear_velocity()
        angular_velocity = config.ISAAC_ROBOT.get_angular_velocity()
        
        # Get joint positions and velocities
        joint_positions = config.ISAAC_ROBOT.get_joint_positions()
        joint_velocities = config.ISAAC_ROBOT.get_joint_velocities()
        
        # Calculate derived state information
        height = robot_position[2]  # Z-axis height
        forward_velocity = linear_velocity[0]  # X-axis velocity (forward/backward)
        lateral_velocity = linear_velocity[1]  # Y-axis velocity (left/right)
        
        # Extract roll, pitch, yaw from quaternion
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat([robot_orientation[1], robot_orientation[2], robot_orientation[3], robot_orientation[0]])  # Isaac uses (w,x,y,z), scipy uses (x,y,z,w)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        
        state = {
            'position': robot_position,
            'orientation': robot_orientation,
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'height': height,
            'forward_velocity': forward_velocity,
            'lateral_velocity': lateral_velocity,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'is_fallen': is_robot_fallen(),
            'ground_contact': get_ground_contacts()
        }
        
        return state
        
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Failed to get robot state: {e}\n")
        return None


def is_robot_fallen(height_threshold=0.05, angle_threshold=0.5):
    """
    Check if the robot has fallen over.
    Args:
        height_threshold: Minimum height to consider robot upright (meters)
        angle_threshold: Maximum roll/pitch angle to consider robot upright (radians)
    Returns:
        bool: True if robot has fallen, False otherwise
    """
    try:
        robot_position, robot_orientation = config.ISAAC_ROBOT.get_world_pose()
        height = robot_position[2]
        
        # Check height
        if height < height_threshold:
            return True
        
        # Check orientation
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat([robot_orientation[1], robot_orientation[2], robot_orientation[3], robot_orientation[0]])
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        
        # Check if roll or pitch is too extreme
        if abs(roll) > angle_threshold or abs(pitch) > angle_threshold:
            return True
        
        return False
        
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Failed to check if robot fallen: {e}\n")
        return True  # Assume fallen if we can't determine state


def get_ground_contacts():
    """
    Get information about which parts of the robot are in contact with the ground.
    Returns:
        dict: Contact information for each leg
    """
    try:
        # This is a simplified version - you might need to implement proper contact detection
        # For now, return basic contact info based on foot positions
        contacts = {
            'FL': False,
            'FR': False,
            'BL': False,
            'BR': False
        }
        
        # You can enhance this by checking actual physics contacts
        # For now, assume contact if the robot hasn't fallen
        if not is_robot_fallen():
            contacts = {leg: True for leg in contacts}
        
        return contacts
        
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Failed to get ground contacts: {e}\n")
        return {'FL': False, 'FR': False, 'BL': False, 'BR': False}


def reset_robot_state():
    """
    Reset the robot to its initial state for a new RL episode.
    """
    try:
        # Reset robot position to initial pose
        initial_position = numpy.array([0.0, 0.0, 0.2])  # Slightly above ground
        initial_orientation = numpy.array([1.0, 0.0, 0.0, 0.0])  # Upright quaternion (w,x,y,z)
        
        config.ISAAC_ROBOT.set_world_pose(initial_position, initial_orientation)
        
        # Reset joint positions to neutral
        neutral_joint_positions = numpy.zeros(len(config.ISAAC_ROBOT.dof_names))
        config.ISAAC_ROBOT.set_joint_positions(neutral_joint_positions)
        
        # Reset velocities
        zero_velocities = numpy.zeros(len(config.ISAAC_ROBOT.dof_names))
        config.ISAAC_ROBOT.set_joint_velocities(zero_velocities)
        
        # Update SERVO_CONFIG to reflect neutral state
        for leg_id in ['FL', 'FR', 'BL', 'BR']:
            for joint_name in ['hip', 'upper', 'lower']:
                config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = 0.0
        
        import logging
        logging.info("(isaac_sim.py): Robot state reset for new RL episode\n")
        
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Failed to reset robot state: {e}\n")


########## RL TRAINING INFRASTRUCTURE ##########

def should_terminate_episode():
    """
    Check if the current RL episode should be terminated.
    Returns:
        bool: True if episode should end, False otherwise
        str: Reason for termination (if any)
    """
    try:
        # Check if robot has fallen
        if is_robot_fallen():
            return True, "fallen"
        
        # Check if robot has moved too far from origin (optional)
        robot_position, _ = config.ISAAC_ROBOT.get_world_pose()
        distance_from_origin = numpy.linalg.norm(robot_position[:2])  # X,Y distance
        if distance_from_origin > 10.0:  # 10 meter limit
            return True, "out_of_bounds"
        
        # Add more termination conditions as needed
        return False, None
        
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Failed to check episode termination: {e}\n")
        return True, "error"


def compute_rl_reward(previous_state, current_state, action, command):
    """
    Compute the reward for the current step in RL training.
    Args:
        previous_state: Robot state from previous timestep
        current_state: Current robot state
        action: Action taken (joint angles)
        command: Movement command ('w', 's', 'a', 'd', etc.)
    Returns:
        float: Reward value
    """
    try:
        reward = 0.0
        
        if current_state is None or previous_state is None:
            return -10.0  # Penalty for invalid states
        
        # Penalty for falling
        if current_state['is_fallen']:
            return -100.0
        
        # Reward for maintaining height
        target_height = 0.15  # Desired robot height
        height_error = abs(current_state['height'] - target_height)
        reward += 1.0 - (height_error * 10.0)  # Penalty grows with height error
        
        # Reward for forward movement if command is 'w'
        if 'w' in command:
            reward += current_state['forward_velocity'] * 5.0
        
        # Reward for backward movement if command is 's'
        if 's' in command:
            reward -= current_state['forward_velocity'] * 5.0
        
        # Reward for lateral movement
        if 'a' in command:
            reward += current_state['lateral_velocity'] * 3.0
        elif 'd' in command:
            reward -= current_state['lateral_velocity'] * 3.0
        
        # Penalty for excessive rotation
        roll_penalty = abs(current_state['roll']) * 2.0
        pitch_penalty = abs(current_state['pitch']) * 2.0
        reward -= (roll_penalty + pitch_penalty)
        
        # Small penalty for high joint velocities (encourage smooth movement)
        joint_velocity_penalty = numpy.sum(numpy.abs(current_state['joint_velocities'])) * 0.01
        reward -= joint_velocity_penalty
        
        # Small reward for staying upright
        reward += 0.1
        
        return reward
        
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Failed to compute RL reward: {e}\n")
        return -10.0
