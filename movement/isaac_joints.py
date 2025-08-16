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

import logging
import numpy
import time
import threading
import queue
import math

##### import necessary functions #####

from isaacsim.core.utils.types import ArticulationAction


########## CREATE DEPENDENCIES ##########

##### build dependencies for isaac sim #####

ISAAC_MOVEMENT_QUEUE = queue.Queue() # queue for isaac sim to avoid physx threading violations
ISAAC_CALIBRATION_COMPLETE = threading.Event()  # signal for calibration completion





#####################################################
############### ISAAC JOINT FUNCTIONS ###############
#####################################################


########## JOINT CALIBRATION ##########

def calibrate_joints_isaac():
    """
    Calibrate all joints by moving each joint through its full range of motion.
    This function runs indefinitely and cycles through each joint one at a time.
    Only for Isaac Sim - uses queue system to avoid PhysX threading violations.
    """
    if not config.USE_SIMULATION or not config.USE_ISAAC_SIM:
        logging.error("(isaac_joints.py): calibrate_joints_isaac() only works with Isaac Sim\n")
        return
    
    # Define joint order for calibration
    joint_order = [
        ('FL', 'hip'),
        ('FL', 'upper'),
        ('FL', 'lower'),
        ('FR', 'hip'),
        ('FR', 'upper'),
        ('FR', 'lower'),
        ('BL', 'hip'),
        ('BL', 'upper'),
        ('BL', 'lower'),
        ('BR', 'hip'),
        ('BR', 'upper'),
        ('BR', 'lower')
    ]
    
    # Calibration parameters
    step_time = 0.1  # seconds between position updates
    steps_per_movement = 10  # number of steps to complete one movement
    
    logging.info("(isaac_joints.py): Starting joint calibration for Isaac Sim...\n")
    
    joint_index = 0
    while True:
        try:
            # Get current joint to calibrate
            leg_id, joint_name = joint_order[joint_index]
            joint_full_name = f"{leg_id}_{joint_name}"
            
            # Get joint configuration
            servo_data = config.SERVO_CONFIG[leg_id][joint_name]
            is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
            
            # Convert angles to radians - use the actual angle values, not PWM values
            full_back_rad = servo_data['FULL_BACK_ANGLE']  # Already in radians
            full_front_rad = servo_data['FULL_FRONT_ANGLE']  # Already in radians
            neutral_rad = 0.0  # Neutral position at 0 radians
            
            # Apply inversion if needed
            if is_inverted:
                full_back_rad, full_front_rad = full_front_rad, full_back_rad
            
            logging.info(f"(isaac_joints.py): Calibrating {joint_full_name} - Back: {math.degrees(full_back_rad):.1f}°, Front: {math.degrees(full_front_rad):.1f}°, Neutral: {math.degrees(neutral_rad):.1f}°\n")
            
            # Move through the sequence: NEUTRAL -> BACK -> FRONT -> NEUTRAL
            movements = [
                (neutral_rad, full_back_rad),      # Neutral to Back
                (full_back_rad, full_front_rad),   # Back to Front
                (full_front_rad, neutral_rad)      # Front to Neutral
            ]
            
            for start_pos, end_pos in movements:
                # Move through intermediate positions
                for step in range(steps_per_movement):
                    # Clear the completion signal
                    ISAAC_CALIBRATION_COMPLETE.clear()
                    
                    # Linear interpolation between start and end positions
                    progress = step / steps_per_movement
                    current_pos = start_pos + (end_pos - start_pos) * progress
                    
                    # Queue the joint position with slow velocity for calibration
                    queue_single_joint_position_isaac(joint_full_name, current_pos, velocity=0.5)
                    
                    # Wait for this position to be processed before continuing
                    ISAAC_CALIBRATION_COMPLETE.wait(timeout=1.0)
                    
                    # Wait before next step
                    time.sleep(step_time)
            
            # Wait 3 seconds before moving to next joint
            time.sleep(3.0)
            
            # Move to next joint
            joint_index = (joint_index + 1) % len(joint_order)
            
        except Exception as e:
            logging.error(f"(isaac_joints.py): Error in joint calibration: {e}\n")
            time.sleep(1)  # Wait before retrying


def queue_single_joint_position_isaac(joint_name, angle_rad, velocity=0.5):
    """
    Queue a single joint position for Isaac Sim calibration.
    Args:
        joint_name: Full joint name (e.g., 'FL_hip')
        angle_rad: Target angle in radians
        velocity: Joint velocity in radians/second (default: 0.5)
    """
    try:
        # Create calibration movement data
        calibration_data = {
            'type': 'calibration',
            'joint_name': joint_name,
            'angle_rad': angle_rad,
            'velocity': velocity
        }
        
        # Queue the calibration data
        ISAAC_MOVEMENT_QUEUE.put(calibration_data)
        
    except Exception as e:
        logging.error(f"(isaac_joints.py): Failed to queue calibration position for {joint_name}: {e}\n")


def apply_single_joint_position_isaac(joint_name, angle_rad, velocity=0.5):
    """
    Apply a single joint position for Isaac Sim calibration.
    Args:
        joint_name: Full joint name (e.g., 'FL_hip')
        angle_rad: Target angle in radians
        velocity: Joint velocity in radians/second (default: 0.5)
    """
    try:
        joint_index = config.JOINT_INDEX_MAP.get(joint_name)
        if joint_index is None:
            logging.error(f"(isaac_joints.py): Joint {joint_name} not found in JOINT_INDEX_MAP\n")
            return
        
        joint_count = len(config.ISAAC_ROBOT.dof_names)
        joint_positions = numpy.zeros(joint_count)
        joint_velocities = numpy.zeros(joint_count)
        joint_positions[joint_index] = angle_rad
        joint_velocities[joint_index] = velocity
        
        action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
        config.ISAAC_ROBOT_ARTICULATION_CONTROLLER.apply_action(action)
        
    except Exception as e:
        logging.error(f"(isaac_joints.py): Failed to apply calibration position for {joint_name}: {e}\n")


########## ISAAC SIM AI AGENT JOINT CONTROL ##########

def apply_joint_angles_isaac(target_angles, mid_angles, movement_rates):
    """
    Apply joint angles directly for Isaac Sim AI agent training.
    This function moves all joints to their target angles in a single ArticulationAction.
    
    Args:
        current_servo_config: Current servo configuration with CURRENT_ANGLE values
        target_angles: Target joint angles for each leg (similar to SERVO_CONFIG structure)
        mid_angles: Mid joint angles for each leg (similar to SERVO_CONFIG structure)
        movement_rates: Movement rate parameters for each leg
    """
    try:
        joint_count = len(config.ISAAC_ROBOT.dof_names)
        joint_positions = numpy.zeros(joint_count)
        joint_velocities = numpy.zeros(joint_count)
        
        # Define joint order (same as working functions)
        joint_order = [
            ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
            ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
            ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
            ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
        ]
        
        # STEP 1: Move to mid angles first
        for leg_id, joint_name in joint_order:
            joint_full_name = f"{leg_id}_{joint_name}"
            joint_index = config.JOINT_INDEX_MAP.get(joint_full_name)
            
            if joint_index is not None:
                # Get mid angle from the AI agent's output
                mid_angle = mid_angles[leg_id][joint_name]
                
                # Get velocity from movement_rates (already in rad/s, no scaling needed)
                velocity = movement_rates[leg_id].get('speed')  # Already in rad/s
                
                joint_positions[joint_index] = mid_angle
                joint_velocities[joint_index] = velocity
                
                # Update the current angle in config
                config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = mid_angle
            else:
                logging.error(f"(isaac_joints.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
        
        # Apply mid angle positions
        mid_action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
        
        config.ISAAC_ROBOT_ARTICULATION_CONTROLLER.apply_action(mid_action)

        time.sleep(0.05)
        
        # STEP 2: Move to target angles
        for leg_id, joint_name in joint_order:
            joint_full_name = f"{leg_id}_{joint_name}"
            joint_index = config.JOINT_INDEX_MAP.get(joint_full_name)
            
            if joint_index is not None:
                # Get target angle from the AI agent's output
                target_angle = target_angles[leg_id][joint_name]
                
                # Get velocity from movement_rates (already in rad/s, no scaling needed)
                velocity = movement_rates[leg_id].get('speed')  # Already in rad/s
                
                joint_positions[joint_index] = target_angle
                joint_velocities[joint_index] = velocity
                
                # Update the current angle in config
                config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = target_angle
            else:
                logging.error(f"(isaac_joints.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
        
        # Apply target angle positions
        target_action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
        
        config.ISAAC_ROBOT_ARTICULATION_CONTROLLER.apply_action(target_action)
        
        #logging.info(f"(isaac_joints.py): Applied AI agent joint angles for Isaac Sim (mid -> target)")

        time.sleep(0.05)
        
    except Exception as e:
        logging.error(f"(isaac_joints.py): Failed to apply AI agent joint angles for Isaac Sim: {e}\n")


########## NEUTRAL POSITION ##########

def neutral_position_isaac(): # used to move all joints to neutral position in isaac sim
    """
    Set all joints to neutral position for Isaac Sim using direct joint control.
    """
    try:
        joint_count = len(config.ISAAC_ROBOT.dof_names)
        joint_positions = numpy.zeros(joint_count)
        joint_velocities = numpy.zeros(joint_count)
        
        # Define joint order (same as calibration)
        joint_order = [
            ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
            ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
            ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
            ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
        ]
        
        # logging.info("(isaac_joints.py): Moving all joints to neutral position in Isaac Sim...\n")
        
        # Set all joint positions at once
        for leg_id, joint_name in joint_order:
            joint_full_name = f"{leg_id}_{joint_name}"
            joint_index = config.JOINT_INDEX_MAP.get(joint_full_name)
            
            if joint_index is not None:
                servo_data = config.SERVO_CONFIG[leg_id][joint_name]
                neutral_angle = servo_data['NEUTRAL_ANGLE']  # Always 0.0 radians
                
                joint_positions[joint_index] = neutral_angle
                joint_velocities[joint_index] = 1.0  # Moderate velocity
                
                # Update the current angle in config
                config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = neutral_angle
            else:
                logging.error(f"(isaac_joints.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
        
        # Apply all joint positions in a single action
        action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
        config.ISAAC_ROBOT_ARTICULATION_CONTROLLER.apply_action(action)
        
        # logging.info("(isaac_joints.py): Applied all joints to neutral positions\n")
        
    except Exception as e:
        logging.error(f"(isaac_joints.py): Failed to move all joints to neutral: {e}\n")


########## BASIC TESTING FUNCTIONS ##########

##### move all joints to full front #####

def move_all_joints_forward_isaac(): # used to test the maximum range 'outward'
    """
    Move all joints to FULL_FRONT positions for Isaac Sim testing.
    """
    try:
        joint_count = len(config.ISAAC_ROBOT.dof_names)
        joint_positions = numpy.zeros(joint_count)
        joint_velocities = numpy.zeros(joint_count)
        
        # Define joint order
        joint_order = [
            ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
            ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
            ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
            ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
        ]
        
        # Set all joint positions at once
        for leg_id, joint_name in joint_order:
            joint_full_name = f"{leg_id}_{joint_name}"
            joint_index = config.JOINT_INDEX_MAP.get(joint_full_name)
            
            if joint_index is not None:
                servo_data = config.SERVO_CONFIG[leg_id][joint_name]
                full_front_angle = servo_data['FULL_FRONT_ANGLE']  # Already in radians
                
                joint_positions[joint_index] = full_front_angle
                joint_velocities[joint_index] = 1.0  # Moderate velocity
                
                # Update the current angle in config
                config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = full_front_angle
            else:
                logging.error(f"(isaac_joints.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
        
        # Apply all joint positions in a single action
        action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
        config.ISAAC_ROBOT_ARTICULATION_CONTROLLER.apply_action(action)
        
        logging.info("(isaac_joints.py): Applied all joints to FULL_FRONT positions\n")
        
    except Exception as e:
        logging.error(f"(isaac_joints.py): Failed to move all joints to FULL_FRONT: {e}\n")

##### move all joints to full back #####

def move_all_joints_backward_isaac(): # used to test the maximum range 'inward'
    """
    Move all joints to FULL_BACK positions for Isaac Sim testing.
    """
    try:
        joint_count = len(config.ISAAC_ROBOT.dof_names)
        joint_positions = numpy.zeros(joint_count)
        joint_velocities = numpy.zeros(joint_count)
        
        # Define joint order
        joint_order = [
            ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
            ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
            ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
            ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
        ]
        
        # Set all joint positions at once
        for leg_id, joint_name in joint_order:
            joint_full_name = f"{leg_id}_{joint_name}"
            joint_index = config.JOINT_INDEX_MAP.get(joint_full_name)
            
            if joint_index is not None:
                servo_data = config.SERVO_CONFIG[leg_id][joint_name]
                full_back_angle = servo_data['FULL_BACK_ANGLE']  # Already in radians
                
                joint_positions[joint_index] = full_back_angle
                joint_velocities[joint_index] = 1.0  # Moderate velocity
                
                # Update the current angle in config
                config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = full_back_angle
            else:
                logging.error(f"(isaac_joints.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
        
        # Apply all joint positions in a single action
        action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
        config.ISAAC_ROBOT_ARTICULATION_CONTROLLER.apply_action(action)
        
        logging.info("(isaac_joints.py): Applied all joints to FULL_BACK positions\n")
        
    except Exception as e:
        logging.error(f"(isaac_joints.py): Failed to move all joints to FULL_BACK: {e}\n")
