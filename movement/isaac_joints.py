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


########## ISAAC SIM AI AGENT JOINT CONTROL ##########

def apply_joint_angles_isaac(target_angles, mid_angles, movement_rates):
    """
    Apply joint angles directly for Isaac Sim AI agent training.
    This function moves all joints to their target angles in a single ArticulationAction.
    
    Args:
        target_angles: Target joint angles for each leg (similar to SERVO_CONFIG structure)
        mid_angles: Mid joint angles for each leg (similar to SERVO_CONFIG structure)
        movement_rates: Individual joint velocities in rad/s for each joint (similar to SERVO_CONFIG structure)
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
                
                # Get individual joint velocity from movement_rates (now per-joint)
                velocity = movement_rates[leg_id][joint_name]  # Individual joint velocity in rad/s
                
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
                
                # Get individual joint velocity from movement_rates (now per-joint)
                velocity = movement_rates[leg_id][joint_name]  # Individual joint velocity in rad/s
                
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
