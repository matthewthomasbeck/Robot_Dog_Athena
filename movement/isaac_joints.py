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

def apply_joint_angles_isaac(all_target_angles, all_mid_angles, all_movement_rates):
    """
    Apply joint angles for all robots in Isaac Sim.
    This function moves all robots to their mid angles, waits, then moves to target angles.
    
    Args:
        all_target_angles: List of target joint angles for each robot
        all_mid_angles: List of mid joint angles for each robot  
        all_movement_rates: List of movement rates for each robot
    """
    try:
        num_robots = len(config.ISAAC_ROBOTS)
        
        # STEP 1: Move all robots to mid angles
        for robot_idx in range(num_robots):
            if robot_idx >= len(config.ISAAC_ROBOT_ARTICULATION_CONTROLLERS):
                logging.error(f"(isaac_joints.py): Robot {robot_idx} controller not found\n")
                continue
                
            joint_count = len(config.ISAAC_ROBOTS[robot_idx].dof_names)
            joint_positions = numpy.zeros(joint_count)
            joint_velocities = numpy.zeros(joint_count)
            
            # Define joint order (same as working functions)
            joint_order = [
                ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
                ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
                ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
                ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
            ]
            
            # Set mid angles for this robot
            for leg_id, joint_name in joint_order:
                joint_full_name = f"{leg_id}_{joint_name}"
                joint_index = config.JOINT_INDEX_MAP.get(joint_full_name)
                
                if joint_index is not None:
                    # Get mid angle from the AI agent's output
                    mid_angle = all_mid_angles[robot_idx][leg_id][joint_name]
                    
                    # Get individual joint velocity from movement_rates
                    velocity = all_movement_rates[robot_idx][leg_id][joint_name]
                    
                    joint_positions[joint_index] = mid_angle
                    joint_velocities[joint_index] = velocity
                else:
                    logging.error(f"(isaac_joints.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
            
            # Apply mid angle positions for this robot
            mid_action = ArticulationAction(
                joint_positions=joint_positions,
                joint_velocities=joint_velocities
            )
            
            config.ISAAC_ROBOT_ARTICULATION_CONTROLLERS[robot_idx].apply_action(mid_action)

        # Wait after applying all mid angles
        #time.sleep(0.05)
        
        # STEP 2: Move all robots to target angles
        for robot_idx in range(num_robots):
            if robot_idx >= len(config.ISAAC_ROBOT_ARTICULATION_CONTROLLERS):
                continue
                
            joint_count = len(config.ISAAC_ROBOTS[robot_idx].dof_names)
            joint_positions = numpy.zeros(joint_count)
            joint_velocities = numpy.zeros(joint_count)
            
            # Define joint order (same as working functions)
            joint_order = [
                ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
                ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
                ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
                ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
            ]
            
            # Set target angles for this robot
            for leg_id, joint_name in joint_order:
                joint_full_name = f"{leg_id}_{joint_name}"
                joint_index = config.JOINT_INDEX_MAP.get(joint_full_name)
                
                if joint_index is not None:
                    # Get target angle from the AI agent's output
                    target_angle = all_target_angles[robot_idx][leg_id][joint_name]
                    
                    # Get individual joint velocity from movement_rates
                    velocity = all_movement_rates[robot_idx][leg_id][joint_name]
                    
                    joint_positions[joint_index] = target_angle
                    joint_velocities[joint_index] = velocity
                    
                    # Update the current angle in lightweight array
                    config.CURRENT_ANGLES[robot_idx][leg_id][joint_name] = target_angle
                else:
                    logging.error(f"(isaac_joints.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
            
            # Apply target angle positions for this robot
            target_action = ArticulationAction(
                joint_positions=joint_positions,
                joint_velocities=joint_velocities
            )
            
            config.ISAAC_ROBOT_ARTICULATION_CONTROLLERS[robot_idx].apply_action(target_action)

        # Wait after applying all target angles
        #time.sleep(0.05)
        
    except Exception as e:
        logging.error(f"(isaac_joints.py): Failed to apply multi-robot joint angles: {e}\n")


def neutral_position_isaac(): # used to move all joints to neutral position in isaac sim
    """
    Set all joints to neutral position for Isaac Sim using direct joint control.
    """
    try:
        # Loop through all robots
        for robot_idx in range(len(config.ISAAC_ROBOTS)):
            if robot_idx >= len(config.ISAAC_ROBOT_ARTICULATION_CONTROLLERS):
                logging.error(f"(isaac_joints.py): Robot {robot_idx} controller not found\n")
                continue
                
            joint_count = len(config.ISAAC_ROBOTS[robot_idx].dof_names)
            joint_positions = numpy.zeros(joint_count)
            joint_velocities = numpy.zeros(joint_count)
            
            # Define joint order (same as calibration)
            joint_order = [
                ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
                ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
                ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
                ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
            ]
            
            # Set all joint positions at once for this robot
            for leg_id, joint_name in joint_order:
                joint_full_name = f"{leg_id}_{joint_name}"
                joint_index = config.JOINT_INDEX_MAP.get(joint_full_name)
                
                if joint_index is not None:
                    # Get neutral angle from static config (same for all robots)
                    servo_data = config.SERVO_CONFIG[leg_id][joint_name]
                    neutral_angle = servo_data['NEUTRAL_ANGLE']  # Always 0.0 radians
                    
                    joint_positions[joint_index] = neutral_angle
                    joint_velocities[joint_index] = 1.0  # Moderate velocity
                    
                    # Update the current angle in lightweight array
                    config.CURRENT_ANGLES[robot_idx][leg_id][joint_name] = neutral_angle
                else:
                    logging.error(f"(isaac_joints.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
            
            # Apply all joint positions in a single action for this robot
            action = ArticulationAction(
                joint_positions=joint_positions,
                joint_velocities=joint_velocities
            )
            config.ISAAC_ROBOT_ARTICULATION_CONTROLLERS[robot_idx].apply_action(action)
            
    except Exception as e:
        logging.error(f"(isaac_joints.py): Failed to move all joints to neutral: {e}\n")
