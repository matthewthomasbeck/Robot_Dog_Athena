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
                
            # Safety check: ensure we have data for this robot
            if robot_idx >= len(all_mid_angles) or robot_idx >= len(all_movement_rates):
                logging.warning(f"(isaac_joints.py): No data available for robot {robot_idx}, skipping...")
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
                    # Safety check: ensure the robot data has the expected structure
                    if (leg_id in all_mid_angles[robot_idx] and 
                        joint_name in all_mid_angles[robot_idx][leg_id] and
                        leg_id in all_movement_rates[robot_idx] and 
                        joint_name in all_movement_rates[robot_idx][leg_id]):
                        
                        # Get mid angle from the AI agent's output
                        mid_angle = all_mid_angles[robot_idx][leg_id][joint_name]
                        
                        # Get individual joint velocity from movement_rates
                        velocity = all_movement_rates[robot_idx][leg_id][joint_name]
                        
                        joint_positions[joint_index] = mid_angle
                        joint_velocities[joint_index] = velocity
                    else:
                        logging.warning(f"(isaac_joints.py): Missing data for robot {robot_idx} {leg_id}_{joint_name}, using neutral position")
                        # Use neutral position as fallback
                        servo_data = config.SERVO_CONFIG[leg_id][joint_name]
                        neutral_angle = servo_data['NEUTRAL_ANGLE']
                        joint_positions[joint_index] = neutral_angle
                        joint_velocities[joint_index] = 1.0
                else:
                    logging.error(f"(isaac_joints.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
            
            # Apply mid angle positions for this robot
            mid_action = ArticulationAction(
                joint_positions=joint_positions,
                joint_velocities=joint_velocities
            )
            
            config.ISAAC_ROBOT_ARTICULATION_CONTROLLERS[robot_idx].apply_action(mid_action)

        # Wait after applying all mid angles
        time.sleep(0.05)
        
        # STEP 2: Move all robots to target angles
        for robot_idx in range(num_robots):
            if robot_idx >= len(config.ISAAC_ROBOT_ARTICULATION_CONTROLLERS):
                continue
                
            # Safety check: ensure we have data for this robot
            if robot_idx >= len(all_target_angles) or robot_idx >= len(all_movement_rates):
                logging.warning(f"(isaac_joints.py): No data available for robot {robot_idx}, skipping...")
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
                    # Safety check: ensure the robot data has the expected structure
                    if (leg_id in all_target_angles[robot_idx] and 
                        joint_name in all_target_angles[robot_idx][leg_id] and
                        leg_id in all_movement_rates[robot_idx] and 
                        joint_name in all_movement_rates[robot_idx][leg_id]):
                        
                        # Get target angle from the AI agent's output
                        target_angle = all_target_angles[robot_idx][leg_id][joint_name]
                        
                        # Get individual joint velocity from movement_rates
                        velocity = all_movement_rates[robot_idx][leg_id][joint_name]
                        
                        joint_positions[joint_index] = target_angle
                        joint_velocities[joint_index] = velocity
                        
                        # Update the current angle in lightweight array
                        config.CURRENT_ANGLES[robot_idx][leg_id][joint_name] = target_angle
                    else:
                        logging.warning(f"(isaac_joints.py): Missing data for robot {robot_idx} {leg_id}_{joint_name}, using neutral position")
                        # Use neutral position as fallback
                        servo_data = config.SERVO_CONFIG[leg_id][joint_name]
                        neutral_angle = servo_data['NEUTRAL_ANGLE']
                        joint_positions[joint_index] = neutral_angle
                        joint_velocities[joint_index] = 1.0
                        config.CURRENT_ANGLES[robot_idx][leg_id][joint_name] = neutral_angle
                else:
                    logging.error(f"(isaac_joints.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
            
            # Apply target angle positions for this robot
            target_action = ArticulationAction(
                joint_positions=joint_positions,
                joint_velocities=joint_velocities
            )
            
            config.ISAAC_ROBOT_ARTICULATION_CONTROLLERS[robot_idx].apply_action(target_action)

        # Wait after applying all target angles
        time.sleep(0.05)
        
    except Exception as e:
        logging.error(f"(isaac_joints.py): Failed to apply multi-robot joint angles: {e}\n")


def reset_individual_robot(robot_id):
    """
    Reset a specific robot to its spawn position with neutral joint angles.
    This function resets only one robot without affecting others.
    
    Args:
        robot_id (int): ID of the robot to reset (0-based index)
    """
    try:
        # Safety check: ensure robot_id is valid
        if robot_id >= len(config.ISAAC_ROBOTS) or robot_id >= len(config.ISAAC_ROBOT_ARTICULATION_CONTROLLERS):
            logging.error(f"(isaac_joints.py): Invalid robot_id {robot_id}\n")
            return False
            
        # Safety check: ensure spawn positions are available
        if not hasattr(config, 'SPAWN_POSITIONS') or robot_id >= len(config.SPAWN_POSITIONS):
            logging.error(f"(isaac_joints.py): Spawn position not available for robot {robot_id}\n")
            return False
        
        logging.info(f"(isaac_joints.py): Resetting robot {robot_id} to spawn position...\n")
        
        # STEP 1: Move robot to its spawn position
        robot_path = f"/World/robot_dog_{robot_id}"
        spawn_pos = config.SPAWN_POSITIONS[robot_id]
        
        import omni.usd
        from pxr import UsdGeom, Gf
        
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(robot_path)
        
        if robot_prim:
            xform = UsdGeom.Xformable(robot_prim)
            if xform:
                xform_ops = xform.GetOrderedXformOps()
                translate_op = None
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                
                if translate_op:
                    translate_op.Set(Gf.Vec3d(spawn_pos[0], spawn_pos[1], spawn_pos[2]))
                    logging.info(f"(isaac_joints.py): Robot {robot_id} repositioned to spawn position {spawn_pos}\n")
                else:
                    logging.warning(f"(isaac_joints.py): No translate operation found on robot {robot_id} prim.\n")
            else:
                logging.warning(f"(isaac_joints.py): Robot {robot_id} prim found but not Xformable.\n")
        else:
            logging.error(f"(isaac_joints.py): Robot {robot_id} prim not found at path {robot_path}\n")
            return False
        
        # STEP 2: Set robot joints to neutral position
        joint_count = len(config.ISAAC_ROBOTS[robot_id].dof_names)
        joint_positions = numpy.zeros(joint_count)
        joint_velocities = numpy.zeros(joint_count)
        
        # Define joint order (same as calibration)
        joint_order = [
            ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
            ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
            ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
            ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
        ]
        
        # Set all joint positions to neutral for this robot
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
                config.CURRENT_ANGLES[robot_id][leg_id][joint_name] = neutral_angle
            else:
                logging.error(f"(isaac_joints.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
        
        # Apply neutral joint positions for this robot
        action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
        config.ISAAC_ROBOT_ARTICULATION_CONTROLLERS[robot_id].apply_action(action)
        
        # Give Isaac Sim a few steps to stabilize after reset
        for _ in range(3):
            config.ISAAC_WORLD.step(render=True)
        
        logging.info(f"(isaac_joints.py): Robot {robot_id} reset complete - position and joints reset\n")
        return True
        
    except Exception as e:
        logging.error(f"(isaac_joints.py): Failed to reset robot {robot_id}: {e}\n")
        return False


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
