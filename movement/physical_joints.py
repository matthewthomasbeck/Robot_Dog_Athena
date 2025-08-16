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
import math

##### import necessary functions #####

from utilities.servos import map_angle_to_servo_position, set_target  # import servo mapping functions





########################################################
############### PHYSICAL JOINT FUNCTIONS ###############
########################################################


########## ANGLE-BASED LEG MOVEMENT ##########

##### swing selected leg #####

def swing_leg(leg_id, current_angles, mid_angles, target_angles, movement_rate):
    """
    Swing leg using direct joint angles instead of coordinates.
    Args:
        leg_id: Leg identifier ('FL', 'FR', 'BL', 'BR')
        current_angles: Current joint angles for the leg
        mid_angles: Mid joint angles for the leg
        target_angles: Target joint angles for the leg
        movement_rate: Movement rate parameters
    """
    try:
        speed = movement_rate.get('speed', 16383)  # default to 16383 (max) if not provided
        acceleration = movement_rate.get('acceleration', 255)  # default to 255 (max) if not provided
        
        # Move to mid angles first, then to target angles
        move_joints_to_angles(leg_id, current_angles, mid_angles, speed, acceleration)
        move_joints_to_angles(leg_id, mid_angles, target_angles, speed, acceleration)

    except Exception as e:
        logging.error(f"(physical_joints.py): Failed to swing leg {leg_id} with angles: {e}\n")

##### move joint from mid to target angles #####

def move_joints_to_angles(leg_id, start_angles, end_angles, speed, acceleration):
    """
    Move leg joints to target angles.
    Args:
        leg_id: Leg identifier ('FL', 'FR', 'BL', 'BR')
        start_angles: Starting joint angles for the leg
        end_angles: Ending joint angles for the leg
        speed: Movement speed
        acceleration: Movement acceleration
    """
    for joint_name in ['hip', 'upper', 'lower']:
        try:
            start_angle = start_angles[joint_name]['CURRENT_ANGLE'] if isinstance(start_angles, dict) else start_angles[joint_name]
            end_angle = end_angles[joint_name]
            
            move_joint(leg_id, joint_name, end_angle, speed, acceleration)
            
        except Exception as e:
            logging.error(f"(physical_joints.py): Failed to move {leg_id}_{joint_name} to angle: {e}\n")

##### move single joint to target angle #####

def move_joint(leg_id, joint_name, target_angle, speed, acceleration):
    """
    Move a single joint to target angle.
    Args:
        leg_id: Leg identifier ('FL', 'FR', 'BL', 'BR')
        joint_name: Joint name ('hip', 'upper', 'lower')
        target_angle: Target angle in radians
        speed: Movement speed
        acceleration: Movement acceleration
    """
    servo_data = config.SERVO_CONFIG[leg_id][joint_name]
    pwm = map_angle_to_servo_position(target_angle, servo_data)
    set_target(servo_data['servo'], pwm, speed, acceleration)
    config.SERVO_CONFIG[leg_id][joint_name]['CURRENT'] = pwm
    config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = target_angle


########## NEUTRAL POSITION ##########

def neutral_position_physical(intensity): # used to move all joints to neutral position on physical robot
    """
    Set all legs to neutral position for physical robot using direct joint control.
    """
    # Define joint order (same as Isaac Sim version for consistency)
    joint_order = [
        ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
        ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
        ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
        ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
    ]
    
    logging.info("(physical_joints.py): Moving all legs to neutral position on physical robot...\n")
    
    # Set default speed and acceleration for neutral positioning
    speed = 16383  # default to max speed
    acceleration = 255  # default to max acceleration
    
    # Move each joint to its neutral position
    for leg_id, joint_name in joint_order:
        try:
            servo_data = config.SERVO_CONFIG[leg_id][joint_name]
            neutral_angle = servo_data['NEUTRAL_ANGLE']  # Always 0.0 radians
            
            # Use the angle-based movement system
            move_joint(leg_id, joint_name, neutral_angle, speed, acceleration)
            
        except Exception as e:
            logging.error(f"(physical_joints.py): Failed to move {leg_id}_{joint_name} to neutral: {e}\n")
