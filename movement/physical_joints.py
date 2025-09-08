##################################################################################
# Copyright (c) 2025 Matthew Thomas Beck                                         #
#                                                                                #
# Personal and educational use only. This code and its associated files may be   #
# copied, modified, and distributed by individuals for non-commercial purposes. #
# Commercial use by companies or for-profit entities is prohibited.              #
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
import time

##### import necessary functions #####

from utilities.servos import map_angle_to_servo_position, set_target, map_radian_to_servo_speed





########################################################
############### PHYSICAL JOINT FUNCTIONS ###############
########################################################


########## ANGLE-BASED LEG MOVEMENT ##########

##### swing selected leg #####

def swing_leg(leg_id, target_angles, movement_rates):
    """
    Swing leg using direct joint angles instead of coordinates.
    Args:
        leg_id: Leg identifier ('FL', 'FR', 'BL', 'BR')
        target_angles: Target joint angles for the leg (dict with 'hip', 'upper', 'lower' keys mapping to angle values in radians)
        movement_rates: Movement rates for each joint (dict with 'hip', 'upper', 'lower' keys mapping to speed values in rad/s)
    """
    try:
        # Move directly to target angles (no mid angles needed)
        move_joints_to_angles(leg_id, target_angles, movement_rates)
        time.sleep(0.05)

    except Exception as e:
        logging.error(f"(physical_joints.py): Failed to swing leg {leg_id} with angles: {e}\n")

##### move joint from mid to target angles #####

def move_joints_to_angles(leg_id, end_angles, movement_rates):
    """
    Move leg joints to target angles.
    Args:
        leg_id: Leg identifier ('FL', 'FR', 'BL', 'BR')
        start_angles: Starting joint angles for the leg (dict with 'hip', 'upper', 'lower' keys mapping to angle values in radians)
        end_angles: Ending joint angles for the leg (dict with 'hip', 'upper', 'lower' keys mapping to angle values in radians)
        movement_rates: Movement rates for each joint (dict with 'hip', 'upper', 'lower' keys mapping to speed values in rad/s)
    """
    for joint_name in ['hip', 'upper', 'lower']:
        try:
            # All angle parameters now have the correct structure with direct angle values
            end_angle = end_angles[joint_name]
            speed = movement_rates[joint_name]
            
            move_joint(leg_id, joint_name, end_angle, speed)
            
        except Exception as e:
            logging.error(f"(physical_joints.py): Failed to move {leg_id}_{joint_name} to angle: {e}\n")

##### move single joint to target angle #####

def move_joint(leg_id, joint_name, target_angle, speed):
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
    speed = map_radian_to_servo_speed(speed)
    set_target(servo_data['servo'], pwm, speed, 255) # use 255 max acceleration
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
    
    # Set default speed for neutral positioning
    speed = 9.5  # default to max speed
    
    # Move each joint to its neutral position
    for leg_id, joint_name in joint_order:
        try:
            servo_data = config.SERVO_CONFIG[leg_id][joint_name]
            neutral_angle = servo_data['NEUTRAL_ANGLE']  # Always 0.0 radians
            
            # Use the angle-based movement system
            move_joint(leg_id, joint_name, neutral_angle, speed)
            
        except Exception as e:
            logging.error(f"(physical_joints.py): Failed to move {leg_id}_{joint_name} to neutral: {e}\n")
