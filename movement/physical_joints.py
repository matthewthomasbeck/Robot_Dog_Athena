##################################################################################
# Copyright (c) 2025 Matthew Thomas Beck                                         #
#                                                                                #
# Licensed under the Creative Commons Attribution-NonCommercial 4.0              #
# International (CC BY-NC 4.0). Personal and educational use is permitted.       #
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
import time

##### import necessary functions #####

from utilities.servos import map_angle_to_servo_position, set_target, map_radian_to_servo_speed





########################################################
############### PHYSICAL JOINT FUNCTIONS ###############
########################################################


########## ANGLE-BASED LEG MOVEMENT ##########

##### swing selected leg #####

def swing_leg(leg_id, target_angles, movement_rates):

    try:
        # Move directly to target angles (no mid angles needed)
        move_joints_to_angles(leg_id, target_angles, movement_rates)
        time.sleep(0.05)

    except Exception as e:
        logging.error(f"(physical_joints.py): Failed to swing leg {leg_id} with angles: {e}\n")

##### move joint from mid to target angles #####

def move_joints_to_angles(leg_id, end_angles, movement_rates):

    ##### move each joint to its target angle #####

    for joint_name in ['hip', 'upper', 'lower']: # loop through each joint in the leg

        try: # attempt to move the joint
            end_angle = end_angles[joint_name]
            speed = movement_rates[joint_name]
            move_joint(leg_id, joint_name, end_angle, speed)
            
        except Exception as e: # if unable to move joint...
            logging.error(f"(physical_joints.py): Failed to move {leg_id}_{joint_name} to angle: {e}\n")

##### move single joint to target angle #####

def move_joint(leg_id, joint_name, target_angle, speed):

    ##### move the joint to the target angle at the specified speed #####

    servo_data = config.SERVO_CONFIG[leg_id][joint_name]
    pwm = map_angle_to_servo_position(target_angle, servo_data)
    speed = map_radian_to_servo_speed(speed)
    set_target(servo_data['servo'], pwm, speed, 255) # use 255 max acceleration
    config.SERVO_CONFIG[leg_id][joint_name]['CURRENT'] = pwm
    config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = target_angle


########## NEUTRAL POSITION ##########

def neutral_position_physical(intensity): # used to move all joints to neutral position on physical robot

    ##### set variables #####

    speed = 9.5  # default to max speed
    joint_order = [
        ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
        ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
        ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
        ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
    ]
    
    logging.info("(physical_joints.py): Moving all legs to neutral position on physical robot...\n")
    
    ##### move each joint to neutral position #####

    for leg_id, joint_name in joint_order: # loop through each joint in the leg

        try: # attempt to move the joint
            servo_data = config.SERVO_CONFIG[leg_id][joint_name]
            neutral_angle = servo_data['NEUTRAL_ANGLE']
            move_joint(leg_id, joint_name, neutral_angle, speed) # angle based movement system
            
        except Exception as e: # if unable to move joint...
            logging.error(f"(physical_joints.py): Failed to move {leg_id}_{joint_name} to neutral: {e}\n")
