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

##### import necessary functions #####

import utilities.servos as initialize_servos # import servo logic functions
from utilities.mathematics import * # import all mathematical functions
from movement.standing.standing_inplace import neutral_standing_position # import neutral standing position function


########## CREATE DEPENDENCIES ##########

##### initialize kinematics #####

k = Kinematics(initialize_servos.LINK_CONFIG) # use link lengths to initialize kinematic functions

##### define servos #####

upper_leg_servos = { # define upper leg servos

    "FL": initialize_servos.SERVO_CONFIG['FL']['upper'],  # front left
    "FR": initialize_servos.SERVO_CONFIG['FR']['upper'],  # front right
    "BL": initialize_servos.SERVO_CONFIG['BL']['upper'],  # back left
    "BR": initialize_servos.SERVO_CONFIG['BR']['upper'],  # back right
}

lower_leg_servos = { # define lower leg servos

    "FL": initialize_servos.SERVO_CONFIG['FL']['lower'],  # front left
    "FR": initialize_servos.SERVO_CONFIG['FR']['lower'],  # front right
    "BL": initialize_servos.SERVO_CONFIG['BL']['lower'],  # back left
    "BR": initialize_servos.SERVO_CONFIG['BR']['lower'],  # back right
}





##############################################################
############### FUNDAMENTAL MOVEMENT FUNCTIONS ###############
##############################################################


########## MOVE FOOT FUNCTION ##########

def move_leg_to_pos(leg_id, pos, speed, acceleration, stride_scalar, use_bezier=False):
    if use_bezier:
        p0 = neutral_standing_position(leg_id, stride_scalar)
        p2 = {
            'x': pos['x'] * stride_scalar,
            'y': pos['y'] * stride_scalar,
            'z': pos['z']
        }
        p1 = {
            'x': (p0['x'] + p2['x']) / 2,
            'y': (p0['y'] + p2['y']) / 2,
            'z': max(p0['z'], p2['z']) + 0.025
        }

        curve = bezier_curve(
            p0=(p0['x'], p0['y'], p0['z']),
            p1=(p1['x'], p1['y'], p1['z']),
            p2=(p2['x'], p2['y'], p2['z']),
            steps=4
        )

        for x, y, z in curve:
            moveLeg(leg_id, x, y, z, speed, acceleration)
            # TODO: test adding time.sleep here

    else:
        moveLeg(
            leg_id,
            x=pos['x'] * stride_scalar,
            y=pos['y'] * stride_scalar,
            z=pos['z'],
            speed=speed,
            acceleration=acceleration
        )


########## MOVE LEG FUNCTION ##########

def moveLeg(leg_id, x, y, z, speed, acceleration):
    hip_angle, upper_angle, lower_angle = k.inverse_kinematics(x, y, z)
    hip_neutral, upper_neutral, lower_neutral = 0, 45, 45

    for joint, angle, neutral in zip(['hip', 'upper', 'lower'],
                                     [hip_angle, upper_angle, lower_angle],
                                     [hip_neutral, upper_neutral, lower_neutral]):

        joint_speed = max(1, speed // 6) if joint in ['upper', 'hip'] else speed
        joint_acceleration = max(1, acceleration // 6) if joint in ['upper', 'hip'] else acceleration

        servo_data = initialize_servos.SERVO_CONFIG[leg_id][joint]
        is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
        pwm = initialize_servos.map_angle_to_servo_position(angle, servo_data, neutral, is_inverted)
        initialize_servos.setTarget(servo_data['servo'], pwm, joint_speed, joint_acceleration)


########## FOOT TUNING ##########

foot_position_FL = {'x': -0.0450, 'y': -0.0165, 'z': -0.0750}
foot_position_FR = {'x': 0.0100, 'y': -0.0015, 'z': -0.1050}
foot_position_BL = {'x': -0.0250, 'y': 0.0065, 'z': -0.0600}
foot_position_BR = {'x': 0.0000, 'y': -0.0085, 'z': -0.0850}

def adjustFL_X(forward=True, delta=0.005):
    direction = 1 if forward else -1
    foot_position_FL['x'] += direction * delta
    _applyFootPosition()
def adjustBR_X(forward=True, delta=0.005):
    direction = 1 if forward else -1
    foot_position_BR['x'] += direction * delta
    _applyFootPositionBR()
def adjustFR_X(forward=True, delta=0.005):
    direction = 1 if forward else -1
    foot_position_FR['x'] += direction * delta
    _applyFootPositionFR()
def adjustBL_X(forward=True, delta=0.005):
    direction = 1 if forward else -1
    foot_position_BL['x'] += direction * delta
    _applyFootPositionBL()

def adjustFL_Y(left=True, delta=0.005):
    direction = 1 if left else -1
    foot_position_FL['y'] += direction * delta
    _applyFootPosition()
def adjustBR_Y(left=True, delta=0.005):
    direction = 1 if left else -1
    foot_position_BR['y'] += direction * delta
    _applyFootPositionBR()
def adjustFR_Y(left=True, delta=0.005):
    direction = 1 if left else -1
    foot_position_FR['y'] += direction * delta
    _applyFootPositionFR()
def adjustBL_Y(left=True, delta=0.005):
    direction = 1 if left else -1
    foot_position_BL['y'] += direction * delta
    _applyFootPositionBL()

def adjustFL_Z(up=True, delta=0.005):
    direction = 1 if up else -1
    foot_position_FL['z'] += direction * delta
    _applyFootPosition()
def adjustBR_Z(up=True, delta=0.005):
    direction = 1 if up else -1
    foot_position_BR['z'] += direction * delta
    _applyFootPositionBR()
def adjustFR_Z(up=True, delta=0.005):
    direction = 1 if up else -1
    foot_position_FR['z'] += direction * delta
    _applyFootPositionFR()
def adjustBL_Z(up=True, delta=0.005):
    direction = 1 if up else -1
    foot_position_BL['z'] += direction * delta
    _applyFootPositionBL()

def _applyFootPosition():
    x, y, z = foot_position_FL['x'], foot_position_FL['y'], foot_position_FL['z']
    moveLeg('FL', x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → FL foot at: x={x:.4f}, y={y:.4f}, z={z:.4f}")
def _applyFootPositionBR():
    x, y, z = foot_position_BR['x'], foot_position_BR['y'], foot_position_BR['z']
    moveLeg('BR', x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → BR foot at: x={x:.4f}, y={y:.4f}, z={z:.4f}")
def _applyFootPositionFR():
    x, y, z = foot_position_FR['x'], foot_position_FR['y'], foot_position_FR['z']
    moveLeg('FR', x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → FR foot at: x={x:.4f}, y={y:.4f}, z={z:.4f}")
def _applyFootPositionBL():
    x, y, z = foot_position_BL['x'], foot_position_BL['y'], foot_position_BL['z']
    moveLeg('BL', x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → BL foot at: x={x:.4f}, y={y:.4f}, z={z:.4f}")