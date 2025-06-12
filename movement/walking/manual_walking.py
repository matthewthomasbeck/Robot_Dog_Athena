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

import time # import time library for time functions
import math # import math library for pi, used with elliptical movement
import logging # import logging for debugging

##### import necessary functions #####

import initialize.initialize_servos as initialize_servos # import servo logic functions
from kinematics.kinematics import Kinematics # import kinematics functions


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

##### leg pairs #####

DIAGONAL_PAIRS = [('FL', 'BR'), ('FR', 'BL')]





#################################################
############### WALKING FUNCTIONS ###############
#################################################


########## CALCULATE INTENSITY ##########

def interpretIntensity(intensity, full_back, full_front): # function to interpret intensity

    ##### find intensity value to calculate arc later #####

    # find intensity by dividing the difference between full_back and full front,
    # converting to positive, dividing by 10, and multiplying by intensity
    arc_length = (abs(full_back - full_front) / 10) * intensity

    ##### find speed and acceleration #####

    if intensity == 1 or intensity == 2:
        speed = int(((16383 / 5) / 10) * intensity)
        acceleration = int(((255 / 5) / 10) * intensity)
    elif intensity == 3 or intensity == 4:
        speed = int(((16383 / 4) / 10) * intensity)
        acceleration = int(((255 / 4) / 10) * intensity)
    elif intensity == 5 or intensity == 6:
        speed = int(((16383 / 3) / 10) * intensity)
        acceleration = int(((255 / 3) / 10) * intensity)
    elif intensity == 7 or intensity == 8:
        speed = int(((16383 / 2) / 10) * intensity)
        acceleration = int(((255 / 2) / 10) * intensity)
    else:
        speed = int((16383 / 10) * intensity)
        acceleration = int((255 / 10) * intensity)

    ##### return arc length speed and acceleration #####

    return arc_length, speed, acceleration # return movement parameters


########## MANUAL TROT ##########


def moveFrontLeftLeg(x, y, z, speed=100, acceleration=10):
    """
    Moves the front left leg to a specified (x, y, z) foot position in meters.
    Uses inverse kinematics and maps angles to servo positions.
    """
    from kinematics.kinematics import Kinematics

    # Run inverse kinematics
    hip_angle, upper_angle, lower_angle = k.inverse_kinematics(x, y, z)

    # Define neutral angle assumptions (you can tune these)
    hip_neutral_angle = 0
    upper_neutral_angle = 90
    lower_neutral_angle = 90

    # Move each joint
    for joint, angle, neutral in zip(
        ['hip', 'upper', 'lower'],
        [hip_angle, upper_angle, lower_angle],
        [hip_neutral_angle, upper_neutral_angle, lower_neutral_angle]
    ):
        servo_data = initialize_servos.SERVO_CONFIG['FL'][joint]
        is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
        pwm = initialize_servos.map_angle_to_servo_position(angle, servo_data, neutral, is_inverted)
        initialize_servos.setTarget(servo_data['servo'], pwm, speed, acceleration)


def moveBackRightLeg(x, y, z, speed=100, acceleration=10):
    hip_angle, upper_angle, lower_angle = k.inverse_kinematics(x, y, z)

    hip_neutral, upper_neutral, lower_neutral = 0, 45, 45

    for joint, angle, neutral in zip(['hip', 'upper', 'lower'], [hip_angle, upper_angle, lower_angle], [hip_neutral, upper_neutral, lower_neutral]):
        servo_data = initialize_servos.SERVO_CONFIG['BR'][joint]
        is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
        pwm = initialize_servos.map_angle_to_servo_position(angle, servo_data, neutral, is_inverted)
        initialize_servos.setTarget(servo_data['servo'], pwm, speed, acceleration)


fl_gait_state = {
    'phase': 'stance',       # current phase: 'stance' or 'swing'
    'last_time': time.time(),  # time of last phase switch (optional here)
    'returned_to_neutral': False
}

br_gait_state = {
    'phase': 'stance',
    'last_time': time.time(),
    'returned_to_neutral': False
}

FL_SWING_POSITION = {'x': -0.0250, 'y': -0.0065,'z': -0.0400}
FL_NEUTRAL_POSITION = {'x': -0.1150, 'y': 0.0065, 'z': -0.0700}
FL_STANCE_POSITION = {'x': -0.1550, 'y': -0.0015, 'z': -0.1100}

BR_SWING_POSITION = {'x': -0.1100, 'y': -0.0085, 'z': -0.1700}
BR_NEUTRAL_POSITION = {'x': -0.0150, 'y': 0.0015, 'z': -0.0700}
BR_STANCE_POSITION = {'x': 0.0250, 'y': -0.0035, 'z': -0.0150}


def updateFrontLeftGaitBD(state):
    global fl_gait_state

    if not state.get('FORWARD', False):
        return  # neutral recovery handled elsewhere

    # If joystick is held, keep alternating between stance/swing
    if fl_gait_state['phase'] == 'stance':
        moveFrontLeftToPosition(FL_SWING_POSITION)
        fl_gait_state['phase'] = 'swing'
    else:
        moveFrontLeftToPosition(FL_STANCE_POSITION)
        fl_gait_state['phase'] = 'stance'

    fl_gait_state['returned_to_neutral'] = False


def updateBackRightGaitBD(state):
    global br_gait_state

    if not state.get('FORWARD', False):
        return

    if br_gait_state['phase'] == 'stance':
        moveBackRightToPosition(BR_SWING_POSITION)
        br_gait_state['phase'] = 'swing'
    else:
        moveBackRightToPosition(BR_STANCE_POSITION)
        br_gait_state['phase'] = 'stance'

    br_gait_state['returned_to_neutral'] = False


def resetFrontLeftForwardGait():
    global fl_gait_state

    if not fl_gait_state['returned_to_neutral']:
        moveFrontLeftToPosition(FL_NEUTRAL_POSITION)
        fl_gait_state['returned_to_neutral'] = True
        logging.info("FL leg returned to neutral forward gait position.")

def resetBackRightForwardGait():
    global br_gait_state

    if not br_gait_state['returned_to_neutral']:
        moveBackRightToPosition(BR_NEUTRAL_POSITION)
        br_gait_state['returned_to_neutral'] = True
        logging.info("BR leg returned to neutral forward gait position.")


def moveFrontLeftToPosition(pos, speed=16383, acceleration=75):
    """
    Moves the FL leg to a given position dictionary with x, y, z.
    """
    moveFrontLeftLeg(
        x=pos['x'],
        y=pos['y'],
        z=pos['z'],
        speed=speed,
        acceleration=acceleration
    )


def moveBackRightToPosition(pos, speed=16383, acceleration=75):
    moveBackRightLeg(
        x=pos['x'],
        y=pos['y'],
        z=pos['z'],
        speed=speed,
        acceleration=acceleration
    )


# TODO FOOT TUNING

foot_position_FL = {
    'x': 0.0,
    'y': initialize_servos.LINK_CONFIG['HIP_OFFSET'],
    'z': -0.10
}
def adjustFL_X(forward=True, delta=0.005):
    direction = 1 if forward else -1
    foot_position_FL['x'] += direction * delta
    _applyFootPosition()
def adjustFL_Y(left=True, delta=0.005):
    direction = 1 if left else -1
    foot_position_FL['y'] += direction * delta
    _applyFootPosition()
def adjustFL_Z(up=True, delta=0.005):
    direction = 1 if up else -1
    foot_position_FL['z'] += direction * delta
    _applyFootPosition()
def _applyFootPosition():
    x, y, z = foot_position_FL['x'], foot_position_FL['y'], foot_position_FL['z']
    moveFrontLeftLeg(x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → FL foot at: x={x:.4f}, y={y:.4f}, z={z:.4f}")

foot_position_BR = {
    'x': 0.0,
    'y': -initialize_servos.LINK_CONFIG['HIP_OFFSET'],  # Negative because BR is on the opposite side
    'z': -0.10
}

def adjustBR_X(forward=True, delta=0.005):
    direction = 1 if forward else -1
    foot_position_BR['x'] += direction * delta
    _applyFootPositionBR()
def adjustBR_Y(left=True, delta=0.005):
    direction = 1 if left else -1
    foot_position_BR['y'] += direction * delta
    _applyFootPositionBR()
def adjustBR_Z(up=True, delta=0.005):
    direction = 1 if up else -1
    foot_position_BR['z'] += direction * delta
    _applyFootPositionBR()
def _applyFootPositionBR():
    x, y, z = foot_position_BR['x'], foot_position_BR['y'], foot_position_BR['z']
    moveBackRightLeg(x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → BR foot at: x={x:.4f}, y={y:.4f}, z={z:.4f}")