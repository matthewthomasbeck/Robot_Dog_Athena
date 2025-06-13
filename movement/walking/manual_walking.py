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

def interpretIntensity(intensity): # function to interpret intensity (full_back/front DEPRECATED)

    ##### find speed, acceleration, stride_scalar #####

    if intensity == 1 or intensity == 2:
        speed = int(((16383 / 5) / 10) * intensity)
        acceleration = int(((255 / 5) / 10) * intensity)
        stride_scalar = 0.2  # default stride scalar for high intensity
    elif intensity == 3 or intensity == 4:
        speed = int(((16383 / 4) / 10) * intensity)
        acceleration = int(((255 / 4) / 10) * intensity)
        stride_scalar = 0.4  # default stride scalar for high intensity
    elif intensity == 5 or intensity == 6:
        speed = int(((16383 / 3) / 10) * intensity)
        acceleration = int(((255 / 3) / 10) * intensity)
        stride_scalar = 0.6  # default stride scalar for high intensity
    elif intensity == 7 or intensity == 8:
        speed = int(((16383 / 2) / 10) * intensity)
        acceleration = int(((255 / 2) / 10) * intensity)
        stride_scalar = 0.8  # default stride scalar for high intensity
    else:
        speed = int((16383 / 10) * intensity)
        acceleration = int((255 / 10) * intensity)
        stride_scalar = 1.0 # default stride scalar for high intensity

    ##### return arc length speed and acceleration #####

    return speed, acceleration, stride_scalar # return movement parameters


########## LEG PHASE CONFIG ##########

##### gait states #####

fl_gait_state = {'phase': 'stance', 'last_time': time.time(), 'returned_to_neutral': False}
br_gait_state = {'phase': 'stance', 'last_time': time.time(), 'returned_to_neutral': False}
fr_gait_state = {'phase': 'swing', 'last_time': time.time(), 'returned_to_neutral': False}
bl_gait_state = {'phase': 'swing', 'last_time': time.time(), 'returned_to_neutral': False}

##### phase positions #####

FL_SWING_POSITION = {'x': -0.0250, 'y': -0.0065,'z': -0.0400}
FL_NEUTRAL_POSITION = {'x': -0.1150, 'y': 0.0065, 'z': -0.0700}
FL_STANCE_POSITION = {'x': -0.1550, 'y': -0.0015, 'z': -0.1100}

BR_SWING_POSITION = {'x': -0.1100, 'y': -0.0085, 'z': -0.1700}
BR_NEUTRAL_POSITION = {'x': -0.0150, 'y': 0.0015, 'z': -0.0700}
BR_STANCE_POSITION = {'x': 0.0250, 'y': -0.0035, 'z': -0.0150}

FR_SWING_POSITION = {'x': -0.1200, 'y': -0.0465, 'z': -0.3950}
FR_NEUTRAL_POSITION = {'x': -0.0000, 'y': 0.0035, 'z': -0.1150}
FR_STANCE_POSITION = {'x': 0.0050, 'y': -0.0015, 'z': -0.0300}

BL_SWING_POSITION = {'x': 0.0150, 'y': -0.0035, 'z': -0.0100}
BL_NEUTRAL_POSITION = {'x': 0.0200, 'y': 0.0115, 'z': -0.0850}
BL_STANCE_POSITION = {'x': 0.0150, 'y': 0.0315, 'z': -0.2250}

##### gait timing #####

GAIT_CYCLE_DURATION = 0.6
GAIT_HALF_PHASE = GAIT_CYCLE_DURATION / 2


########## GAIT FUNCTIONS ##########

def trotForward(intensity):

    speed, acceleration, stride_scalar = interpretIntensity(intensity)

    updateFrontLeftGaitBD({'FORWARD': True}, speed, acceleration, stride_scalar)
    time.sleep(.15)
    updateBackRightGaitBD({'FORWARD': True}, speed, acceleration, stride_scalar)
    time.sleep(.15)
    updateFrontRightGaitBD({'FORWARD': True}, speed, acceleration, stride_scalar)
    time.sleep(.15)
    updateBackLeftGaitBD({'FORWARD': True}, speed, acceleration, stride_scalar)
    time.sleep(.15)


########## UPDATE LEG GAITS ##########

def updateFrontLeftGaitBD(state, speed, acceleration, stride_scalar):
    global fl_gait_state

    if not state.get('FORWARD', False):
        return  # neutral recovery handled elsewhere

    # If joystick is held, keep alternating between stance/swing
    if fl_gait_state['phase'] == 'stance':
        moveFrontLeftToPosition(FL_SWING_POSITION, speed, acceleration, stride_scalar)
        fl_gait_state['phase'] = 'swing'
    else:
        moveFrontLeftToPosition(FL_STANCE_POSITION, speed, acceleration, stride_scalar)
        fl_gait_state['phase'] = 'stance'

    fl_gait_state['returned_to_neutral'] = False

def updateBackRightGaitBD(state, speed, acceleration, stride_scalar):
    global br_gait_state

    if not state.get('FORWARD', False):
        return

    if br_gait_state['phase'] == 'stance':
        moveBackRightToPosition(BR_SWING_POSITION, speed, acceleration, stride_scalar)
        br_gait_state['phase'] = 'swing'
    else:
        moveBackRightToPosition(BR_STANCE_POSITION, speed, acceleration, stride_scalar)
        br_gait_state['phase'] = 'stance'

    br_gait_state['returned_to_neutral'] = False

def updateFrontRightGaitBD(state, speed, acceleration, stride_scalar):
    global fr_gait_state

    if not state.get('FORWARD', False):
        return  # neutral recovery handled elsewhere

    # If joystick is held, keep alternating between stance/swing
    if fr_gait_state['phase'] == 'stance':
        moveFrontRightToPosition(FR_SWING_POSITION, speed, acceleration, stride_scalar)
        fr_gait_state['phase'] = 'swing'
    else:
        moveFrontRightToPosition(FR_STANCE_POSITION, speed, acceleration, stride_scalar)
        fr_gait_state['phase'] = 'stance'

    fr_gait_state['returned_to_neutral'] = False

def updateBackLeftGaitBD(state, speed, acceleration, stride_scalar):
    global bl_gait_state

    if not state.get('FORWARD', False):
        return  # neutral recovery handled elsewhere

    # If joystick is held, keep alternating between stance/swing
    if bl_gait_state['phase'] == 'stance':
        moveBackLeftToPosition(BL_SWING_POSITION, speed, acceleration, stride_scalar)
        bl_gait_state['phase'] = 'swing'
    else:
        moveBackLeftToPosition(BL_STANCE_POSITION, speed, acceleration, stride_scalar)
        bl_gait_state['phase'] = 'stance'

    bl_gait_state['returned_to_neutral'] = False


########## RESET LEG GAITS ##########

def resetFrontLeftForwardGait():
    global fl_gait_state

    if not fl_gait_state['returned_to_neutral']:
        moveFrontLeftToPosition(FL_NEUTRAL_POSITION, speed=16383, acceleration=255, stride_scalar=1) # reset quickly
        fl_gait_state['returned_to_neutral'] = True
        logging.info("FL leg returned to neutral forward gait position.")

def resetBackRightForwardGait():
    global br_gait_state

    if not br_gait_state['returned_to_neutral']:
        moveBackRightToPosition(BR_NEUTRAL_POSITION, speed=16383, acceleration=255, stride_scalar=1) # reset quickly
        br_gait_state['returned_to_neutral'] = True
        logging.info("BR leg returned to neutral forward gait position.")

def resetFrontRightForwardGait():
    global fr_gait_state

    if not fr_gait_state['returned_to_neutral']:
        moveFrontRightToPosition(FR_NEUTRAL_POSITION, speed=16383, acceleration=255, stride_scalar=1) # reset quickly
        fr_gait_state['returned_to_neutral'] = True
        logging.info("FR leg returned to neutral forward gait position.")

def resetBackLeftForwardGait():
    global bl_gait_state

    if not bl_gait_state['returned_to_neutral']:
        moveBackLeftToPosition(BL_NEUTRAL_POSITION, speed=16383, acceleration=255, stride_scalar=1) # reset quickly
        bl_gait_state['returned_to_neutral'] = True
        logging.info("BL leg returned to neutral forward gait position.")


########## MOVE FOOT FUNCTIONS ##########

def moveFrontLeftToPosition(pos, speed, acceleration, stride_scalar):
    moveFrontLeftLeg(
        x=pos['x'] * stride_scalar,
        y=pos['y'],
        z=pos['z'],
        speed=speed,
        acceleration=acceleration
    )


def moveBackRightToPosition(pos, speed, acceleration, stride_scalar):
    moveBackRightLeg(
        x=pos['x'] * stride_scalar,
        y=pos['y'],
        z=pos['z'],
        speed=speed,
        acceleration=acceleration
    )

def moveFrontRightToPosition(pos, speed, acceleration, stride_scalar):
    moveFrontRightLeg(
        x=pos['x'] * stride_scalar,
        y=-pos['y'],  # Invert y for front right
        z=pos['z'],
        speed=speed,
        acceleration=acceleration
    )

def moveBackLeftToPosition(pos, speed, acceleration, stride_scalar):
    moveBackLeftLeg(
        x=pos['x'] * stride_scalar,
        y=-pos['y'],  # Invert y for back left
        z=pos['z'],
        speed=speed,
        acceleration=acceleration
    )


########## MOVE LEG FUNCTIONS ##########

def moveFrontLeftLeg(x, y, z, speed, acceleration):

    hip_angle, upper_angle, lower_angle = k.inverse_kinematics(x, y, z)
    hip_neutral_angle, upper_neutral_angle, lower_neutral_angle = 0, 90, 90 # TODO fix why the fuck these angles are 90 and not 45????

    for joint, angle, neutral in zip(['hip', 'upper', 'lower'], [hip_angle, upper_angle, lower_angle], [hip_neutral_angle, upper_neutral_angle, lower_neutral_angle]
    ):

        joint_speed = max(1, speed // 6) if joint in ['upper', 'hip'] else speed
        joint_acceleration = max(1, acceleration // 6) if joint in ['upper', 'hip'] else acceleration

        servo_data = initialize_servos.SERVO_CONFIG['FL'][joint]
        is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
        pwm = initialize_servos.map_angle_to_servo_position(angle, servo_data, neutral, is_inverted)
        initialize_servos.setTarget(servo_data['servo'], pwm, joint_speed, joint_acceleration)


def moveBackRightLeg(x, y, z, speed, acceleration):

    hip_angle, upper_angle, lower_angle = k.inverse_kinematics(x, y, z)
    hip_neutral, upper_neutral, lower_neutral = 0, 45, 45

    for joint, angle, neutral in zip(['hip', 'upper', 'lower'], [hip_angle, upper_angle, lower_angle], [hip_neutral, upper_neutral, lower_neutral]):

        joint_speed = max(1, speed // 6) if joint in ['upper', 'hip'] else speed
        joint_acceleration = max(1, acceleration // 6) if joint in ['upper', 'hip'] else acceleration

        servo_data = initialize_servos.SERVO_CONFIG['BR'][joint]
        is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
        pwm = initialize_servos.map_angle_to_servo_position(angle, servo_data, neutral, is_inverted)
        initialize_servos.setTarget(servo_data['servo'], pwm, joint_speed, joint_acceleration)

def moveFrontRightLeg(x, y, z, speed, acceleration):

    hip_angle, upper_angle, lower_angle = k.inverse_kinematics(x, y, z)
    hip_neutral, upper_neutral, lower_neutral = 0, 45, 45

    for joint, angle, neutral in zip(['hip', 'upper', 'lower'], [hip_angle, upper_angle, lower_angle], [hip_neutral, upper_neutral, lower_neutral]):

        joint_speed = max(1, speed // 6) if joint in ['upper', 'hip'] else speed
        joint_acceleration = max(1, acceleration // 6) if joint in ['upper', 'hip'] else acceleration

        servo_data = initialize_servos.SERVO_CONFIG['FR'][joint]
        is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
        pwm = initialize_servos.map_angle_to_servo_position(angle, servo_data, neutral, is_inverted)
        initialize_servos.setTarget(servo_data['servo'], pwm, joint_speed, joint_acceleration)

def moveBackLeftLeg(x, y, z, speed, acceleration):

    hip_angle, upper_angle, lower_angle = k.inverse_kinematics(x, y, z)
    hip_neutral, upper_neutral, lower_neutral = 0, 45, 45

    for joint, angle, neutral in zip(['hip', 'upper', 'lower'], [hip_angle, upper_angle, lower_angle], [hip_neutral, upper_neutral, lower_neutral]):

        joint_speed = max(1, speed // 6) if joint in ['upper', 'hip'] else speed
        joint_acceleration = max(1, acceleration // 6) if joint in ['upper', 'hip'] else acceleration

        servo_data = initialize_servos.SERVO_CONFIG['BL'][joint]
        is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
        pwm = initialize_servos.map_angle_to_servo_position(angle, servo_data, neutral, is_inverted)
        initialize_servos.setTarget(servo_data['servo'], pwm, joint_speed, joint_acceleration)


########## FOOT TUNING ##########

##### front left tuning #####

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
    moveFrontLeftLeg(x, y, z, speed=16383, acceleration=255, stride_scalar=1)
    logging.info(f"TUNING → FL foot at: x={x:.4f}, y={y:.4f}, z={z:.4f}")

##### back right tuning #####

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
    moveBackRightLeg(x, y, z, speed=16383, acceleration=255, stride_scalar=1)
    logging.info(f"TUNING → BR foot at: x={x:.4f}, y={y:.4f}, z={z:.4f}")

##### front right tuning #####

foot_position_FR = {
    'x': foot_position_BR['x'],  # Same x as BR
    'y': -foot_position_BR['y'],  # Invert y for front right
    'z': foot_position_BR['z']  # Same z as BR
}

def adjustFR_X(forward=True, delta=0.005):
    direction = 1 if forward else -1
    foot_position_FR['x'] += direction * delta
    _applyFootPositionFR()
def adjustFR_Y(left=True, delta=0.005):
    direction = 1 if left else -1
    foot_position_FR['y'] += direction * delta
    _applyFootPositionFR()
def adjustFR_Z(up=True, delta=0.005):
    direction = 1 if up else -1
    foot_position_FR['z'] += direction * delta
    _applyFootPositionFR()
def _applyFootPositionFR():
    x, y, z = foot_position_FR['x'], foot_position_FR['y'], foot_position_FR['z']
    moveFrontRightLeg(x, y, z, speed=16383, acceleration=255, stride_scalar=1)
    logging.info(f"TUNING → FR foot at: x={x:.4f}, y={y:.4f}, z={z:.4f}")

##### back left tuning #####

foot_position_BL = {
    'x': foot_position_FL['x'],  # Same x as FL
    'y': -foot_position_FL['y'],  # Invert y for back left
    'z': foot_position_FL['z']  # Same z as FL
}

def adjustBL_X(forward=True, delta=0.005):
    direction = 1 if forward else -1
    foot_position_BL['x'] += direction * delta
    _applyFootPositionBL()
def adjustBL_Y(left=True, delta=0.005):
    direction = 1 if left else -1
    foot_position_BL['y'] += direction * delta
    _applyFootPositionBL()
def adjustBL_Z(up=True, delta=0.005):
    direction = 1 if up else -1
    foot_position_BL['z'] += direction * delta
    _applyFootPositionBL()
def _applyFootPositionBL():
    x, y, z = foot_position_BL['x'], foot_position_BL['y'], foot_position_BL['z']
    moveBackLeftLeg(x, y, z, speed=16383, acceleration=255, stride_scalar=1)
    logging.info(f"TUNING → BL foot at: x={x:.4f}, y={y:.4f}, z={z:.4f}")