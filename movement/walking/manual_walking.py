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
import logging # import logging for debugging

##### import necessary functions #####

import initialize.initialize_servos as initialize_servos # import servo logic functions
import mathematics.mathematics as mathematics # import all mathematical functions
from movement.positions_config import * # import leg positions config


########## CREATE DEPENDENCIES ##########

##### initialize kinematics #####

k = mathematics.Kinematics(initialize_servos.LINK_CONFIG) # use link lengths to initialize kinematic functions

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





#################################################
############### WALKING FUNCTIONS ###############
#################################################


########## LEG PHASE CONFIG ##########

##### gait states #####

fl_gait_state = {'phase': 'stance', 'last_time': time.time(), 'returned_to_neutral': False}
br_gait_state = {'phase': 'stance', 'last_time': time.time(), 'returned_to_neutral': False}
fr_gait_state = {'phase': 'swing', 'last_time': time.time(), 'returned_to_neutral': False}
bl_gait_state = {'phase': 'swing', 'last_time': time.time(), 'returned_to_neutral': False}


########## GAIT FUNCTIONS ##########

def trotForward(intensity):

    speed, acceleration, stride_scalar = mathematics.interpret_intensity(intensity) # TODO experiment with timing difference

    updateLegGait('FL', {'FORWARD': True}, speed, acceleration, stride_scalar)
    time.sleep(.15)
    updateLegGait('BR', {'FORWARD': True}, speed, acceleration, stride_scalar)
    time.sleep(.15)
    updateLegGait('FR', {'FORWARD': True}, speed, acceleration, stride_scalar)
    time.sleep(.15)
    updateLegGait('BL', {'FORWARD': True}, speed, acceleration, stride_scalar)
    time.sleep(.15)


########## UPDATE LEG GAITS ##########

def updateLegGait(leg_id, state, speed, acceleration, stride_scalar):
    if not state.get('FORWARD', False):
        return  # Skip if not moving forward

    gait_states = {
        'FL': fl_gait_state,
        'FR': fr_gait_state,
        'BL': bl_gait_state,
        'BR': br_gait_state
    }

    swing_positions = {
        'FL': FL_SWING,
        'FR': FR_SWING,
        'BL': BL_SWING,
        'BR': BR_SWING
    }

    stance_positions = {
        'FL': FL_STANCE,
        'FR': FR_STANCE,
        'BL': BL_STANCE,
        'BR': BR_STANCE
    }

    gait_state = gait_states[leg_id]

    if gait_state['phase'] == 'stance':
        moveLegToPosition(leg_id, swing_positions[leg_id], speed, acceleration, stride_scalar, use_bezier=False)
        gait_state['phase'] = 'swing'
    else:
        moveLegToPosition(leg_id, stance_positions[leg_id], speed, acceleration, stride_scalar, use_bezier=False)
        gait_state['phase'] = 'stance'

    gait_state['returned_to_neutral'] = False


########## RESET LEG FORWARD GAIT ##########

def resetLegForwardGait(leg_id):
    gait_states = {
        'FL': fl_gait_state,
        'FR': fr_gait_state,
        'BL': bl_gait_state,
        'BR': br_gait_state
    }

    neutral_positions = {
        'FL': FL_NEUTRAL,
        'FR': FR_NEUTRAL,
        'BL': BL_NEUTRAL,
        'BR': BR_NEUTRAL
    }

    gait_state = gait_states[leg_id]
    neutral_position = neutral_positions[leg_id]

    if not gait_state['returned_to_neutral']:
        moveLegToPosition(
            leg_id,
            pos=neutral_position,
            speed=16383,
            acceleration=255,
            stride_scalar=1,
            use_bezier=False
        )
        gait_state['returned_to_neutral'] = True
        logging.info(f"{leg_id} leg returned to neutral forward gait position.")


########## MOVE FOOT FUNCTION ##########

def moveLegToPosition(leg_id, pos, speed, acceleration, stride_scalar, use_bezier=False):
    if use_bezier:
        p0 = getNeutralPosition(leg_id, stride_scalar)
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

        curve = mathematics.bezier_curve(
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


########## GET NEUTRAL POSITION ##########

def getNeutralPosition(leg_id, stride_scalar):
    config = initialize_servos.SERVO_CONFIG[leg_id]

    # This assumes you’ve stored neutral X/Y/Z positions for each leg somewhere
    # You already have FL_NEUTRAL_POSITION etc., so maybe do:
    NEUTRAL_POSITIONS = {
        'FL': FL_NEUTRAL,
        'FR': FR_NEUTRAL,
        'BL': BL_NEUTRAL,
        'BR': BR_NEUTRAL
    }

    neutral = NEUTRAL_POSITIONS[leg_id]

    return {
        'x': neutral['x'] * stride_scalar,
        'y': neutral['y'] * stride_scalar,
        'z': neutral['z']
    }


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