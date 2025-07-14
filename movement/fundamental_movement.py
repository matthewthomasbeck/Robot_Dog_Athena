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

##### import necessary functions #####

from utilities.servos import map_angle_to_servo_position, set_target # import servo mapping functions
import utilities.config as config # import configuration data for servos and link lengths
from utilities.mathematics import * # import all mathematical functions
from utilities.opencv import load_and_compile_model, run_inference  # load inference functions

##### import necessary libraries #####

import time # import time for proper leg sequencing


########## CREATE DEPENDENCIES ##########

##### initialize kinematics #####

k = Kinematics(config.LINK_CONFIG) # use link lengths to initialize kinematic functions

##### load reinforcement learning model #####

COMPILED_MODEL, INPUT_LAYER, OUTPUT_LAYER = load_and_compile_model()  # load and compile model

##### define servos #####

upper_leg_servos = { # define upper leg servos

    "FL": config.SERVO_CONFIG['FL']['upper'],  # front left
    "FR": config.SERVO_CONFIG['FR']['upper'],  # front right
    "BL": config.SERVO_CONFIG['BL']['upper'],  # back left
    "BR": config.SERVO_CONFIG['BR']['upper'],  # back right
}

lower_leg_servos = { # define lower leg servos

    "FL": config.SERVO_CONFIG['FL']['lower'],  # front left
    "FR": config.SERVO_CONFIG['FR']['lower'],  # front right
    "BL": config.SERVO_CONFIG['BL']['lower'],  # back left
    "BR": config.SERVO_CONFIG['BR']['lower'],  # back right
}





##############################################################
############### FUNDAMENTAL MOVEMENT FUNCTIONS ###############
##############################################################


########## GAIT FUNCTION ##########

def move_direction(direction, frame, intensity): # function to trot forward

    ##### set variables #####

    gait_states = {
        'FL': config.FL_GAIT_STATE, 'FR': config.FR_GAIT_STATE,
        'BL': config.BL_GAIT_STATE, 'BR': config.BR_GAIT_STATE
    }

    ##### get intensity #####

    try: # try to calculate intensity of the trot

        speed, acceleration = interpret_intensity(intensity)

    except Exception as e: # if interpretation fails...

        logging.error(f"(fundamental_movement.py): Failed to interpret intensity in move_direction(): {e}\n")
        return

    ##### run inference before moving ###

    logging.debug(
        f"(fundamental_movement.py): Running inference for command {direction} with intensity {intensity}...\n"
    )

    try:
        run_inference(
            COMPILED_MODEL,
            INPUT_LAYER,
            OUTPUT_LAYER,
            frame,
            run_inference=False
        )
        logging.info(f"(fundamental_movement.py): Ran inference for command {direction} with intensity {intensity}\n")
    except Exception as e:
        logging.error(f"(fundamental_movement.py): Failed to run inference for command: {e}\n")

    ##### move legs forward #####

    try: # try to update leg gait

        if gait_states['FL']['phase'] == 'swing': # if front left about to push...
            set_leg_phase_NEW('BL', {'FORWARD': True}, frame, speed, acceleration)
            set_leg_phase_NEW('FR', {'FORWARD': True}, frame, speed, acceleration)
            time.sleep(0.5)
            set_leg_phase_NEW('BR', {'FORWARD': True}, frame, speed, acceleration)
            set_leg_phase_NEW('FL', {'FORWARD': True}, frame, speed, acceleration)

        elif gait_states['FR']['phase'] == 'swing': # if front right about to push...
            set_leg_phase_NEW('BR', {'FORWARD': True}, frame, speed, acceleration)
            set_leg_phase_NEW('FL', {'FORWARD': True}, frame, speed, acceleration)
            time.sleep(0.5)
            set_leg_phase_NEW('BL', {'FORWARD': True}, frame, speed, acceleration)
            set_leg_phase_NEW('FR', {'FORWARD': True}, frame, speed, acceleration)

        logging.info(f"(fundamental_movement.py): Executed move_direction() with intensity: {intensity}\n")

        time.sleep(0.1) # wait for legs to reach positions

    except Exception as e: # if gait update fails...

        logging.error(f"(fundamental_movement.py): Failed to gait-cycle legs in move_direction(): {e}\n")


########## SET LEG PHASE ##########

def set_leg_phase_NEW(leg_id, state, frame, speed, acceleration):

    if not state.get('FORWARD', False):
        return

    gait_states = {
        'FL': config.FL_GAIT_STATE, 'FR': config.FR_GAIT_STATE,
        'BL': config.BL_GAIT_STATE, 'BR': config.BR_GAIT_STATE
    }
    swing_positions = {
        'FL': config.FL_SWING, 'FR': config.FR_SWING,
        'BL': config.BL_SWING, 'BR': config.BR_SWING
    }
    touchdown_positions = {
        'FL': config.FL_TOUCHDOWN, 'FR': config.FR_TOUCHDOWN,
        'BL': config.BL_TOUCHDOWN, 'BR': config.BR_TOUCHDOWN
    }
    midstance_positions = {
        'FL': config.FL_MIDSTANCE, 'FR': config.FR_MIDSTANCE,
        'BL': config.BL_MIDSTANCE, 'BR': config.BR_MIDSTANCE
    }
    neutral_positions = {
        'FL': config.FL_NEUTRAL, 'FR': config.FR_NEUTRAL,
        'BL': config.BL_NEUTRAL, 'BR': config.BR_NEUTRAL
    }
    tippytoes_positions = {
        'FL': config.FL_TIPPYTOES, 'FR': config.FR_TIPPYTOES,
        'BL': config.BL_TIPPYTOES, 'BR': config.BR_TIPPYTOES
    }
    stance_positions = {
        'FL': config.FL_STANCE, 'FR': config.FR_STANCE,
        'BL': config.BL_STANCE, 'BR': config.BR_STANCE
    }

    gait_state = gait_states[leg_id]
    phase = gait_state['phase']

    try:

        if phase == 'stance':

            # Begin new cycle: lift foot into swing
            move_foot_to_pos(leg_id, swing_positions[leg_id], speed, acceleration, use_bezier=True)
            time.sleep(0.2)
            move_foot_to_pos(leg_id, touchdown_positions[leg_id], speed, acceleration, use_bezier=True)
            gait_state['phase'] = 'swing'

        elif phase == 'swing':
            # Final phase: move foot into full stance
            move_foot_to_pos(leg_id, tippytoes_positions[leg_id], speed, acceleration, use_bezier=False)
            time.sleep(0.2)
            move_foot_to_pos(leg_id, stance_positions[leg_id], speed, acceleration, use_bezier=False)
            gait_state['phase'] = 'stance'

        gait_state['returned_to_neutral'] = False

    except Exception as e:
        logging.error(f"(fundamental_movement.py): Failed gait move for {leg_id} in phase {phase}: {e}\n")


########## MOVE FOOT FUNCTION ##########

def move_foot_to_pos(leg_id, pos, speed, acceleration, use_bezier=False):
    if use_bezier:
        p0 = get_neutral_positions(leg_id)
        p2 = {
            'x': pos['x'],
            'y': pos['y'],
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
            move_leg(leg_id, x, y, z, speed, acceleration)
            # TODO: test adding time.sleep here

    else:
        move_leg(
            leg_id,
            x=pos['x'],
            y=pos['y'],
            z=pos['z'],
            speed=speed,
            acceleration=acceleration
        )


########## MOVE LEG FUNCTION ##########

def move_leg(leg_id, x, y, z, speed, acceleration):
    hip_angle, upper_angle, lower_angle = k.inverse_kinematics(x, y, z)
    hip_neutral, upper_neutral, lower_neutral = 0, 45, 45

    for joint, angle, neutral in zip(['hip', 'upper', 'lower'],
                                     [hip_angle, upper_angle, lower_angle],
                                     [hip_neutral, upper_neutral, lower_neutral]):

        joint_speed = max(1, speed // 2) if joint in ['upper', 'hip'] else speed
        joint_acceleration = max(1, acceleration // 2) if joint in ['upper', 'hip'] else acceleration

        # TODO comment this if I ever need lower servos to move more quickly
        #joint_speed = speed
        #joint_acceleration = acceleration

        servo_data = config.SERVO_CONFIG[leg_id][joint]
        is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
        pwm = map_angle_to_servo_position(angle, servo_data, neutral, is_inverted)
        set_target(servo_data['servo'], pwm, joint_speed, joint_acceleration)


##### GET NEUTRAL POSITIONS #####

def get_neutral_positions(leg_id):

    NEUTRAL_POSITIONS = {
        'FL': config.FL_NEUTRAL,
        'FR': config.FR_NEUTRAL,
        'BL': config.BL_NEUTRAL,
        'BR': config.BR_NEUTRAL
    }

    neutral = NEUTRAL_POSITIONS[leg_id]

    return {
        'x': neutral['x'],
        'y': neutral['y'],
        'z': neutral['z']
    }


########## FOOT TUNING ##########

def adjustFL_X(forward=True, delta=0.005):
    direction = 1 if forward else -1
    config.FL_TUNE['x'] += direction * delta
    _applyFootPosition()
def adjustBR_X(forward=True, delta=0.005):
    direction = 1 if forward else -1
    config.BR_TUNE['x'] += direction * delta
    _applyFootPositionBR()
def adjustFR_X(forward=True, delta=0.005):
    direction = 1 if forward else -1
    config.FR_TUNE['x'] += direction * delta
    _applyFootPositionFR()
def adjustBL_X(forward=True, delta=0.005):
    direction = 1 if forward else -1
    config.BL_TUNE['x'] += direction * delta
    _applyFootPositionBL()

def adjustFL_Y(left=True, delta=0.005):
    direction = 1 if left else -1
    config.FL_TUNE['y'] += direction * delta
    _applyFootPosition()
def adjustBR_Y(left=True, delta=0.005):
    direction = 1 if left else -1
    config.BR_TUNE['y'] += direction * delta
    _applyFootPositionBR()
def adjustFR_Y(left=True, delta=0.005):
    direction = 1 if left else -1
    config.FR_TUNE['y'] += direction * delta
    _applyFootPositionFR()
def adjustBL_Y(left=True, delta=0.005):
    direction = 1 if left else -1
    config.BL_TUNE['y'] += direction * delta
    _applyFootPositionBL()

def adjustFL_Z(up=True, delta=0.005):
    direction = 1 if up else -1
    config.FL_TUNE['z'] += direction * delta
    _applyFootPosition()
def adjustBR_Z(up=True, delta=0.005):
    direction = 1 if up else -1
    config.BR_TUNE['z'] += direction * delta
    _applyFootPositionBR()
def adjustFR_Z(up=True, delta=0.005):
    direction = 1 if up else -1
    config.FR_TUNE['z'] += direction * delta
    _applyFootPositionFR()
def adjustBL_Z(up=True, delta=0.005):
    direction = 1 if up else -1
    config.BL_TUNE['z'] += direction * delta
    _applyFootPositionBL()

def _applyFootPosition():
    x, y, z = config.FL_TUNE['x'], config.FL_TUNE['y'], config.FL_TUNE['z']
    move_leg('FL', x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → FL foot at: 'x': {x:.4f}, 'y': {y:.4f}, 'z': {z:.4f}")
def _applyFootPositionBR():
    x, y, z = config.BR_TUNE['x'], config.BR_TUNE['y'], config.BR_TUNE['z']
    move_leg('BR', x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → BR foot at: 'x': {x:.4f}, 'y': {y:.4f}, 'z': {z:.4f}")
def _applyFootPositionFR():
    x, y, z = config.FR_TUNE['x'], config.FR_TUNE['y'], config.FR_TUNE['z']
    move_leg('FR', x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → FR foot at: 'x': {x:.4f}, 'y': {y:.4f}, 'z': {z:.4f}")
def _applyFootPositionBL():
    x, y, z = config.BL_TUNE['x'], config.BL_TUNE['y'], config.BL_TUNE['z']
    move_leg('BL', x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → BL foot at: 'x': {x:.4f}, 'y': {y:.4f}, 'z': {z:.4f}")