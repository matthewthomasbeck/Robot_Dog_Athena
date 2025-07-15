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

##### import from config #####

from utilities.config import RL_NOT_CNN, INFERENCE_CONFIG, USE_SIMULATION # import whether to use RL or CNN model

##### import necessary functions #####

# load function/models for gait adjustment and person detection
from utilities.opencv import load_and_compile_model, run_gait_adjustment_standard, run_gait_adjustment_blind, run_person_detection
import utilities.config as config # import configuration data for servos and link lengths
from utilities.mathematics import * # import all mathematical functions

##### import necessary libraries #####

import time # import time for proper leg sequencing
import threading # import threading for thread management

##### import dependencies based on simulation use #####

if USE_SIMULATION:
    import pybullet
    import math
elif not USE_SIMULATION:
    from utilities.servos import map_angle_to_servo_position, set_target  # import servo mapping functions


########## CREATE DEPENDENCIES ##########

##### initialize variables for kinematics and neural processing #####

k = Kinematics(config.LINK_CONFIG) # use link lengths to initialize kinematic functions

##### simulation variables (set by control_logic.py) #####

if USE_SIMULATION:
    ROBOT_ID = None  # Will be set by control_logic.py
    JOINT_MAP = {}   # Will be set by control_logic.py
else:
    ROBOT_ID = None
    JOINT_MAP = {}

##### load and compile models #####

if RL_NOT_CNN:
    # TODO Be aware that multiple models loaded on one NCS2 may be an issue... might be worth benching one of these
    STANDARD_RL_MODEL, STANDARD_INPUT_LAYER, STANDARD_OUTPUT_LAYER = load_and_compile_model(INFERENCE_CONFIG['STANDARD_RL_PATH'])
    BLIND_RL_MODEL, BLIND_INPUT_LAYER, BLIND_OUTPUT_LAYER = load_and_compile_model(INFERENCE_CONFIG['BLIND_RL_PATH'])
elif not RL_NOT_CNN:
    CNN_MODEL, CNN_INPUT_LAYER, CNN_OUTPUT_LAYER = load_and_compile_model(INFERENCE_CONFIG['CNN_PATH'])

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


########## CENTRAL GAIT FUNCTION ##########

def move_direction(commands, frame, intensity, imageless_gait): # function to trot forward

    ##### preprocess commands and intensity #####

    commands = sorted(commands.split('+')) # alphabetize commands so they are uniform
    speed, acceleration = interpret_intensity(intensity) # get speed and acceleration from intensity score

    ##### run inference before moving #####

    logging.debug(
        f"(fundamental_movement.py): Running inference for command(s) {commands} with intensity {intensity}...\n"
    )
    try: # try to run a model
        if not USE_SIMULATION: # if user wants to use real servos...
            if RL_NOT_CNN: # if running gait adjustment (production)...

                ##### run RL model(s) #####

                if not imageless_gait: # if not using imageless gait adjustment...
                    target_coordinates, mid_coordinates = run_gait_adjustment_standard( # run image support RL model
                        STANDARD_RL_MODEL,
                        STANDARD_INPUT_LAYER,
                        STANDARD_OUTPUT_LAYER,
                        commands,
                        frame,
                        speed,
                        acceleration,
                        config.CURRENT_FEET_COORDINATES
                    )
                else: # if using imageless gait adjustment...
                    target_coordinates, mid_coordinates = run_gait_adjustment_blind( # run blind RL model
                        BLIND_RL_MODEL,
                        BLIND_INPUT_LAYER,
                        BLIND_OUTPUT_LAYER,
                        commands,
                        speed,
                        acceleration,
                        config.CURRENT_FEET_COORDINATES
                    )

                ##### move legs and update current position #####

                # move legs and update cur pos
                thread_leg_movement(
                    config.CURRENT_FEET_COORDINATES,
                    mid_coordinates,
                    target_coordinates,
                    speed,
                    acceleration
                )

            else: # if running person detection (testing)...
                run_person_detection(
                    CNN_MODEL,
                    CNN_INPUT_LAYER,
                    CNN_OUTPUT_LAYER,
                    frame,
                    run_inference=False
                )
            logging.info(f"(fundamental_movement.py): Ran AI for command(s) {commands} with intensity {intensity}\n")

        elif USE_SIMULATION: # if running code in simulator...

            ##### rl agent integration point #####
            # gather state for RL agent (define get_simulation_state later if needed)
            state = None # TODO replace with actual state extraction if needed
            if not imageless_gait:  # if not using imageless gait adjustment (image-based agent)...
                target_coordinates, mid_coordinates = get_rl_action_standard(state, commands, intensity, frame)
                logging.warning(
                    "(fundamental_movement.py): Using get_rl_action_standard placeholder. Replace with RL agent output when available."
                )
            elif imageless_gait:  # if using imageless gait adjustment (no image)...
                target_coordinates, mid_coordinates = get_rl_action_blind(state, commands, intensity)
                logging.warning(
                    "(fundamental_movement.py): Using get_rl_action_blind placeholder. Replace with RL agent output when available."
                )

            ##### move legs and update current position #####

            # move legs and update cur pos
            thread_leg_movement(
                config.CURRENT_FEET_COORDINATES,
                mid_coordinates,
                target_coordinates,
                speed,
                acceleration
            )

    except Exception as e: # if either model fails...
        logging.error(f"(fundamental_movement.py): Failed to run AI for command: {e}\n")

    ##### move legs #####

    try: # try to update leg gait
        # TODO somehow move legs after model has been activated
        logging.info(f"(fundamental_movement.py): Executed move_direction() with intensity: {intensity}\n")
        time.sleep(0.1) # wait for legs to reach positions

    except Exception as e: # if gait update fails...
        logging.error(f"(fundamental_movement.py): Failed to gait-cycle legs in move_direction(): {e}\n")


########## THREAD LEG MOVEMENT ##########

def thread_leg_movement(current_coordinates, target_coordinates, mid_coordinates, speed, acceleration):

    leg_threads = []  # create a list to hold threads for each leg
    for leg_id in ['FL', 'FR', 'BL', 'BR']:  # loop through each leg and create a thread to move
        t = threading.Thread(
            target=swing_leg,
            args=(
                leg_id,
                current_coordinates[leg_id],
                mid_coordinates[leg_id],
                target_coordinates[leg_id],
                speed,
                acceleration
            )
        )
        leg_threads.append(t)
        t.start()
    for t in leg_threads:  # wait for all legs to finish
        t.join()
    for leg_id in ['FL', 'FR', 'BL', 'BR']:  # update current foot positions
        config.CURRENT_FEET_COORDINATES[leg_id] = target_coordinates[leg_id].copy()


########## SET LEG PHASE ##########

def swing_leg(leg_id, current_coordinate, mid_coordinate, target_coordinate, speed, acceleration):

    try:
        move_foot_to_pos(leg_id, current_coordinate, mid_coordinate, speed, acceleration, use_bezier=False)
        move_foot_to_pos(leg_id, mid_coordinate, target_coordinate, speed, acceleration, use_bezier=False)

    except Exception as e:
        logging.error(f"(fundamental_movement.py): Failed to swing leg: {e}\n")


########## MOVE FOOT FUNCTION ##########

def move_foot_to_pos(leg_id, start_coordinate, end_coordinate, speed, acceleration, use_bezier):

    if use_bezier: # if user wants to use bezier curves for foot movement...
        p0 = {'x': start_coordinate['x'], 'y': start_coordinate['y'], 'z': start_coordinate['z']}
        p2 = {'x': end_coordinate['x'], 'y': end_coordinate['y'], 'z': end_coordinate['z']}
        p1 = {'x': (p0['x'] + p2['x']) / 2, 'y': (p0['y'] + p2['y']) / 2, 'z': max(p0['z'], p2['z']) + 0.025}
        curve = bezier_curve(
            p0=(p0['x'], p0['y'], p0['z']),
            p1=(p1['x'], p1['y'], p1['z']),
            p2=(p2['x'], p2['y'], p2['z']),
            steps=4
        )
        for x, y, z in curve: # iterate through the bezier curve points
            move_foot(leg_id, x, y, z, speed, acceleration)

    else: # if user does not want to use bezier curves for foot movement...
        move_foot(
            leg_id,
            x=end_coordinate['x'],
            y=end_coordinate['y'],
            z=end_coordinate['z'],
            speed=speed,
            acceleration=acceleration
        )

def move_foot_to_pos_OLD(leg_id, pos, speed, acceleration, use_bezier):

    if use_bezier: # if user wants to use bezier curves for foot movement...
        p0 = get_neutral_positions(leg_id) # get the neutral position of the leg (old hand-tuned method for testing)
        p2 = {'x': pos['x'], 'y': pos['y'], 'z': pos['z']}
        p1 = {'x': (p0['x'] + p2['x']) / 2, 'y': (p0['y'] + p2['y']) / 2, 'z': max(p0['z'], p2['z']) + 0.025}
        curve = bezier_curve(
            p0=(p0['x'], p0['y'], p0['z']),
            p1=(p1['x'], p1['y'], p1['z']),
            p2=(p2['x'], p2['y'], p2['z']),
            steps=4
        )
        for x, y, z in curve: # iterate through the bezier curve points
            move_foot(leg_id, x, y, z, speed, acceleration)

    else: # if user does not want to use bezier curves for foot movement...
        move_foot(leg_id, x=pos['x'], y=pos['y'], z=pos['z'], speed=speed, acceleration=acceleration)


########## MOVE LEG FUNCTION ##########

def move_foot(leg_id, x, y, z, speed, acceleration):

    ##### collect angles for each joint of leg #####

    hip_angle, upper_angle, lower_angle = k.inverse_kinematics(x, y, z)
    hip_neutral, upper_neutral, lower_neutral = 0, 45, 45

    ##### move each joint of the leg to desired angle #####

    for joint, angle, neutral in zip(
            ['hip', 'upper', 'lower'],
            [hip_angle, upper_angle, lower_angle],
            [hip_neutral, upper_neutral, lower_neutral]
    ):

        joint_speed = max(1, speed // 3) if joint in ['upper', 'hip'] else speed
        joint_acceleration = max(1, acceleration // 3) if joint in ['upper', 'hip'] else acceleration
        # TODO comment this if I ever need upper servos to move more quickly
        #joint_speed = speed
        #joint_acceleration = acceleration

        if not USE_SIMULATION: # if user wants to use real servos...
            servo_data = config.SERVO_CONFIG[leg_id][joint]
            is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
            pwm = map_angle_to_servo_position(angle, servo_data, neutral, is_inverted)
            set_target(servo_data['servo'], pwm, joint_speed, joint_acceleration)

        elif USE_SIMULATION: # if user wants to control simulated robot...
            angle_rad = math.radians(angle) # convert angles to radians for simulation
            joint_key = (leg_id, joint) # get joint key for simulation
            if joint_key in JOINT_MAP: # if joint exists...
                joint_index = JOINT_MAP[joint_key] # get joint index from joint map
                pybullet.setJointMotorControl2( # set joint position
                    bodyUniqueId=ROBOT_ID,
                    jointIndex=joint_index,
                    controlMode=pybullet.POSITION_CONTROL,
                    targetPosition=angle_rad,
                    force=4.41
                )
            else:
                logging.warning(f"(fundamental_movement.py): Joint {joint_key} not found in PyBullet joint map")


##### GET NEUTRAL POSITIONS #####

def get_neutral_positions(leg_id): # function to get the neutral position of a leg, used in manual movement

    NEUTRAL_POSITIONS = {
        'FL': config.FL_NEUTRAL,
        'FR': config.FR_NEUTRAL,
        'BL': config.BL_NEUTRAL,
        'BR': config.BR_NEUTRAL
    }
    neutral = NEUTRAL_POSITIONS[leg_id]
    return {'x': neutral['x'], 'y': neutral['y'], 'z': neutral['z'] }


########## INITIALIZE SIMULATION ##########

def set_simulation_variables(robot_id, joint_map):

    global ROBOT_ID, JOINT_MAP
    ROBOT_ID = robot_id
    JOINT_MAP = joint_map


########## STANDARD RL MODEL ACTION SPACE ##########

def get_rl_action_standard(state, commands, intensity, frame): # this is a placeholder function for the standard model
    """
    Placeholder for RL agent's policy WITH image input (standard gait adjustment).
    Args:
        state: The current state of the robot/simulation (to be defined).
        commands: The movement commands.
        intensity: The movement intensity.
        frame: The current image frame (for vision-based agent).
    Returns:
        target_positions: dict of target foot positions for each leg.
        mid_positions: dict of mid foot positions for each leg.
    TODO: Replace this with a call to your RL agent's policy/model (with image input).
    """
    # For now, just return the current positions as both mid and target
    target_positions = {leg: config.CURRENT_FEET_COORDINATES[leg].copy() for leg in ['FL', 'FR', 'BL', 'BR']}
    mid_positions = {leg: config.CURRENT_FEET_COORDINATES[leg].copy() for leg in ['FL', 'FR', 'BL', 'BR']}
    return target_positions, mid_positions


########## BLIND RL MODEL ACTION SPACE ##########

def get_rl_action_blind(state, commands, intensity): # this is a placeholder function for the blind model
    """
    Placeholder for RL agent's policy WITHOUT image input (imageless gait adjustment).
    Args:
        state: The current state of the robot/simulation (to be defined).
        commands: The movement commands.
        intensity: The movement intensity.
    Returns:
        target_positions: dict of target foot positions for each leg.
        mid_positions: dict of mid foot positions for each leg.
    TODO: Replace this with a call to your RL agent's policy/model (no image input).
    """
    # For now, just return the current positions as both mid and target
    target_positions = {leg: config.CURRENT_FEET_COORDINATES[leg].copy() for leg in ['FL', 'FR', 'BL', 'BR']}
    mid_positions = {leg: config.CURRENT_FEET_COORDINATES[leg].copy() for leg in ['FL', 'FR', 'BL', 'BR']}
    return target_positions, mid_positions


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
    move_foot('FL', x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → FL foot at: 'x': {x:.4f}, 'y': {y:.4f}, 'z': {z:.4f}")
def _applyFootPositionBR():
    x, y, z = config.BR_TUNE['x'], config.BR_TUNE['y'], config.BR_TUNE['z']
    move_foot('BR', x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → BR foot at: 'x': {x:.4f}, 'y': {y:.4f}, 'z': {z:.4f}")
def _applyFootPositionFR():
    x, y, z = config.FR_TUNE['x'], config.FR_TUNE['y'], config.FR_TUNE['z']
    move_foot('FR', x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → FR foot at: 'x': {x:.4f}, 'y': {y:.4f}, 'z': {z:.4f}")
def _applyFootPositionBL():
    x, y, z = config.BL_TUNE['x'], config.BL_TUNE['y'], config.BL_TUNE['z']
    move_foot('BL', x, y, z, speed=16383, acceleration=255)
    logging.info(f"TUNING → BL foot at: 'x': {x:.4f}, 'y': {y:.4f}, 'z': {z:.4f}")
