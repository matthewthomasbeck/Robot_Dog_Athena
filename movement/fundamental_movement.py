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

#from utilities.config import RL_NOT_CNN, INFERENCE_CONFIG, USE_SIMULATION, USE_ISAAC_SIM, ISAAC_SIM_APP

##### import necessary functions #####

import utilities.config as config # import configuration data for servos and link lengths
from utilities.mathematics import * # import all mathematical functions

##### import necessary libraries #####

import time # import time for proper leg sequencing
import threading # import threading for thread management
import math # import math for angle conversions

##### import dependencies based on simulation use #####

if config.USE_SIMULATION:
    if config.USE_ISAAC_SIM:
        from isaacsim.core.api.controllers.articulation_controller import ArticulationController
        from isaacsim.core.utils.types import ArticulationAction
        import numpy
        import queue

        ARTICULATION_CONTROLLER = ArticulationController() # create new articulation controller instance
        ARTICULATION_CONTROLLER.initialize(config.ISAAC_ROBOT) # initialize the controller
        # Note: config.JOINT_INDEX_MAP is set in control_logic.py set_isaac_dependencies()
        
        # Queue system for Isaac Sim to avoid PhysX threading violations
        ISAAC_MOVEMENT_QUEUE = queue.Queue()
        ISAAC_CALIBRATION_COMPLETE = threading.Event()  # Signal for calibration completion

    else:
        import pybullet
        import math

elif not config.USE_SIMULATION:
    # load function/models for gait adjustment and person detection
    from utilities.opencv import load_and_compile_model, run_gait_adjustment_standard, run_gait_adjustment_blind, \
        run_person_detection
    from utilities.servos import map_angle_to_servo_position, set_target  # import servo mapping functions
    if config.RL_NOT_CNN:
        # TODO Be aware that multiple models loaded on one NCS2 may be an issue... might be worth benching one of these
        STANDARD_RL_MODEL, STANDARD_INPUT_LAYER, STANDARD_OUTPUT_LAYER = load_and_compile_model(
            config.INFERENCE_CONFIG['STANDARD_RL_PATH'])
        BLIND_RL_MODEL, BLIND_INPUT_LAYER, BLIND_OUTPUT_LAYER = load_and_compile_model(
            config.INFERENCE_CONFIG['BLIND_RL_PATH'])
    elif not config.RL_NOT_CNN:
        CNN_MODEL, CNN_INPUT_LAYER, CNN_OUTPUT_LAYER = load_and_compile_model(config.INFERENCE_CONFIG['CNN_PATH'])


########## CREATE DEPENDENCIES ##########

##### initialize variables for kinematics and neural processing #####

k = Kinematics(config.LINK_CONFIG) # use link lengths to initialize kinematic functions

##### simulation variables (set by control_logic.py) #####

if config.USE_SIMULATION:
    ROBOT_ID = None  # will be set by control_logic.py
    JOINT_MAP = {}   # will be set by control_logic.py
else:
    ROBOT_ID = None
    JOINT_MAP = {}

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

    ##### BASIC TESTING - MOVE ALL JOINTS FORWARD/BACKWARD #####

    if 'w' in commands:
        # Move all joints to FULL_FRONT
        logging.info("(fundamental_movement.py): Moving all joints FORWARD for testing...\n")
        if config.USE_SIMULATION and config.USE_ISAAC_SIM:
            # Isaac Sim: Use direct joint control
            _move_all_joints_forward_isaac()
        else:
            # Physical: Use PWM control
            _move_all_joints_forward_physical(speed, acceleration)
        return

    elif 's' in commands:
        # Move all joints to FULL_BACK
        logging.info("(fundamental_movement.py): Moving all joints BACKWARD for testing...\n")
        if config.USE_SIMULATION and config.USE_ISAAC_SIM:
            # Isaac Sim: Use direct joint control
            _move_all_joints_backward_isaac()
        else:
            # Physical: Use PWM control
            _move_all_joints_backward_physical(speed, acceleration)
        return

    ##### run inference before moving #####

    logging.debug(
        f"(fundamental_movement.py): Running inference for command(s) {commands} with intensity {intensity}...\n"
    )
    try: # try to run a model
        if not config.USE_SIMULATION: # if user wants to use real servos...
            if config.RL_NOT_CNN: # if running gait adjustment (production)...

                ##### run RL model(s) #####

                if not imageless_gait: # if not using imageless gait adjustment...
                    target_coordinates, mid_coordinates, movement_rates = run_gait_adjustment_standard( # run standard
                        STANDARD_RL_MODEL,
                        STANDARD_INPUT_LAYER,
                        STANDARD_OUTPUT_LAYER,
                        commands,
                        frame,
                        intensity,
                        config.CURRENT_FEET_COORDINATES
                    )
                else: # if using imageless gait adjustment...
                    target_coordinates, mid_coordinates, movement_rates = run_gait_adjustment_blind( # run blind
                        BLIND_RL_MODEL,
                        BLIND_INPUT_LAYER,
                        BLIND_OUTPUT_LAYER,
                        commands,
                        intensity,
                        config.CURRENT_FEET_COORDINATES
                    )

                ##### move legs and update current position #####

                # move legs and update cur pos
                thread_leg_movement(
                    config.CURRENT_FEET_COORDINATES,
                    mid_coordinates,
                    target_coordinates,
                    movement_rates
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

        elif config.USE_SIMULATION: # if running code in simulator...

            if config.USE_ISAAC_SIM:

                ##### rl agent integration point #####
                # gather state for RL agent (define get_simulation_state later if needed)
                state = None # TODO replace with actual state extraction if needed
                if not imageless_gait:  # if not using imageless gait adjustment (image-based agent)...
                    target_coordinates, mid_coordinates, movement_rates = get_rl_action_standard(
                        state,
                        commands,
                        intensity,
                        frame
                    )
                    logging.warning(
                        "(fundamental_movement.py): Using get_rl_action_standard placeholder. Replace with RL agent output when available."
                    )
                elif imageless_gait:  # if using imageless gait adjustment (no image)...
                    target_coordinates, mid_coordinates, movement_rates = get_rl_action_blind(
                        state,
                        commands,
                        intensity
                    )
                    logging.warning(
                        "(fundamental_movement.py): Using get_rl_action_blind placeholder. Replace with RL agent output when available."
                    )

                ##### queue leg movement for Isaac Sim (avoid PhysX threading violations) #####

                # Queue the movement data instead of threading
                movement_data = {
                    'current_coordinates': config.CURRENT_FEET_COORDINATES.copy(),
                    'mid_coordinates': mid_coordinates,
                    'target_coordinates': target_coordinates,
                    'movement_rates': movement_rates
                }
                ISAAC_MOVEMENT_QUEUE.put(movement_data)
                logging.debug(f"(fundamental_movement.py): Queued movement data for Isaac Sim: {commands}\n")

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

def thread_leg_movement(current_coordinates, target_coordinates, mid_coordinates, movement_rates):

    leg_threads = []  # create a list to hold threads for each leg
    for leg_id in ['FL', 'FR', 'BL', 'BR']:  # loop through each leg and create a thread to move
        t = threading.Thread(
            target=swing_leg,
            args=(
                leg_id,
                current_coordinates[leg_id],
                mid_coordinates[leg_id],
                target_coordinates[leg_id],
                movement_rates[leg_id]
            )
        )
        leg_threads.append(t)
        t.start()
    for t in leg_threads:  # wait for all legs to finish
        t.join()
    for leg_id in ['FL', 'FR', 'BL', 'BR']:  # update current foot positions
        config.CURRENT_FEET_COORDINATES[leg_id] = target_coordinates[leg_id].copy()


########## SET LEG PHASE ##########

def swing_leg(leg_id, current_coordinate, mid_coordinate, target_coordinate, movement_rate):

    try:
        speed = movement_rate.get('speed', 16383)  # default to 16383 (max) if not provided
        acceleration = movement_rate.get('acceleration', 255)  # default to 255 (max) if not provided
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

    if config.USE_SIMULATION and config.USE_ISAAC_SIM:
        # For Isaac Sim, use the queue system to avoid PhysX threading violations
        logging.debug(f"(fundamental_movement.py): Using Isaac Sim queue for move_foot_to_pos_OLD: {leg_id} to {pos}\n")
        
        # Create movement data for Isaac Sim
        movement_data = {
            'current_coordinates': {leg_id: config.CURRENT_FEET_COORDINATES[leg_id].copy()},
            'mid_coordinates': {leg_id: pos.copy()},  # Direct movement, no mid point
            'target_coordinates': {leg_id: pos.copy()},
            'movement_rates': {leg_id: {'speed': speed, 'acceleration': acceleration}}
        }
        
        # Queue the movement data
        ISAAC_MOVEMENT_QUEUE.put(movement_data)
        return

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


########## MOVE FOOT FUNCTION ##########

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

        if not config.USE_SIMULATION and not config.USE_ISAAC_SIM: # if user wants to use real servos...
            servo_data = config.SERVO_CONFIG[leg_id][joint]
            is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
            pwm = map_angle_to_servo_position(angle, servo_data, neutral, is_inverted)
            set_target(servo_data['servo'], pwm, joint_speed, joint_acceleration)

        elif config.USE_SIMULATION and not config.USE_ISAAC_SIM: # if user wants to control simulated robot with PyBullet...
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
                logging.warning(f"(fundamental_movement.py): Joint {joint_key} not found in PyBullet joint map\n")


        elif config.USE_SIMULATION and config.USE_ISAAC_SIM:
            servo_data = config.SERVO_CONFIG[leg_id][joint]
            is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
            adjusted_angle = -angle if is_inverted else angle
            angle_rad = math.radians(adjusted_angle)
            joint_name = f"{leg_id}_{joint}"

            try:
                joint_index = config.JOINT_INDEX_MAP.get(joint_name)
                if joint_index is None:
                    logging.error(f"(fundamental_movement.py): Joint {joint_name} not found in JOINT_INDEX_MAP: {config.JOINT_INDEX_MAP}\n")
                    return
                
                joint_count = len(config.ISAAC_ROBOT.dof_names)
                joint_positions = numpy.zeros(joint_count)
                joint_velocities = numpy.zeros(joint_count)
                joint_accelerations = numpy.zeros(joint_count)
                joint_positions[joint_index] = angle_rad
                joint_velocities[joint_index] = joint_speed
                joint_accelerations[joint_index] = joint_acceleration  # Optional, not used
                action = ArticulationAction(
                    joint_positions=joint_positions,
                    joint_velocities=joint_velocities
                    # joint_accelerations=joint_accelerations  # not supported
                )
                ARTICULATION_CONTROLLER.apply_action(action)
                logging.debug(
                    f"(fundamental_movement.py) Successfully moved joint {joint_name} to {angle_rad:.3f} rad (inverted: {is_inverted}).\n"
                )

            except Exception as e:
                logging.error(
                    f"(fundamental_movement.py): Failed to apply Isaac Sim joint action for {joint_name}: {e}\n"
                )


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


########## ISAAC SIM QUEUE-BASED MOVEMENT FUNCTIONS ##########

def process_isaac_movement_queue():
    """
    Process queued movements for Isaac Sim in the main thread to avoid PhysX violations.
    This function should be called from the main loop after each simulation step.
    """
    if not config.USE_SIMULATION or not config.USE_ISAAC_SIM:
        return
    
    # Process any new movement data
    while not ISAAC_MOVEMENT_QUEUE.empty():
        try:
            movement_data = ISAAC_MOVEMENT_QUEUE.get_nowait()
            
            # Check if this is calibration data
            if isinstance(movement_data, dict) and movement_data.get('type') == 'calibration':
                # Apply the joint position directly
                joint_name = movement_data['joint_name']
                angle_rad = movement_data['angle_rad']
                velocity = movement_data.get('velocity', 0.5)  # Default to 0.5 if not specified
                _apply_single_joint_position_isaac(joint_name, angle_rad, velocity)
                
                # Signal that this calibration movement is complete
                ISAAC_CALIBRATION_COMPLETE.set()
                
        except queue.Empty:
            break


########## JOINT CALIBRATION ##########

def calibrate_joints_isaac():
    """
    Calibrate all joints by moving each joint through its full range of motion.
    This function runs indefinitely and cycles through each joint one at a time.
    Only for Isaac Sim - uses queue system to avoid PhysX threading violations.
    """
    if not config.USE_SIMULATION or not config.USE_ISAAC_SIM:
        logging.error("(fundamental_movement.py): calibrate_joints_isaac() only works with Isaac Sim\n")
        return
    
    # Define joint order for calibration
    joint_order = [
        ('FL', 'hip'),
        ('FL', 'upper'),
        ('FL', 'lower'),
        ('FR', 'hip'),
        ('FR', 'upper'),
        ('FR', 'lower'),
        ('BL', 'hip'),
        ('BL', 'upper'),
        ('BL', 'lower'),
        ('BR', 'hip'),
        ('BR', 'upper'),
        ('BR', 'lower')
    ]
    
    # Calibration parameters
    step_time = 0.1  # seconds between position updates
    steps_per_movement = 10  # number of steps to complete one movement
    
    logging.info("(fundamental_movement.py): Starting joint calibration for Isaac Sim...\n")
    
    joint_index = 0
    while True:
        try:
            # Get current joint to calibrate
            leg_id, joint_name = joint_order[joint_index]
            joint_full_name = f"{leg_id}_{joint_name}"
            
            # Get joint configuration
            servo_data = config.SERVO_CONFIG[leg_id][joint_name]
            is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
            
            # Convert angles to radians - use the actual angle values, not PWM values
            full_back_rad = servo_data['FULL_BACK_ANGLE']  # Already in radians
            full_front_rad = servo_data['FULL_FRONT_ANGLE']  # Already in radians
            neutral_rad = 0.0  # Neutral position at 0 radians
            
            # Apply inversion if needed
            if is_inverted:
                full_back_rad, full_front_rad = full_front_rad, full_back_rad
            
            logging.info(f"(fundamental_movement.py): Calibrating {joint_full_name} - Back: {math.degrees(full_back_rad):.1f}°, Front: {math.degrees(full_front_rad):.1f}°, Neutral: {math.degrees(neutral_rad):.1f}°\n")
            
            # Move through the sequence: NEUTRAL -> BACK -> FRONT -> NEUTRAL
            movements = [
                (neutral_rad, full_back_rad),      # Neutral to Back
                (full_back_rad, full_front_rad),   # Back to Front
                (full_front_rad, neutral_rad)      # Front to Neutral
            ]
            
            for start_pos, end_pos in movements:
                # Move through intermediate positions
                for step in range(steps_per_movement):
                    # Clear the completion signal
                    ISAAC_CALIBRATION_COMPLETE.clear()
                    
                    # Linear interpolation between start and end positions
                    progress = step / steps_per_movement
                    current_pos = start_pos + (end_pos - start_pos) * progress
                    
                    # Queue the joint position with slow velocity for calibration
                    _queue_single_joint_position_isaac(joint_full_name, current_pos, velocity=0.5)
                    
                    # Wait for this position to be processed before continuing
                    ISAAC_CALIBRATION_COMPLETE.wait(timeout=1.0)
                    
                    # Wait before next step
                    time.sleep(step_time)
            
            # Wait 3 seconds before moving to next joint
            time.sleep(3.0)
            
            # Move to next joint
            joint_index = (joint_index + 1) % len(joint_order)
            
        except Exception as e:
            logging.error(f"(fundamental_movement.py): Error in joint calibration: {e}\n")
            time.sleep(1)  # Wait before retrying


def _queue_single_joint_position_isaac(joint_name, angle_rad, velocity=0.5):
    """
    Queue a single joint position for Isaac Sim calibration.
    Args:
        joint_name: Full joint name (e.g., 'FL_hip')
        angle_rad: Target angle in radians
        velocity: Joint velocity in radians/second (default: 0.5)
    """
    try:
        # Create calibration movement data
        calibration_data = {
            'type': 'calibration',
            'joint_name': joint_name,
            'angle_rad': angle_rad,
            'velocity': velocity
        }
        
        # Queue the calibration data
        ISAAC_MOVEMENT_QUEUE.put(calibration_data)
        
    except Exception as e:
        logging.error(f"(fundamental_movement.py): Failed to queue calibration position for {joint_name}: {e}\n")


def _apply_single_joint_position_isaac(joint_name, angle_rad, velocity=0.5):
    """
    Apply a single joint position for Isaac Sim calibration.
    Args:
        joint_name: Full joint name (e.g., 'FL_hip')
        angle_rad: Target angle in radians
        velocity: Joint velocity in radians/second (default: 0.5)
    """
    try:
        joint_index = config.JOINT_INDEX_MAP.get(joint_name)
        if joint_index is None:
            logging.error(f"(fundamental_movement.py): Joint {joint_name} not found in JOINT_INDEX_MAP\n")
            return
        
        joint_count = len(config.ISAAC_ROBOT.dof_names)
        joint_positions = numpy.zeros(joint_count)
        joint_velocities = numpy.zeros(joint_count)
        joint_positions[joint_index] = angle_rad
        joint_velocities[joint_index] = velocity
        
        action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
        ARTICULATION_CONTROLLER.apply_action(action)
        
    except Exception as e:
        logging.error(f"(fundamental_movement.py): Failed to apply calibration position for {joint_name}: {e}\n")


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


########## BASIC TESTING FUNCTIONS ##########

def _move_all_joints_forward_isaac():
    """
    Move all joints to FULL_FRONT positions for Isaac Sim testing.
    """
    try:
        joint_count = len(config.ISAAC_ROBOT.dof_names)
        joint_positions = numpy.zeros(joint_count)
        joint_velocities = numpy.zeros(joint_count)
        
        # Define joint order
        joint_order = [
            ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
            ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
            ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
            ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
        ]
        
        # Set all joint positions at once
        for leg_id, joint_name in joint_order:
            joint_full_name = f"{leg_id}_{joint_name}"
            joint_index = config.JOINT_INDEX_MAP.get(joint_full_name)
            
            if joint_index is not None:
                servo_data = config.SERVO_CONFIG[leg_id][joint_name]
                full_front_angle = servo_data['FULL_FRONT_ANGLE']  # Already in radians
                
                joint_positions[joint_index] = full_front_angle
                joint_velocities[joint_index] = 1.0  # Moderate velocity
                
                # Update the current angle in config
                config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = full_front_angle
            else:
                logging.error(f"(fundamental_movement.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
        
        # Apply all joint positions in a single action
        action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
        ARTICULATION_CONTROLLER.apply_action(action)
        
        logging.info("(fundamental_movement.py): Applied all joints to FULL_FRONT positions\n")
        
    except Exception as e:
        logging.error(f"(fundamental_movement.py): Failed to move all joints to FULL_FRONT: {e}\n")


def _move_all_joints_backward_isaac():
    """
    Move all joints to FULL_BACK positions for Isaac Sim testing.
    """
    try:
        joint_count = len(config.ISAAC_ROBOT.dof_names)
        joint_positions = numpy.zeros(joint_count)
        joint_velocities = numpy.zeros(joint_count)
        
        # Define joint order
        joint_order = [
            ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
            ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
            ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
            ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
        ]
        
        # Set all joint positions at once
        for leg_id, joint_name in joint_order:
            joint_full_name = f"{leg_id}_{joint_name}"
            joint_index = config.JOINT_INDEX_MAP.get(joint_full_name)
            
            if joint_index is not None:
                servo_data = config.SERVO_CONFIG[leg_id][joint_name]
                full_back_angle = servo_data['FULL_BACK_ANGLE']  # Already in radians
                
                joint_positions[joint_index] = full_back_angle
                joint_velocities[joint_index] = 1.0  # Moderate velocity
                
                # Update the current angle in config
                config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = full_back_angle
            else:
                logging.error(f"(fundamental_movement.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
        
        # Apply all joint positions in a single action
        action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
        ARTICULATION_CONTROLLER.apply_action(action)
        
        logging.info("(fundamental_movement.py): Applied all joints to FULL_BACK positions\n")
        
    except Exception as e:
        logging.error(f"(fundamental_movement.py): Failed to move all joints to FULL_BACK: {e}\n")


def _move_all_joints_forward_physical(speed, acceleration):
    """
    Move all joints to FULL_FRONT positions for physical robot testing.
    """
    from utilities.servos import set_target
    
    for leg in config.SERVO_CONFIG:
        for joint in config.SERVO_CONFIG[leg]:
            try:
                servo_data = config.SERVO_CONFIG[leg][joint]
                set_target(servo_data['servo'], servo_data['FULL_FRONT'], speed, acceleration)
            except Exception as e:
                logging.error(f"(fundamental_movement.py): Failed to move {leg}_{joint} to FULL_FRONT: {e}\n")


def _move_all_joints_backward_physical(speed, acceleration):
    """
    Move all joints to FULL_BACK positions for physical robot testing.
    """
    from utilities.servos import set_target
    
    for leg in config.SERVO_CONFIG:
        for joint in config.SERVO_CONFIG[leg]:
            try:
                servo_data = config.SERVO_CONFIG[leg][joint]
                set_target(servo_data['servo'], servo_data['FULL_BACK'], speed, acceleration)
            except Exception as e:
                logging.error(f"(fundamental_movement.py): Failed to move {leg}_{joint} to FULL_BACK: {e}\n")
