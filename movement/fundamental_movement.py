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

    ##### run inference before moving #####

    logging.debug(
        f"(fundamental_movement.py): Running inference for command(s) {commands} with intensity {intensity}...\n"
    )
    try: # try to run a model
        if not config.USE_SIMULATION: # if user wants to use real servos...
            if config.RL_NOT_CNN: # if running gait adjustment (production)...

                ##### run RL model(s) #####

                if not imageless_gait: # if not using imageless gait adjustment...
                    target_angles, mid_angles, movement_rates = run_gait_adjustment_standard( # run standard
                        STANDARD_RL_MODEL,
                        STANDARD_INPUT_LAYER,
                        STANDARD_OUTPUT_LAYER,
                        commands,
                        frame,
                        intensity,
                        config.SERVO_CONFIG
                    )
                else: # if using imageless gait adjustment...
                    target_angles, mid_angles, movement_rates = run_gait_adjustment_blind( # run blind
                        BLIND_RL_MODEL,
                        BLIND_INPUT_LAYER,
                        BLIND_OUTPUT_LAYER,
                        commands,
                        intensity,
                        config.SERVO_CONFIG
                    )

                ##### move legs and update current position #####

                # move legs and update current angles
                thread_leg_movement_angles(
                    config.SERVO_CONFIG,
                    mid_angles,
                    target_angles,
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
                # Import Isaac Sim RL functions
                from training.isaac_sim import get_rl_action_standard, get_rl_action_blind
                
                if not imageless_gait:  # if not using imageless gait adjustment (image-based agent)...
                    target_angles, mid_angles, movement_rates = get_rl_action_standard(
                        state,
                        commands,
                        intensity,
                        frame
                    )
                    logging.warning(
                        "(fundamental_movement.py): Using get_rl_action_standard placeholder. Replace with RL agent output when available."
                    )
                elif imageless_gait:  # if using imageless gait adjustment (no image)...
                    target_angles, mid_angles, movement_rates = get_rl_action_blind(
                        state,
                        commands,
                        intensity
                    )
                    logging.warning(
                        "(fundamental_movement.py): Using get_rl_action_blind placeholder. Replace with RL agent output when available."
                    )

                ##### apply direct joint control for Isaac Sim #####

                # Apply the joint angles directly
                apply_joint_angles_isaac(
                    config.SERVO_CONFIG,
                    mid_angles,
                    target_angles,
                    movement_rates
                )
                logging.debug(f"(fundamental_movement.py): Applied joint angles for Isaac Sim: {commands}\n")

    except Exception as e: # if either model fails...
        logging.error(f"(fundamental_movement.py): Failed to run AI for command: {e}\n")

    ##### move legs #####

    try: # try to update leg gait
        #logging.info(f"(fundamental_movement.py): Executed move_direction() with intensity: {intensity}\n")
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


########## ANGLE-BASED LEG MOVEMENT ##########

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
        logging.error(f"(fundamental_movement.py): Failed to swing leg {leg_id} with angles: {e}\n")


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
            logging.error(f"(fundamental_movement.py): Failed to move {leg_id}_{joint_name} to angle: {e}\n")


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
    if not config.USE_SIMULATION and not config.USE_ISAAC_SIM:  # Physical robot
        servo_data = config.SERVO_CONFIG[leg_id][joint_name]
        is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
        pwm = map_angle_to_servo_position(target_angle, servo_data, 0, is_inverted)
        set_target(servo_data['servo'], pwm, speed, acceleration)
        
    elif config.USE_SIMULATION and not config.USE_ISAAC_SIM:  # PyBullet
        # Import simulation variables from isaac_sim.py
        from training.isaac_sim import ROBOT_ID, JOINT_MAP
        angle_rad = target_angle  # Already in radians
        joint_key = (leg_id, joint_name)
        if joint_key in JOINT_MAP:
            joint_index = JOINT_MAP[joint_key]
            pybullet.setJointMotorControl2(
                bodyUniqueId=ROBOT_ID,
                jointIndex=joint_index,
                controlMode=pybullet.POSITION_CONTROL,
                targetPosition=angle_rad,
                force=4.41
            )
        else:
            logging.warning(f"(fundamental_movement.py): Joint {joint_key} not found in PyBullet joint map\n")
    
    # Update current angle in config
    config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = target_angle


def move_foot_to_pos_physical(leg_id, pos, speed, acceleration, use_bezier):
    """
    Move foot to position for physical robot (legacy coordinate-based system).
    This is kept for neutral position compatibility.
    """
    if use_bezier:
        # Bezier curve implementation would go here
        # For now, just move directly to position
        pass
    
    # Use inverse kinematics to convert position to angles
    hip_angle, upper_angle, lower_angle = k.inverse_kinematics(pos['x'], pos['y'], pos['z'])
    
    # Move each joint
    for joint, angle in zip(['hip', 'upper', 'lower'], [hip_angle, upper_angle, lower_angle]):
        servo_data = config.SERVO_CONFIG[leg_id][joint]
        is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
        pwm = map_angle_to_servo_position(angle, servo_data, 0, is_inverted)
        set_target(servo_data['servo'], pwm, speed, acceleration)
        
        # Update current angle
        config.SERVO_CONFIG[leg_id][joint]['CURRENT_ANGLE'] = math.radians(angle)


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


########## ISAAC SIM AI AGENT JOINT CONTROL ##########

def apply_joint_angles_isaac(current_servo_config, target_angles, mid_angles, movement_rates):
    """
    Apply joint angles directly for Isaac Sim AI agent training.
    This function moves all joints to their target angles in a single ArticulationAction.
    
    Args:
        current_servo_config: Current servo configuration with CURRENT_ANGLE values
        target_angles: Target joint angles for each leg (similar to SERVO_CONFIG structure)
        mid_angles: Mid joint angles for each leg (similar to SERVO_CONFIG structure)
        movement_rates: Movement rate parameters for each leg
    """
    try:
        joint_count = len(config.ISAAC_ROBOT.dof_names)
        joint_positions = numpy.zeros(joint_count)
        joint_velocities = numpy.zeros(joint_count)
        
        # Define joint order (same as working functions)
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
                # Get target angle from the AI agent's output
                target_angle = target_angles[leg_id][joint_name]
                
                # Get velocity from movement_rates (convert to reasonable range)
                # Default to 1.0 if not specified, similar to working functions
                velocity = movement_rates[leg_id].get('speed', 1000) / 1000.0  # Convert to reasonable velocity
                
                joint_positions[joint_index] = target_angle
                joint_velocities[joint_index] = velocity
                
                # Update the current angle in config
                config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = target_angle
            else:
                logging.error(f"(fundamental_movement.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
        
        # Apply all joint positions in a single action (same as working functions)
        action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
        ARTICULATION_CONTROLLER.apply_action(action)
        
        #logging.debug(f"(fundamental_movement.py): Applied AI agent joint angles for Isaac Sim\n")
        
    except Exception as e:
        logging.error(f"(fundamental_movement.py): Failed to apply AI agent joint angles for Isaac Sim: {e}\n")


########## NEUTRAL POSITION FUNCTIONS ##########

def neutral_position(intensity):
    """
    Set all legs to neutral position with two modes:
    - Physical robot: Uses move_foot_to_pos with 3D coordinates
    - Isaac Sim: Uses direct joint control with NEUTRAL_ANGLE
    """
    
    ##### get intensity #####
    try:
        speed, acceleration = interpret_intensity(intensity)
    except Exception as e:
        logging.error(f"(fundamental_movement.py): Failed to interpret intensity in neutral_position(): {e}\n")
        return

    ##### move legs to neutral based on simulation mode #####
    try:
        if config.USE_SIMULATION and config.USE_ISAAC_SIM:
            # Isaac Sim mode: Direct joint control using NEUTRAL_ANGLE
            _neutral_position_isaac()
        else:
            # Physical robot mode: Use 3D coordinates
            _neutral_position_physical(speed, acceleration)
            
    except Exception as e:
        logging.error(f"(fundamental_movement.py): Failed to move legs to neutral position: {e}\n")


def _neutral_position_isaac():
    """
    Set all joints to neutral position for Isaac Sim using direct joint control.
    """
    try:
        joint_count = len(config.ISAAC_ROBOT.dof_names)
        joint_positions = numpy.zeros(joint_count)
        joint_velocities = numpy.zeros(joint_count)
        
        # Define joint order (same as calibration)
        joint_order = [
            ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
            ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
            ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
            ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
        ]
        
        #logging.info("(fundamental_movement.py): Moving all joints to neutral position in Isaac Sim...\n")
        
        # Set all joint positions at once
        for leg_id, joint_name in joint_order:
            joint_full_name = f"{leg_id}_{joint_name}"
            joint_index = config.JOINT_INDEX_MAP.get(joint_full_name)
            
            if joint_index is not None:
                servo_data = config.SERVO_CONFIG[leg_id][joint_name]
                neutral_angle = servo_data['NEUTRAL_ANGLE']  # Always 0.0 radians
                
                joint_positions[joint_index] = neutral_angle
                joint_velocities[joint_index] = 1.0  # Moderate velocity
                
                # Update the current angle in config
                config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = neutral_angle
            else:
                logging.error(f"(fundamental_movement.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
        
        # Apply all joint positions in a single action
        action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
        ARTICULATION_CONTROLLER.apply_action(action)
        
        logging.info("(fundamental_movement.py): Applied all joints to neutral positions\n")
        
    except Exception as e:
        logging.error(f"(fundamental_movement.py): Failed to move all joints to neutral: {e}\n")


def _neutral_position_physical(speed, acceleration):
    """
    Set all legs to neutral position for physical robot using 3D coordinates.
    """
    # Define neutral positions for each leg
    neutral_positions = {
        'FL': config.FL_NEUTRAL,
        'FR': config.FR_NEUTRAL,
        'BL': config.BL_NEUTRAL,
        'BR': config.BR_NEUTRAL
    }
    
    logging.info("(fundamental_movement.py): Moving all legs to neutral position on physical robot...\n")
    
    # Move each leg to its neutral position
    for leg_id, neutral_pos in neutral_positions.items():
        try:
            # Use the old coordinate-based system for physical robot neutral position
            # This maintains compatibility with existing neutral position definitions
            move_foot_to_pos_physical(
                leg_id,
                neutral_pos,
                speed,
                acceleration,
                use_bezier=False
            )
        except Exception as e:
            logging.error(f"(fundamental_movement.py): Failed to move {leg_id} leg to neutral: {e}\n")


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

