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

##### import config #####

import utilities.config as config # import configuration data for servos and link lengths

##### import necessary libraries #####

import time # import time for proper leg sequencing
import threading # import threading for thread management
import random # import random for random angle radians
import logging # import logging for error handling

##### import dependencies for physical robot

if not config.USE_SIMULATION:

    ##### import necessary functions #####

    from movement.physical_joints import swing_leg, neutral_position_physical
    from utilities.inference import load_and_compile_model, run_gait_adjustment_standard, run_gait_adjustment_blind, \
        run_person_detection # load function/models for gait adjustment and person detection

    if config.RL_NOT_CNN:
        # TODO Be aware that multiple models loaded on one NCS2 may be an issue... might be worth benching one of these
        #STANDARD_RL_MODEL, STANDARD_INPUT_LAYER, STANDARD_OUTPUT_LAYER = load_and_compile_model(
            #config.INFERENCE_CONFIG['STANDARD_RL_PATH'])
        BLIND_RL_MODEL, BLIND_INPUT_LAYER, BLIND_OUTPUT_LAYER = load_and_compile_model(
            config.INFERENCE_CONFIG['BLIND_RL_PATH'])
    elif not config.RL_NOT_CNN:
        CNN_MODEL, CNN_INPUT_LAYER, CNN_OUTPUT_LAYER = load_and_compile_model(config.INFERENCE_CONFIG['CNN_PATH'])

##### import dependencies for isaac sim #####

elif config.USE_SIMULATION:

    ##### import libraries for isaac sim #####

    import queue
        
    ##### import functions for isaac sim #####

    from isaacsim.core.api.controllers.articulation_controller import ArticulationController
    from training.isaac_sim import build_isaac_joint_index_map
    from movement.isaac_joints import neutral_position_isaac, apply_joint_angles_isaac
    from training.training import get_rl_action_standard, get_rl_action_blind
        
    ##### build dependencies for isaac sim #####

    config.JOINT_INDEX_MAP = build_isaac_joint_index_map(config.ISAAC_ROBOT.dof_names)
    config.ISAAC_ROBOT_ARTICULATION_CONTROLLER = ArticulationController() # create new articulation controller instance
    config.ISAAC_ROBOT_ARTICULATION_CONTROLLER.initialize(config.ISAAC_ROBOT) # initialize the controller


########## CREATE DEPENDENCIES ##########

##### simulation variables (set by control_logic.py) #####

if config.USE_SIMULATION:
    ROBOT_ID = None  # will be set by control_logic.py
    JOINT_MAP = {}   # will be set by control_logic.py
else:
    ROBOT_ID = None
    JOINT_MAP = {}

##### define servos #####

#upper_leg_servos = { # define upper leg servos

    #"FL": config.SERVO_CONFIG['FL']['upper'],  # front left
    #"FR": config.SERVO_CONFIG['FR']['upper'],  # front right
    #"BL": config.SERVO_CONFIG['BL']['upper'],  # back left
    #"BR": config.SERVO_CONFIG['BR']['upper'],  # back right
#}

#lower_leg_servos = { # define lower leg servos

    #"FL": config.SERVO_CONFIG['FL']['lower'],  # front left
    #"FR": config.SERVO_CONFIG['FR']['lower'],  # front right
    #"BL": config.SERVO_CONFIG['BL']['lower'],  # back left
    #"BR": config.SERVO_CONFIG['BR']['lower'],  # back right
#}





##############################################################
############### MOVEMENT COORDINATOR FUNCTIONS ###############
##############################################################


########## CENTRAL GAIT FUNCTION ##########

def move_direction(commands, frame, intensity, imageless_gait): # function to trot forward

    ##### preprocess commands and intensity #####

    commands = sorted(commands.split('+')) # alphabetize commands so they are uniform

    ##### run inference before moving #####

    # logging.debug(
    #     f"(movement_coordinator.py): Running inference for command(s) {commands} with intensity {intensity}...\n"
    # )
    try: # try to run a model
        if not config.USE_SIMULATION: # if user wants to use real servos...
            if config.RL_NOT_CNN: # if running gait adjustment (production)...

                ##### run RL model(s) #####

                logging.debug("Inference input:\n")
                logging.debug(f"(movement_coordinator.py): Commands: {commands}\n")
                logging.debug(f"(movement_coordinator.py): Intensity: {intensity}\n")

                if not imageless_gait: # if not using imageless gait adjustment...
                    # TODO use the blind model until I get image support going
                    target_angles, mid_angles, movement_rates = run_gait_adjustment_blind(  # run blind
                        BLIND_RL_MODEL,
                        BLIND_INPUT_LAYER,
                        BLIND_OUTPUT_LAYER,
                        commands,
                        intensity,
                        config.SERVO_CONFIG
                    )
                    #target_angles, mid_angles, movement_rates = run_gait_adjustment_standard( # run standard
                        #STANDARD_RL_MODEL,
                        #STANDARD_INPUT_LAYER,
                        #STANDARD_OUTPUT_LAYER,
                        #commands,
                        #frame,
                        #intensity,
                        #config.SERVO_CONFIG
                    #)

                else: # if using imageless gait adjustment...
                    target_angles, mid_angles, movement_rates = run_gait_adjustment_blind( # run blind
                        BLIND_RL_MODEL,
                        BLIND_INPUT_LAYER,
                        BLIND_OUTPUT_LAYER,
                        commands,
                        intensity,
                        config.SERVO_CONFIG
                    )

                logging.debug("(movement_coordinator.py): Inference Results:\n")
                logging.debug(f"(movement_coordinator.py): Target angles: {target_angles}\n")
                logging.debug(f"(movement_coordinator.py): Mid angles: {mid_angles}\n")
                logging.debug(f"(movement_coordinator.py): Movement rates: {movement_rates}\n")

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
            logging.info(f"(movement_coordinator.py): Ran AI for command(s) {commands} with intensity {intensity}\n")

        elif config.USE_SIMULATION: # if running code in simulator...

            ##### rl agent integration point #####
            # gather state for RL agent (define get_simulation_state later if needed)
            current_angles = {
                'FL': {
                    'hip': config.SERVO_CONFIG['FL']['hip']['CURRENT_ANGLE'],
                    'upper': config.SERVO_CONFIG['FL']['upper']['CURRENT_ANGLE'],
                    'lower': config.SERVO_CONFIG['FL']['lower']['CURRENT_ANGLE']
                },
                'FR': {
                    'hip': config.SERVO_CONFIG['FR']['hip']['CURRENT_ANGLE'],
                    'upper': config.SERVO_CONFIG['FR']['upper']['CURRENT_ANGLE'],
                    'lower': config.SERVO_CONFIG['FR']['lower']['CURRENT_ANGLE']
                },
                'BL': {
                    'hip': config.SERVO_CONFIG['BL']['hip']['CURRENT_ANGLE'],
                    'upper': config.SERVO_CONFIG['BL']['upper']['CURRENT_ANGLE'],
                    'lower': config.SERVO_CONFIG['BL']['lower']['CURRENT_ANGLE']
                },
                'BR': {
                    'hip': config.SERVO_CONFIG['BR']['hip']['CURRENT_ANGLE'],
                    'upper': config.SERVO_CONFIG['BR']['upper']['CURRENT_ANGLE'],
                    'lower': config.SERVO_CONFIG['BR']['lower']['CURRENT_ANGLE']
                }
            }
            
            if not imageless_gait:  # if not using imageless gait adjustment (image-based agent)...
                target_angles, mid_angles, movement_rates = get_rl_action_standard(
                    current_angles,
                    commands,
                    intensity,
                    frame
                )
                logging.warning(
                    "(movement_coordinator.py): Using get_rl_action_standard placeholder. Replace with RL agent output when available."
                )
            elif imageless_gait:  # if using imageless gait adjustment (no image)...
                target_angles, mid_angles, movement_rates = get_rl_action_blind(
                    current_angles,
                    commands,
                    intensity
                )
                #logging.warning(
                #    "(movement_coordinator.py): Using get_rl_action_blind placeholder. Replace with RL agent output when available."
                #)

            ##### apply direct joint control for Isaac Sim #####

            # Apply the joint angles directly
            apply_joint_angles_isaac(
                config.SERVO_CONFIG,
                mid_angles,
                target_angles,
                movement_rates
            )
            # logging.debug(f"(movement_coordinator.py): Applied joint angles for Isaac Sim: {commands}\n")

    except Exception as e: # if either model fails...
        logging.error(f"(movement_coordinator.py): Failed to run AI for command: {e}\n")

    ##### move legs #####

    try: # try to update leg gait
        # TODO somehow move legs after model has been activated
        # logging.info(f"(movement_coordinator.py): Executed move_direction() with intensity: {intensity}\n")
        time.sleep(0.1) # wait for legs to reach positions

    except Exception as e: # if gait update fails...
        logging.error(f"(movement_coordinator.py): Failed to gait-cycle legs in move_direction(): {e}\n")


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


########## RANDOM ACTION FUNCTION ##########

def get_random_action(state, commands, intensity): # used to generate random movement for testing
    """
    Generate random joint movements for testing purposes.
    This function was moved from get_rl_action_blind to preserve random movement capability.
    
    Args:
        state: The current state of the robot/simulation (to be defined).
        commands: The movement commands.
        intensity: The movement intensity.
    Returns:
        target_angles: dict of target joint angles for each leg (similar to SERVO_CONFIG structure).
        mid_angles: dict of mid joint angles for each leg (similar to SERVO_CONFIG structure).
        movement_rates: dict of movement rate parameters for each leg.
    """
    
    target_angles = {}
    mid_angles = {}
    movement_rates = {}
    
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        target_angles[leg_id] = {}
        mid_angles[leg_id] = {}
        movement_rates[leg_id] = {'speed': 1.0, 'acceleration': 0.5}  # 1 rad/s, 0.5 rad/s²
        
        for joint_name in ['hip', 'upper', 'lower']:
            servo_data = config.SERVO_CONFIG[leg_id][joint_name]
            
            # Get the valid range for this joint
            full_back_angle = servo_data['FULL_BACK_ANGLE']  # Already in radians
            full_front_angle = servo_data['FULL_FRONT_ANGLE']  # Already in radians
            
            # Ensure we have the correct order (back < front)
            min_angle = min(full_back_angle, full_front_angle)
            max_angle = max(full_back_angle, full_front_angle)
            
            # Generate random angles within the valid range
            target_angle = random.uniform(min_angle, max_angle)
            mid_angle = random.uniform(min_angle, max_angle)
            
            target_angles[leg_id][joint_name] = target_angle
            mid_angles[leg_id][joint_name] = mid_angle
    
    return target_angles, mid_angles, movement_rates


########## NEUTRAL POSITION FUNCTION ##########

def neutral_position(intensity):

    ##### move legs to neutral based on simulation mode #####

    try: # try to move legs to neutral position
        if config.USE_SIMULATION: # if using isaac sim...
            neutral_position_isaac() # dont use intensity, velocity is not controllable so there is no point
        else: # if using physical robot...
            neutral_position_physical(intensity) # pass intensity
            
    except Exception as e: # if failed to move legs to neutral position...
        logging.error(f"(movement_coordinator.py): Failed to move legs to neutral position: {e}\n")
        