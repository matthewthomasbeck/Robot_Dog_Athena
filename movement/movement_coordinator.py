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

import utilities.config as config # import configuration data for servos and link lengths

##### import necessary libraries #####

import time # import time for proper leg sequencing
import threading # import threading for thread management
import random # import random for random angle radians
import logging # import logging for error handling

##### import necessary functions #####

from movement.physical_joints import swing_leg, neutral_position_physical
from utilities.inference import load_and_compile_model, run_gait_adjustment_standard, run_gait_adjustment_blind, \
    run_person_detection # load function/models for gait adjustment and person detection
from utilities.accelerometer import get_orientation_vectors # import helper to build Isaac Lab IMU vectors

if config.RL_NOT_CNN:
    # TODO Be aware that multiple models loaded on one NCS2 may be an issue... might be worth benching one of these
    #STANDARD_RL_MODEL, STANDARD_INPUT_LAYER, STANDARD_OUTPUT_LAYER = load_and_compile_model(
        #config.INFERENCE_CONFIG['STANDARD_RL_PATH'])
    BLIND_RL_MODEL, BLIND_INPUT_LAYER, BLIND_OUTPUT_LAYER = load_and_compile_model(
        config.INFERENCE_CONFIG['BLIND_RL_PATH'])
elif not config.RL_NOT_CNN:
    CNN_MODEL, CNN_INPUT_LAYER, CNN_OUTPUT_LAYER = load_and_compile_model(config.INFERENCE_CONFIG['CNN_PATH'])


########## CREATE DEPENDENCIES ##########

##### simulation variables (set by control_logic.py) #####

ROBOT_ID = None
JOINT_MAP = {}





##############################################################
############### MOVEMENT COORDINATOR FUNCTIONS ###############
##############################################################


########## CENTRAL GAIT FUNCTION ##########

def move_direction(commands, camera_frames, intensity, imageless_gait): # function to trot forward

    ##### preprocess commands and intensity #####

    if isinstance(commands, list): # if new format...
        # Filter out tilt commands (arrowup/arrowdown) and create RL model input
        rl_commands = []
        tilt_command = None

        ##### extract movement commands #####

        if commands[0] is not None:  # forward/backward
            rl_commands.append(commands[0])
        if commands[1] is not None:  # left/right
            rl_commands.append(commands[1])
        if commands[2] is not None:  # rotation
            rl_commands.append(commands[2])
        if commands[3] is not None:  # tilt (store for later use if needed)
            tilt_command = commands[3]

        commands = rl_commands
        
        # Log the command processing
        #logging.debug(f"(movement_coordinator.py): Processed fixed-length list: {rl_commands}")
        #if tilt_command:
            #logging.debug(f"(movement_coordinator.py): Tilt command detected: {tilt_command} (not passed to RL model)")

    else: # if old format...
        commands = sorted(commands.split('+')) # alphabetize commands so they are uniform

    ##### collect orientation data (Isaac Lab-style vectors) #####

    orientation_vectors = get_orientation_vectors()

    ##### run inference before moving #####

    try: # try to run a model
        if config.RL_NOT_CNN: # if running gait adjustment (production)...

            ##### run RL model(s) #####

            logging.debug("Inference input:\n")
            logging.debug(f"(movement_coordinator.py): Commands: {commands}\n")
            logging.debug(f"(movement_coordinator.py): Intensity: {intensity}\n")
            logging.debug(f"(movement_coordinator.py): Orientation vectors: {orientation_vectors}\n")

            if not imageless_gait: # if not using imageless gait adjustment...
                # TODO use the blind model until I get image support going
                target_angles, movement_rates = run_gait_adjustment_blind(  # run blind
                    BLIND_RL_MODEL,
                    BLIND_INPUT_LAYER,
                    BLIND_OUTPUT_LAYER,
                    commands,
                    intensity,
                    orientation_vectors
                )
                #target_angles, movement_rates = run_gait_adjustment_standard( # run standard
                    #STANDARD_RL_MODEL,
                    #STANDARD_INPUT_LAYER,
                    #STANDARD_OUTPUT_LAYER,
                    #commands,
                    #camera_frames[0]['inference_frame'],
                    #intensity,
                    #orientation
                #)

            else: # if using imageless gait adjustment...
                target_angles, movement_rates = run_gait_adjustment_blind( # run blind
                    BLIND_RL_MODEL,
                    BLIND_INPUT_LAYER,
                    BLIND_OUTPUT_LAYER,
                    commands,
                    intensity,
                    orientation_vectors
                )

            logging.debug("(movement_coordinator.py): Inference Results:\n")
            logging.debug(f"(movement_coordinator.py): Target angles: {target_angles}\n")
            logging.debug(f"(movement_coordinator.py): Movement rates: {movement_rates}\n")

            ##### move legs and wait long enough for arcs to finish #####

            step_wait = _estimate_step_wait_s(target_angles, movement_rates)
            thread_leg_movement(
                config.SERVO_CONFIG,
                target_angles,
                movement_rates
            )
            time.sleep(step_wait)

        else: # if running person detection (testing)...
            run_person_detection(
                CNN_MODEL,
                CNN_INPUT_LAYER,
                CNN_OUTPUT_LAYER,
                frame,
                run_inference=False
            )
            time.sleep(float(config.GAIT_CONFIG['POLICY_DT_S']))
        logging.info(f"(movement_coordinator.py): Ran AI for command(s) {commands} with intensity {intensity}\n")

    except Exception as e: # if either model fails...
        logging.error(f"(movement_coordinator.py): Failed to run AI for command: {e}\n")


def _estimate_step_wait_s(target_angles, movement_rates):
    """Seconds to wait after commanding targets so servos can finish the arc.

    Maestro set_target is fire-and-forget; without this, the next policy step
    overwrites mid-swing and produces half-arcs.
    """
    gait = config.GAIT_CONFIG
    policy_dt = float(gait['POLICY_DT_S'])
    margin = float(gait['SETTLE_MARGIN_S'])
    max_wait = float(gait['MAX_STEP_WAIT_S'])
    default_speed = float(gait['SERVO_SPEED_RAD_S'])

    max_travel_s = 0.0
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        if leg_id not in target_angles:
            continue
        for joint_name in ['hip', 'upper', 'lower']:
            if joint_name not in target_angles[leg_id]:
                continue
            current = float(config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'])
            target = float(target_angles[leg_id][joint_name])
            speed = float(movement_rates.get(leg_id, {}).get(joint_name, default_speed))
            speed = max(speed, 0.25)
            max_travel_s = max(max_travel_s, abs(target - current) / speed)

    wait = max(policy_dt, max_travel_s + margin)
    wait = min(wait, max_wait)
    logging.debug(
        f"(movement_coordinator.py): step_wait={wait:.3f}s "
        f"(travel={max_travel_s:.3f}s, policy_dt={policy_dt:.3f}s)\n"
    )
    return wait


########## THREAD LEG MOVEMENT ##########

def thread_leg_movement(current_servo_config, target_angles, movement_rates): # function to separate leg movement
    
    leg_threads = []  # create a list to hold threads for each leg
    for leg_id in ['FL', 'FR', 'BL', 'BR']:  # loop through each leg and create a thread to move
        t = threading.Thread(
            target=swing_leg,
            args=(
                leg_id,
                target_angles[leg_id],
                movement_rates[leg_id]
            )
        )
        leg_threads.append(t)
        t.start()
    for t in leg_threads:  # wait for all legs to finish
        t.join()


########## ISAAC LAB JOINT MIRROR ##########

def apply_isaac_joint_targets(joints, speed_rad_s=None):
    """Apply absolute joint targets from Isaac Lab (radians), skipping RL.

    ``joints`` keys are Isaac names like ``FL_hip`` or nested ``{"FL": {"hip": ...}}``.
    """
    cfg = config.ISAAC_MIRROR_CONFIG
    if speed_rad_s is None:
        speed_rad_s = float(cfg["DEFAULT_SPEED_RAD_S"])
    max_delta = float(cfg["MAX_DELTA_RAD"])

    # Normalize to nested {leg: {joint: angle}}
    nested: dict = {leg: {} for leg in ["FL", "FR", "BL", "BR"]}
    if joints and any(k in joints for k in ["FL", "FR", "BL", "BR"]):
        for leg_id in ["FL", "FR", "BL", "BR"]:
            if leg_id in joints and isinstance(joints[leg_id], dict):
                nested[leg_id].update(joints[leg_id])
    else:
        for joint_key, angle in (joints or {}).items():
            if "_" not in str(joint_key):
                continue
            leg_id, joint_name = str(joint_key).split("_", 1)
            if leg_id in nested and joint_name in ["hip", "upper", "lower"]:
                nested[leg_id][joint_name] = float(angle)

    target_angles = {}
    movement_rates = {}
    for leg_id in ["FL", "FR", "BL", "BR"]:
        target_angles[leg_id] = {}
        movement_rates[leg_id] = {}
        for joint_name in ["hip", "upper", "lower"]:
            servo_data = config.SERVO_CONFIG[leg_id][joint_name]
            min_angle = min(servo_data["FULL_BACK_ANGLE"], servo_data["FULL_FRONT_ANGLE"])
            max_angle = max(servo_data["FULL_BACK_ANGLE"], servo_data["FULL_FRONT_ANGLE"])

            joint_key = f"{leg_id}_{joint_name}"
            if joint_name in nested[leg_id]:
                desired = float(nested[leg_id][joint_name])
            else:
                desired = float(config.ISAAC_DEFAULT_JOINT_POS[joint_key])

            current = float(servo_data["CURRENT_ANGLE"])
            # Rate-limit large jumps from a bad packet / reconnect
            desired = max(current - max_delta, min(current + max_delta, desired))
            desired = float(max(min_angle, min(max_angle, desired)))

            target_angles[leg_id][joint_name] = desired
            movement_rates[leg_id][joint_name] = float(speed_rad_s)

    thread_leg_movement(config.SERVO_CONFIG, target_angles, movement_rates)


def apply_isaac_default_pose(speed_rad_s=None):
    """Hold the Isaac Lab default standing/camber pose."""
    joints = {k: float(v) for k, v in config.ISAAC_DEFAULT_JOINT_POS.items()}
    apply_isaac_joint_targets(joints, speed_rad_s=speed_rad_s)


########## RANDOM ACTION FUNCTION ##########

def get_random_action(state, commands, intensity): # used to generate random movement for testing

    ##### set variables #####

    target_angles = {}
    mid_angles = {}
    movement_rates = {}

    ##### generate random angles and rates #####

    for leg_id in ['FL', 'FR', 'BL', 'BR']: # loop through each leg

        target_angles[leg_id] = {}
        mid_angles[leg_id] = {}
        movement_rates[leg_id] = {'speed': 1.0, 'acceleration': 0.5}  # 1 rad/s, 0.5 rad/s²
        
        for joint_name in ['hip', 'upper', 'lower']: # loop through each joint

            ##### get valid range #####

            servo_data = config.SERVO_CONFIG[leg_id][joint_name]
            full_back_angle = servo_data['FULL_BACK_ANGLE']
            full_front_angle = servo_data['FULL_FRONT_ANGLE']

            ##### ensure correct order #####

            min_angle = min(full_back_angle, full_front_angle)
            max_angle = max(full_back_angle, full_front_angle)
            
            ##### generate random angles #####

            target_angle = random.uniform(min_angle, max_angle)
            mid_angle = random.uniform(min_angle, max_angle)
            target_angles[leg_id][joint_name] = target_angle
            mid_angles[leg_id][joint_name] = mid_angle
    
    return target_angles, mid_angles, movement_rates


########## NEUTRAL POSITION FUNCTION ##########

def neutral_position(intensity):

    ##### move legs to neutral based on simulation mode #####

    try: # try to move legs to neutral position
        neutral_position_physical(intensity) # pass intensity
    except Exception as e: # if failed to move legs to neutral position...
        logging.error(f"(movement_coordinator.py): Failed to move legs to neutral position: {e}\n")
  