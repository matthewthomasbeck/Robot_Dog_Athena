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

import threading
import queue

##### import necessary utilities #####

from utilities.log import initialize_logging  # import logging setup
from utilities.receiver import initialize_receiver, interpret_commands  # import receiver initialization functions
from utilities.camera import initialize_camera, decode_frame  # import to start camera logic
import utilities.internet as internet  # dynamically import internet utilities to be constantly updated (sending frames)
import utilities.config as config  # import config for DEFAULT_INTENSITY

##### import movement functions #####

from movement.fundamental_movement import *  # import fundamental movement functions
from movement.standing.standing import *  # import standing functions

########## CREATE DEPENDENCIES ##########

##### initialize all utilities #####

LOGGER = initialize_logging()  # set up logging
CAMERA_PROCESS = initialize_camera()  # create camera process
CHANNEL_DATA = initialize_receiver()  # get pigpio instance, decoders, and channel data
SOCK = internet.initialize_backend_socket()  # initialize EC2 socket connection
COMMAND_QUEUE = internet.initialize_command_queue(SOCK)  # initialize command queue for socket communication
FRAME_QUEUE = queue.Queue(maxsize=1)

##### create different control mode #####

MODE = 'web'  # default mode is radio control, can be changed to 'radio' or 'web' for variable control
IS_COMPLETE = True
IS_NEUTRAL = False  # set global neutral standing boolean
CURRENT_LEG = 'FL'  # set global current leg
logging.info(f"(control_logic.py): IS_COMPLETE set to true.\n")
logging.debug("(control_logic.py): Starting control_logic.py script...\n")  # log start of script


#########################################
############### RUN ROBOT ###############
#########################################


########## RUN ROBOTIC PROCESS ##########

def _perception_loop(CHANNEL_DATA):  # central function that runs robot

    global IS_COMPLETE, IS_NEUTRAL, CURRENT_LEG

    ##### set/initialize variables #####

    mjpeg_buffer = b''  # initialize buffer for MJPEG frames

    ##### run robotic logic #####

    try:  # try to run robot startup sequence

        squatting_position(1)
        time.sleep(3)
        # tippytoes_position(1) TODO keep me commented out for now
        # time.sleep(3) TODO me too
        neutral_position(1)
        time.sleep(3)
        IS_NEUTRAL = True  # set is_neutral to True
        time.sleep(3)  # wait for 3 seconds

    except Exception as e:  # if there is an error, log error

        logging.error(f"(control_logic.py): Failed to move to neutral standing position in runRobot: {e}\n")

    try:  # try to run main robotic process

        ##### stream video, run inference, and control the robot #####

        while True:
            mjpeg_buffer, frame_data, frame = decode_frame(CAMERA_PROCESS, mjpeg_buffer)
            internet.stream_to_backend(SOCK, frame_data)  # stream frame data to ec2 instance
            command = None

            if MODE == 'web' and COMMAND_QUEUE is not None and not COMMAND_QUEUE.empty():
                command = COMMAND_QUEUE.get()
                if IS_COMPLETE: # if movement is complete, run command
                    logging.info(f"(control_logic.py): Received command '{command}' from queue (WILL RUN).\n")
                else:
                    logging.info(f"(control_logic.py): Received command '{command}' from queue (WILL BLOCK).\n")

            # LEGACY RADIO COMMAND CODE, UNDER NO CIRCUMSTANCES REMOVE WHATSOEVER (I will get around to renewing radio support for override system)
            #if MODE == 'radio':
                #commands = interpret_commands(CHANNEL_DATA)
                #for channel, (action, intensity) in commands.items():
                    #is_neutral = _execute_radio_commands(channel, action, intensity, is_neutral)

            # NEW MULTI-CHANNEL RADIO COMMAND SYSTEM
            if MODE == 'radio':
                commands = interpret_commands(CHANNEL_DATA)
                if IS_COMPLETE:  # if movement is complete, process radio commands
                    logging.debug(f"(control_logic.py): Processing radio commands: {commands}\n")
                    threading.Thread(target=_handle_command, args=(commands, frame), daemon=True).start()

            # WEB COMMAND HANDLING
            if command and IS_COMPLETE: # if command present and movement complete...
                logging.debug(f"(control_logic.py): Running command: {command}...\n")
                threading.Thread(target=_handle_command, args=(command, frame), daemon=True).start()

            # NEUTRAL POSITION HANDLING (for both modes)
            elif not command and IS_COMPLETE and not IS_NEUTRAL: # if no command and movement complete and not neutral...
                logging.info(f"(control_logic.py): No command received, returning to neutral position...\n")
                threading.Thread(target=_handle_command, args=('n', frame), daemon=True).start()

    except KeyboardInterrupt:  # if user ends program...
        logging.info("(control_logic.py): KeyboardInterrupt received, exiting.\n")

    except Exception as e:  # if something breaks and only God knows what it is...
        logging.error(f"(control_logic.py): Unexpected exception in main loop: {e}\n")
        exit(1)


########## HANDLE COMMANDS ##########

def _handle_command(command, frame):

    logging.debug(f"(control_logic.py): Threading command: {command}...\n")

    global IS_COMPLETE, IS_NEUTRAL, CURRENT_LEG

    IS_COMPLETE = False  # block new commands until movement is complete
    logging.info(f"(control_logic.py): IS_COMPLETE set to false.\n")

    # TODO deprecated CNN-based inference model
    #if command != 'n': # don't waste energy running inference for neutral command
        #try:
            #run_inference(
                #COMPILED_MODEL,
                #INPUT_LAYER,
                #OUTPUT_LAYER,
                #frame,
                #run_inference=False
            #)
            #logging.info(f"(control_logic.py): Ran inference for command: {command}\n")
        #except Exception as e:
            #logging.error(f"(control_logic.py): Failed to run inference for command: {e}\n")

    # Accept both string and list (for future-proofing)
    if isinstance(command, str):
        if '+' in command:
            keys = command.split('+')
        elif command == 'n':
            keys = []
        else:
            keys = [command]
    elif isinstance(command, (list, tuple)):
        keys = list(command)
    elif isinstance(command, dict):
        # Radio commands come as a dict from interpret_commands
        keys = []  # Not used for radio mode
    else:
        keys = []

    if MODE == 'radio':
        try:
            logging.debug(f"(control_logic.py): Executing radio commands: {command}...\n")
            IS_NEUTRAL, CURRENT_LEG = _execute_radio_commands(
                command,  # command should be the full commands dict from interpret_commands
                frame,
                IS_NEUTRAL,
                CURRENT_LEG,
                tune_mode=False
            )
            logging.info(f"(control_logic.py): Executed radio commands: {command}\n")
            IS_COMPLETE = True
        except Exception as e:
            logging.error(f"(control_logic.py): Failed to execute radio commands: {e}\n")
            IS_NEUTRAL = False
            IS_COMPLETE = True

    elif MODE == 'web':
        try:
            logging.debug(f"(control_logic.py): Executing keyboard command: {keys}\n")
            IS_NEUTRAL, CURRENT_LEG = _execute_keyboard_commands(
                keys,
                frame,
                IS_NEUTRAL,
                CURRENT_LEG,
                config.DEFAULT_INTENSITY,
                tune_mode=False
            )
            logging.info(f"(control_logic.py): Executed keyboard command: {keys}\n")
            IS_COMPLETE = True
        except Exception as e:
            logging.error(f"(control_logic.py): Failed to execute keyboard command: {e}\n")
            IS_NEUTRAL = False
            IS_COMPLETE = True

    logging.info(f"(control_logic.py): IS_COMPLETE set to true.\n")

##### keyboard commands for tuning mode and normal operation #####

def _execute_keyboard_commands(keys, frame, is_neutral, current_leg, intensity, tune_mode):
    # keys: list of pressed keys, e.g. ['w', 'd', 'arrowup']

    if not tune_mode:
        # Cancel out contradictory keys
        if 'w' in keys and 's' in keys:
            keys = [k for k in keys if k not in ['w', 's']]
        if 'a' in keys and 'd' in keys:
            keys = [k for k in keys if k not in ['a', 'd']]
        if 'arrowleft' in keys and 'arrowright' in keys:
            keys = [k for k in keys if k not in ['arrowleft', 'arrowright']]
        if 'arrowup' in keys and 'arrowdown' in keys:
            keys = [k for k in keys if k not in ['arrowup', 'arrowdown']]

        # Movement direction logic
        direction = None

        # Movement (WASD and diagonals)
        if 'w' in keys and 'd' in keys:
            direction = 'w+d'
        elif 'w' in keys and 'a' in keys:
            direction = 'w+a'
        elif 's' in keys and 'd' in keys:
            direction = 's+d'
        elif 's' in keys and 'a' in keys:
            direction = 's+a'
        elif 'w' in keys:
            direction = 'w'
        elif 's' in keys:
            direction = 's'
        elif 'a' in keys:
            direction = 'a'
        elif 'd' in keys:
            direction = 'd'

        # Rotation (arrow keys)
        if 'arrowleft' in keys:
            direction = 'arrowleft'
        elif 'arrowright' in keys:
            direction = 'arrowright'

        # Tilt (arrow up/down)
        if 'arrowup' in keys:
            direction = 'arrowup'
        elif 'arrowdown' in keys:
            direction = 'arrowdown'

        # Neutral and special actions
        if 'n' in keys or not keys:
            logging.info(f"(control_logic.py): NEUTRAL\n")
            neutral_position(10)
            is_neutral = True
        elif ' ' in keys:
            logging.info(f"(control_logic.py): space: LIE DOWN\n")
            squatting_position(1)
            is_neutral = False
        elif direction:
            logging.info(f"(control_logic.py): {keys}: {direction}\n")
            #trot_forward(intensity)
            move_direction(direction, frame, intensity)
            is_neutral = False
        else:
            logging.warning(f"(control_logic.py): Invalid command: {keys}\n")

    else:
        # Tuning mode - handle individual key actions
        for key in keys:
            if key in ADJUSTMENT_FUNCS.get(current_leg, {}):
                try:
                    ADJUSTMENT_FUNCS[current_leg][key]()
                    logging.debug(f"(control_logic.py): Tuning {current_leg} with {key}\n")
                except Exception as e:
                    logging.error(f"(control_logic.py): Failed to adjust {current_leg} with {key}: {e}\n")
            elif key == 'q':
                # Switch to next leg
                legs = ['FL', 'FR', 'BL', 'BR']
                current_index = legs.index(current_leg)
                current_leg = legs[(current_index + 1) % len(legs)]
                logging.info(f"(control_logic.py): Switched to leg: {current_leg}\n")
            elif key == 'e':
                # Switch to previous leg
                legs = ['FL', 'FR', 'BL', 'BR']
                current_index = legs.index(current_leg)
                current_leg = legs[(current_index - 1) % len(legs)]
                logging.info(f"(control_logic.py): Switched to leg: {current_leg}\n")

    return is_neutral, current_leg


##### radio commands for multi-channel processing #####

def _execute_radio_commands(commands, frame, is_neutral, current_leg, tune_mode):
    # commands: dict of channel states, e.g. {'channel_5': ('MOVE_FORWARD', 8), 'channel_6': ('SHIFT_LEFT', 5)}

    if not tune_mode:
        # Extract active commands (non-neutral)
        active_commands = {channel: (action, intensity) for channel, (action, intensity) in commands.items()
                          if action != 'NEUTRAL'}

        if not active_commands:
            # All channels neutral, return to neutral position
            logging.info(f"(control_logic.py): All channels neutral, returning to neutral position\n")
            neutral_position(10)
            is_neutral = True
            return is_neutral, current_leg

        # Cancel out contradictory commands (similar to keyboard logic)
        # Check for contradictory movement commands
        move_actions = []
        for channel in ['channel_5', 'channel_6']:
            if channel in active_commands:
                action = active_commands[channel][0]
                if 'MOVE' in action or 'SHIFT' in action:
                    move_actions.append((channel, action))

        # Remove contradictory movements (this is handled by the direction logic below)
        # Instead, we'll prioritize based on intensity and handle in direction determination

        # Determine primary direction and intensity
        direction = None
        max_intensity = 0

        # Movement combinations (channels 5 and 6)
        move_forward = active_commands.get('channel_5', ('NEUTRAL', 0))[0] == 'MOVE FORWARD'
        move_backward = active_commands.get('channel_5', ('NEUTRAL', 0))[0] == 'MOVE BACKWARD'
        shift_left = active_commands.get('channel_6', ('NEUTRAL', 0))[0] == 'SHIFT LEFT'
        shift_right = active_commands.get('channel_6', ('NEUTRAL', 0))[0] == 'SHIFT RIGHT'

        # Rotation (channel 3)
        rotate_left = active_commands.get('channel_3', ('NEUTRAL', 0))[0] == 'ROTATE LEFT'
        rotate_right = active_commands.get('channel_3', ('NEUTRAL', 0))[0] == 'ROTATE RIGHT'

        # Tilt (channel 4)
        tilt_up = active_commands.get('channel_4', ('NEUTRAL', 0))[0] == 'TILT UP'
        tilt_down = active_commands.get('channel_4', ('NEUTRAL', 0))[0] == 'TILT DOWN'

        # Special actions (channels 0, 1, 2, 7)
        look_up = active_commands.get('channel_0', ('NEUTRAL', 0))[0] == 'LOOK UP'
        look_down = active_commands.get('channel_0', ('NEUTRAL', 0))[0] == 'LOOK DOWN'
        trigger_shoot = active_commands.get('channel_1', ('NEUTRAL', 0))[0] == 'TRIGGER SHOOT'
        squat_down = active_commands.get('channel_2', ('NEUTRAL', 0))[0] == 'SQUAT DOWN'
        squat_up = active_commands.get('channel_2', ('NEUTRAL', 0))[0] == 'SQUAT UP'
        plus_action = active_commands.get('channel_7', ('NEUTRAL', 0))[0] == '+'
        minus_action = active_commands.get('channel_7', ('NEUTRAL', 0))[0] == '-'

        # Build complex direction string for computer-friendly parsing
        direction_parts = []

        # Handle diagonal movements (combine channels 5 and 6)
        if move_forward and shift_left:
            direction_parts.append('w+a')
            max_intensity = max(
                active_commands.get('channel_5', ('NEUTRAL', 0))[1],
                active_commands.get('channel_6', ('NEUTRAL', 0))[1]
            )
        elif move_forward and shift_right:
            direction_parts.append('w+d')
            max_intensity = max(
                active_commands.get('channel_5', ('NEUTRAL', 0))[1],
                active_commands.get('channel_6', ('NEUTRAL', 0))[1]
            )
        elif move_backward and shift_left:
            direction_parts.append('s+a')
            max_intensity = max(
                active_commands.get('channel_5', ('NEUTRAL', 0))[1],
                active_commands.get('channel_6', ('NEUTRAL', 0))[1]
            )
        elif move_backward and shift_right:
            direction_parts.append('s+d')
            max_intensity = max(
                active_commands.get('channel_5', ('NEUTRAL', 0))[1],
                active_commands.get('channel_6', ('NEUTRAL', 0))[1]
            )
        # Handle single movements
        elif move_forward:
            direction_parts.append('w')
            max_intensity = active_commands.get('channel_5', ('NEUTRAL', 0))[1]
        elif move_backward:
            direction_parts.append('s')
            max_intensity = active_commands.get('channel_5', ('NEUTRAL', 0))[1]
        elif shift_left:
            direction_parts.append('a')
            max_intensity = active_commands.get('channel_6', ('NEUTRAL', 0))[1]
        elif shift_right:
            direction_parts.append('d')
            max_intensity = active_commands.get('channel_6', ('NEUTRAL', 0))[1]

        # Handle rotation (can combine with movement)
        if rotate_left:
            direction_parts.append('arrowleft')
            max_intensity = max(max_intensity, active_commands.get('channel_3', ('NEUTRAL', 0))[1])
        elif rotate_right:
            direction_parts.append('arrowright')
            max_intensity = max(max_intensity, active_commands.get('channel_3', ('NEUTRAL', 0))[1])

        # Handle tilt (can combine with movement and rotation)
        if tilt_up:
            direction_parts.append('arrowup')
            max_intensity = max(max_intensity, active_commands.get('channel_4', ('NEUTRAL', 0))[1])
        elif tilt_down:
            direction_parts.append('arrowdown')
            max_intensity = max(max_intensity, active_commands.get('channel_4', ('NEUTRAL', 0))[1])

        # Handle special actions (these don't add to direction but are logged)
        special_actions = []
        if look_up:
            special_actions.append('LOOK_UP')
        elif look_down:
            special_actions.append('LOOK_DOWN')
        if trigger_shoot:
            special_actions.append('TRIGGER_SHOOT')
        if squat_down:
            special_actions.append('SQUAT_DOWN')
        elif squat_up:
            special_actions.append('SQUAT_UP')
        if plus_action:
            special_actions.append('PLUS')
        elif minus_action:
            special_actions.append('MINUS')

        # Combine all direction parts
        if direction_parts:
            direction = '+'.join(direction_parts)
            logging.info(f"(control_logic.py): Radio commands {active_commands}: {direction} (intensity: {max_intensity})\n")
            if special_actions:
                logging.info(f"(control_logic.py): Special actions: {special_actions}\n")
            move_direction(direction, frame, max_intensity)
            is_neutral = False
        elif special_actions:
            # Only special actions, no movement
            logging.info(f"(control_logic.py): Special actions only: {special_actions}\n")
            # Handle special actions here
            if 'SQUAT_DOWN' in special_actions:
                squatting_position(1)
                is_neutral = False
            elif 'SQUAT_UP' in special_actions:
                neutral_position(10)
                is_neutral = True
            # TODO: Implement other special actions

    return is_neutral, current_leg


########## EXECUTE KEYBOARD COMMANDS ##########

##### temporary dictionary of commands #####

ADJUSTMENT_FUNCS = {
    'FL': {
        'x+': adjustFL_X,
        'x-': lambda: adjustFL_X(forward=False),
        'y+': adjustFL_Y,
        'y-': lambda: adjustFL_Y(left=False),
        'z+': adjustFL_Z,
        'z-': lambda: adjustFL_Z(up=False),
    },
    'FR': {
        'x+': adjustFR_X,
        'x-': lambda: adjustFR_X(forward=False),
        'y+': adjustFR_Y,
        'y-': lambda: adjustFR_Y(left=False),
        'z+': adjustFR_Z,
        'z-': lambda: adjustFR_Z(up=False),
    },
    'BL': {
        'x+': adjustBL_X,
        'x-': lambda: adjustBL_X(forward=False),
        'y+': adjustBL_Y,
        'y-': lambda: adjustBL_Y(left=False),
        'z+': adjustBL_Z,
        'z-': lambda: adjustBL_Z(up=False),
    },
    'BR': {
        'x+': adjustBR_X,
        'x-': lambda: adjustBR_X(forward=False),
        'y+': adjustBR_Y,
        'y-': lambda: adjustBR_Y(left=False),
        'z+': adjustBR_Z,
        'z-': lambda: adjustBR_Z(up=False),
    }
}


##### run robotic process #####

_perception_loop(CHANNEL_DATA)  # run robot process
