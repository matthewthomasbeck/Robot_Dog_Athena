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

import time  # import time library for gait timing
import threading
import queue

##### import necessary utilities #####

from utilities.log import initialize_logging  # import logging setup
from utilities.receiver import initialize_receiver, interpret_commands  # import receiver initialization functions
from utilities.camera import initialize_camera, decode_frame  # import to start camera logic
import utilities.internet as internet  # dynamically import internet utilities to be constantly updated (sending frames)

##### import movement functions #####

from movement.fundamental_movement import *  # import fundamental movement functions
from movement.standing.standing import *  # import standing functions
from movement.walking.forward import *  # import walking functions

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

            if command and IS_COMPLETE: # if command presend and movement complete...

                logging.debug(f"(control_logic.py): Running command: {command}...\n")
                threading.Thread(target=_handle_command, args=(command, frame), daemon=True).start()

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
    else:
        keys = []

    if MODE == 'radio':
        try:
            logging.debug(f"(control_logic.py): Executing radio command: {command}...\n")
            logging.info(f"(control_logic.py): Executed radio command: {command}\n")
            IS_NEUTRAL = False # TODO this is temporary, replace with actual radio command later
            IS_COMPLETE = True
        except Exception as e:
            logging.error(f"(control_logic.py): Failed to execute radio command: {e}\n")
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
                intensity=10,
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
        # Movement direction logic
        direction = None

        # Movement (WASD and diagonals)
        if 'w' in keys and 'd' in keys:
            direction = 'FORWARD_RIGHT'
        elif 'w' in keys and 'a' in keys:
            direction = 'FORWARD_LEFT'
        elif 's' in keys and 'd' in keys:
            direction = 'BACKWARD_RIGHT'
        elif 's' in keys and 'a' in keys:
            direction = 'BACKWARD_LEFT'
        elif 'w' in keys:
            direction = 'FORWARD'
        elif 's' in keys:
            direction = 'BACKWARD'
        elif 'a' in keys:
            direction = 'LEFT'
        elif 'd' in keys:
            direction = 'RIGHT'

        # Rotation (arrow keys)
        if 'arrowleft' in keys and 'arrowright' in keys:
            # Cancel out, do nothing
            pass
        elif 'arrowleft' in keys:
            direction = 'ROTATE_LEFT'
        elif 'arrowright' in keys:
            direction = 'ROTATE_RIGHT'

        # Tilt (arrow up/down)
        if 'arrowup' in keys and 'arrowdown' in keys:
            # Cancel out, do nothing
            pass
        elif 'arrowup' in keys:
            direction = 'TILT_UP'
        elif 'arrowdown' in keys:
            direction = 'TILT_DOWN'

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
            trot_forward(intensity)
            #move_direction(direction, frame, intensity)
            is_neutral = False
        else:
            logging.warning(f"(control_logic.py): Invalid command: {keys}\n")

    else:
        # Tuning mode (unchanged, but now keys is a list)
        for key in keys:
            # ... (your tuning logic here, similar to before)
            pass

    return is_neutral, current_leg


########## INTERPRET COMMANDS ##########

# function to interpret commands from channel data and do things
def _execute_radio_commands(channel, action, frame, intensity, is_neutral):

    ##### squat channel 2 #####

    if channel == 'channel_2':
        if action == 'NEUTRAL' or action == 'SQUAT DOWN':
            if action == 'SQUAT DOWN':
                # TODO I can hand-do this
                pass
        elif action == 'SQUAT UP': # returns to neutral
            # TODO I can hand-do this
            pass

    ##### tilt channel 0 #####

    if channel == 'channel_0':
        if action == 'LOOK DOWN':
            # logging.info(f"(control_logic.py): {channel}:{action}\n")
            # TODO I can hand-do this
            pass
        elif action == 'LOOK UP':
            # logging.info(f"(control_logic.py): {channel}:{action}\n")
            # TODO I can hand-do this
            pass

    ##### trigger channel 1 #####

    elif channel == 'channel_1':
        if action == 'TRIGGER SHOOT':
            # logging.info(f"(control_logic.py): {channel}:{action}\n")
            # TODO no hardware support for this yet
            pass

    ##### rotation channel 3 #####

    elif channel == 'channel_3':

        if action == 'ROTATE LEFT':
            logging.info(f"(control_logic.py): {channel}:{action}\n")
            try:
                move_direction('ROTATE LEFT', frame, intensity)
                is_neutral = False
            except Exception as e:
                logging.error(f"(control_logic.py): Failed to rotate left in executeCommands: {e}\n")

        elif action == 'NEUTRAL':
            is_neutral = True

        elif action == 'ROTATE RIGHT':
            logging.info(f"(control_logic.py): {channel}:{action}\n")
            try:
                move_direction('ROTATE RIGHT', frame, intensity)
                is_neutral = False
            except Exception as e:
                logging.error(f"(control_logic.py): Failed to rotate right in executeCommands: {e}\n")

    ##### look channel 4 #####

    elif channel == 'channel_4':

        if action == 'TILT DOWN':
            logging.info(f"(control_logic.py): {channel}:{action}\n")
            try:
                move_direction('TILT DOWN', frame, intensity)
                is_neutral = False
                pass
            except Exception as e:
                logging.error(f"(control_logic.py): Failed to look down in executeCommands: {e}\n")

        elif action == 'NEUTRAL':
            is_neutral = True
            pass

        elif action == 'TILT UP':
            logging.info(f"(control_logic.py): {channel}:{action}\n")
            try:
                move_direction('TILT UP', frame, intensity)
                is_neutral = False
                pass
            except Exception as e:
                logging.error(f"(control_logic.py): Failed to look up in executeCommands: {e}\n")

    ##### move channel 5 #####

    elif channel == 'channel_5':

        if action == 'MOVE FORWARD':
            logging.info(f"(control_logic.py): {channel}:{action}\n")
            try:
                move_direction('MOVE FORWARD', frame, intensity)
                is_neutral = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to move forward in executeCommands: {e}\n")

        elif action == 'NEUTRAL':
            try:
                if is_neutral == False:
                    neutral_position(10)
                    is_neutral = True
            except Exception as e:
                logging.error(
                    f"(control_logic.py): Failed to move to neutral standing position in executeCommands: {e}\n")

        elif action == 'MOVE BACKWARD':
            logging.info(f"(control_logic.py): {channel}:{action}\n")
            try:
                move_direction('MOVE BACKWARD', frame, intensity)
                is_neutral = False
            except Exception as e:
                logging.error(f"(control_logic.py): Failed to move backward in executeCommands: {e}\n")

    ##### shift channel 6 #####

    elif channel == 'channel_6':

        if action == 'SHIFT LEFT':
            logging.info(f"(control_logic.py): {channel}:{action}\n")
            try:
                move_direction('SHIFT LEFT', frame, intensity)
                is_neutral = False
            except Exception as e:
                logging.error(f"(control_logic.py): Failed to shift left in executeCommands: {e}\n")

        elif action == 'NEUTRAL':
            is_neutral = True

        elif action == 'SHIFT RIGHT':
            logging.info(f"(control_logic.py): {channel}:{action}\n")
            try:
                move_direction('SHIFT RIGHT', frame, intensity)
                is_neutral = False
            except Exception as e:
                logging.error(f"(control_logic.py): Failed to shift right in executeCommands: {e}\n")

    ##### extra channel 7 #####

    elif channel == 'channel_7':
        if action == '+':
            logging.info(f"(control_logic.py): {channel}:{action}\n")
        elif action == '-':
            logging.info(f"(control_logic.py): {channel}:{action}\n")

    ##### update is neutral standing #####

    return is_neutral  # return neutral standing boolean for position awareness


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
