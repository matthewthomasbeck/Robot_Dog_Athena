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
from utilities.opencv import load_and_compile_model, run_inference  # load inference functions
import utilities.internet as internet  # dynamically import internet utilities to be constantly updated (sending frames)

##### import movement functions #####

from movement.standing.standing import *  # import standing functions
from movement.walking.forward import *  # import walking functions
from movement.fundamental_movement import *  # import fundamental movement functions

##### import config #####

from utilities.config import LOOP_RATE_HZ  # import loop rate for actions per second

########## CREATE DEPENDENCIES ##########

##### initialize all utilities #####

LOGGER = initialize_logging()  # set up logging
CAMERA_PROCESS = initialize_camera()  # create camera process
COMPILED_MODEL, INPUT_LAYER, OUTPUT_LAYER = load_and_compile_model()  # load and compile model
CHANNEL_DATA = initialize_receiver()  # get pigpio instance, decoders, and channel data
SOCK = internet.initialize_backend_socket()  # initialize EC2 socket connection
COMMAND_QUEUE = internet.initialize_command_queue(SOCK)  # initialize command queue for socket communication
FRAME_QUEUE = queue.Queue(maxsize=1)

##### create different control mode #####

MODE = 'web'  # default mode is radio control, can be changed to 'radio' or 'web' for variable control
IS_NEUTRAL = False  # set global neutral standing boolean
CURRENT_LEG = 'FL'  # set global current leg
logging.debug("(control_logic.py): Starting control_logic.py script...\n")  # log start of script


#########################################
############### RUN ROBOT ###############
#########################################


########## RUN ROBOTIC PROCESS ##########

def _perception_loop(CHANNEL_DATA):  # central function that runs robot

    global IS_NEUTRAL, CURRENT_LEG

    ##### set/initialize variables #####

    mjpeg_buffer = b''  # initialize buffer for MJPEG frames
    LOOP_RATE = LOOP_RATE_HZ * 3  # calculate frame interval based on loop rate times 3 to keep up with camera

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

            # TODO get rid of these lines if delay unbearable
            # start_time = time.time()  # start time to measure actions per second

            mjpeg_buffer, frame_data, frame = decode_frame(CAMERA_PROCESS, mjpeg_buffer)
            internet.stream_to_backend(SOCK, frame_data)  # stream frame data to ec2 instance
            command = None

            if MODE == 'web' and COMMAND_QUEUE is not None and not COMMAND_QUEUE.empty():
                command = COMMAND_QUEUE.get()

                logging.info(f"(control_logic.py): Received command '{command}' from queue.\n")

            # LEGACY RADIO COMMAND CODE, UNDER NO CIRCUMSTANCES REMOVE WHATSOEVER (I will get around to renewing radio support for override)
            #if MODE == 'radio':
                #commands = interpret_commands(CHANNEL_DATA)
                #for channel, (action, intensity) in commands.items():
                    #is_neutral = _execute_radio_commands(channel, action, intensity, is_neutral)

            if command and IS_NEUTRAL:

                logging.info(f"(control_logic.py): Running command: {command}...\n")
                IS_NEUTRAL = False  # block new commands until movement is complete and neutral standing is true
                threading.Thread(target=_handle_command, args=(command, frame), daemon=True).start()

    except KeyboardInterrupt:  # if user ends program...
        logging.info("(control_logic.py): KeyboardInterrupt received, exiting.\n")

    except Exception as e:  # if something breaks and only God knows what it is...
        logging.error(f"(control_logic.py): Unexpected exception in main loop: {e}\n")
        exit(1)


########## HANDLE COMMANDS ##########

def _handle_command(command, frame):

    logging.debug(f"(control_logic.py): Threading command: {command}...\n")

    global IS_NEUTRAL, CURRENT_LEG

    run_inference( # dont run inference for now
        COMPILED_MODEL,
        INPUT_LAYER,
        OUTPUT_LAYER,
        frame,
        run_inference=False
    )

    logging.info(f"(control_logic.py): Ran inference for command: {command}\n")

    if MODE == 'radio':
        try:
            logging.debug(f"(control_logic.py): Executing radio command: {command}...\n")
            logging.info(f"(control_logic.py): Executed radio command: {command}\n")

        except Exception as e:
            logging.error(f"(control_logic.py): Failed to execute radio command: {e}\n")

        pass

    elif MODE == 'web':
        try:
            logging.debug(f"(control_logic.py): Executing keyboard command: {command}...\n")
            IS_NEUTRAL, CURRENT_LEG = _execute_keyboard_commands(
                command.strip(),
                IS_NEUTRAL,
                CURRENT_LEG,
                intensity=10,
                tune_mode=False
            )
            logging.info(f"(control_logic.py): Executed keyboard command: {command}\n")

        except Exception as e:
            logging.error(f"(control_logic.py): Failed to execute keyboard command: {e}\n")

    IS_NEUTRAL = True


########## INTERPRET COMMANDS ##########

# function to interpret commands from channel data and do things
def _execute_radio_commands(channel, action, intensity, is_neutral):

    ##### squat channel 2 #####

    if channel == 'channel_2':
        if action == 'NEUTRAL' or action == 'SQUAT DOWN':
            if action == 'SQUAT DOWN':
                pass
                # function to squat
        elif action == 'SQUAT UP':
            # function to neutral
            logging.info(f"(control_logic.py): {channel}: {action}\n")

    ##### tilt channel 0 #####

    if channel == 'channel_0':
        if action == 'TILT DOWN':
            logging.info(f"(control_logic.py): {channel}: {action}\n")
        elif action == 'TILT UP':
            logging.info(f"(control_logic.py): {channel}: {action}\n")

    ##### trigger channel 1 #####

    elif channel == 'channel_1':

        if action == 'TRIGGER SHOOT':
            logging.info(f"(control_logic.py): {channel}: {action}\n")

    ##### rotation channel 3 #####

    elif channel == 'channel_3':

        if action == 'ROTATE LEFT':

            logging.info(f"(control_logic.py): {channel}: {action}\n")

            try:
                is_neutral = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to rotate left in executeCommands: {e}\n")

        elif action == 'NEUTRAL':

            is_neutral = True

        elif action == 'ROTATE RIGHT':

            logging.info(f"(control_logic.py): {channel}: {action}\n")

            try:
                is_neutral = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to rotate right in executeCommands: {e}\n")

    ##### look channel 4 #####

    elif channel == 'channel_4':

        if action == 'LOOK DOWN':

            logging.info(f"(control_logic.py): {channel}: {action}\n")

            try:
                is_neutral = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to look down in executeCommands: {e}\n")

        elif action == 'NEUTRAL':

            is_neutral = True

        elif action == 'LOOK UP':

            logging.info(f"(control_logic.py): {channel}: {action}\n")

            try:
                is_neutral = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to look up in executeCommands: {e}\n")

    ##### move channel 5 #####

    elif channel == 'channel_5':

        if action == 'MOVE FORWARD':

            logging.info(f"(control_logic.py): {channel}: {action}\n")

            try:
                trot_forward(intensity)
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

            logging.info(f"(control_logic.py): {channel}: {action}\n")

            try:
                is_neutral = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to move backward in executeCommands: {e}\n")

    ##### shift channel 6 #####

    elif channel == 'channel_6':

        if action == 'SHIFT LEFT':

            logging.info(f"(control_logic.py): {channel}: {action}\n")

            try:
                is_neutral = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to shift left in executeCommands: {e}\n")

        elif action == 'NEUTRAL':

            is_neutral = True

        elif action == 'SHIFT RIGHT':

            logging.info(f"(control_logic.py): {channel}: {action}\n")

            try:
                is_neutral = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to shift right in executeCommands: {e}\n")

    ##### extra channel 7 #####

    elif channel == 'channel_7':
        if action == '+':
            logging.info(f"(control_logic.py): {channel}: {action}\n")
        elif action == '-':
            logging.info(f"(control_logic.py): {channel}: {action}\n")

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


##### keyboard commands for tuning mode and normal operation #####

def _execute_keyboard_commands(key, is_neutral, current_leg, intensity=10, tune_mode=False):
    if tune_mode:

        if key == 'q':  # x axis positive
            ADJUSTMENT_FUNCS[current_leg]['x+']()
            is_neutral = False

        elif key == 'a':  # x axis negative
            ADJUSTMENT_FUNCS[current_leg]['x-']()
            is_neutral = False

        elif key == 'w':  # y axis positive
            ADJUSTMENT_FUNCS[current_leg]['y+']()
            is_neutral = False

        elif key == 's':  # y axis negative
            ADJUSTMENT_FUNCS[current_leg]['y-']()
            is_neutral = False

        elif key == 'e':  # z axis positive
            ADJUSTMENT_FUNCS[current_leg]['z+']()
            is_neutral = False

        elif key == 'd':  # z axis negative
            ADJUSTMENT_FUNCS[current_leg]['z-']()
            is_neutral = False

        elif key == '1':  # set current leg to front left

            current_leg = 'FL'  # Set current leg to front left
            is_neutral = False

        elif key == '2':  # set current leg to front right

            current_leg = 'FR'  # Set current leg to front right
            is_neutral = False

        elif key == '3':  # set current leg to back left

            current_leg = 'BL'  # Set current leg to back left
            is_neutral = False

        elif key == '4':  # set current leg to back right

            current_leg = 'BR'  # Set current leg to back right
            is_neutral = False

        elif key == 'n':
            if not is_neutral:
                neutral_position(10)
                is_neutral = True

    else:  # Normal operation mode

        if key == 'q':
            logging.info("(control_logic.py): Exiting control logic.\n")
            return is_neutral  # Exit condition

        elif key == 'w':  # Move forward
            trot_forward(intensity)
            is_neutral = False

        elif key == 's':  # Move backward
            # trotBackward(intensity)
            is_neutral = False

        elif key == 'a':  # Shift left
            # trotLeft(intensity)
            is_neutral = False

        elif key == 'd':  # Shift right
            # trotRight(intensity)
            is_neutral = False

        elif key == '\x1b[C':  # Rotate right
            # rotateRight(intensity)
            is_neutral = False

        elif key == '\x1b[D':  # Rotate left
            # rotateLeft(intensity)
            is_neutral = False

        elif key == '\x1b[A':  # Look up
            # adjustBL_Z(up=False)
            is_neutral = False

        elif key == '\x1b[B':  # Look down
            # adjustBL_Z(up=True)
            is_neutral = False

        elif key == 'i':  # Tilt up
            # adjustBL_Y(left=False)
            is_neutral = False

        elif key == 'k':  # Tilt down
            # adjustBL_Y(left=True)
            is_neutral = False

        elif key == ' ':  # Lie down
            squatting_position(1)
            is_neutral = False

        elif key == 'n':  # Neutral position
            if not is_neutral:
                neutral_position(10)
                is_neutral = True

    return is_neutral, current_leg  # Return updated neutral standing state


##### run robotic process #####

_perception_loop(CHANNEL_DATA)  # run robot process
