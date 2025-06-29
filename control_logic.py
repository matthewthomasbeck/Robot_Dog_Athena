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

##### import necessary utilities #####

from utilities.log import initialize_logging  # import logging setup
from utilities.receiver import initialize_receiver, interpret_commands  # import receiver initialization functions
from utilities.camera import initialize_camera  # import to start camera logic
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
SOCK = internet.initialize_ec2_socket()  # initialize EC2 socket connection
COMMAND_QUEUE = internet.initialize_command_queue(SOCK)  # initialize command queue for socket communication

##### create different control mode #####

MODE = 'web'  # default mode is radio control, can be changed to 'ssh' or 'ssh-tune' for SSH control
logging.debug("(control_logic.py): Starting control_logic.py script...\n")  # log start of script


#########################################
############### RUN ROBOT ###############
#########################################


########## RUN ROBOTIC PROCESS ##########

def _run_robot(CHANNEL_DATA):  # central function that runs robot

    ##### set/initialize variables #####

    is_neutral = False  # assume robot is not in neutral standing position until neutralStandingPosition() is called
    current_leg = 'FL'  # default current leg for tuning mode
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
        is_neutral = True  # set is_neutral to True
        time.sleep(3)  # wait for 3 seconds

    except Exception as e:  # if there is an error, log error

        logging.error(f"(control_logic.py): Failed to move to neutral standing position in runRobot: {e}\n")

    try:  # try to run main robotic process

        # MODE = detect_ssh_and_prompt_mode() # detect mode to possibly start socket server TODO comment this out whenever I don't need to tune

        ##### stream video, run inference, and control the robot #####

        while True:

            # TODO get rid of these lines if delay unbearable
            # start_time = time.time()  # start time to measure actions per second

            # let run_inference be true or false if I want to avoid running inference or not (it's laggy as hell)
            mjpeg_buffer, frame_data = run_inference(
                COMPILED_MODEL,
                INPUT_LAYER,
                OUTPUT_LAYER,
                CAMERA_PROCESS,
                mjpeg_buffer,
                run_inference=False
            )

            internet.stream_to_ec2(SOCK, frame_data)  # stream frame data to ec2 instance

            ##### decode and execute commands #####

            if MODE.startswith("ssh"):
                server = internet.initialize_socket()
                logging.debug("(control_logic.py): Waiting for SSH control client to connect to socket...\n")
                conn, _ = server.accept()
                conn.setblocking(True)
                logging.info("(control_logic.py): SSH client connected.\n")

            # TODO OLD CODE Handle commands
            # commands = interpret_commands(CHANNEL_DATA)
            # for channel, (action, intensity) in commands.items():
            # is_neutral = executeRadioCommands(channel, action, intensity, is_neutral)

            if MODE == 'radio':  # if mode is radio...
                commands = interpret_commands(CHANNEL_DATA)  # interpret commands from CHANNEL_DATA from R/C receiver

                for channel, (action, intensity) in commands.items():  # loop through each channel and its action
                    is_neutral = _execute_radio_commands(channel, action, intensity, is_neutral)  # run radio commands

            elif MODE.startswith("ssh"):  # if mode is SSH...
                try:  # attempt to read from SSH socket connection
                    key = conn.recv(3).decode()  # read up to 3 bytes from socket
                    if not key:  # if no data received...
                        continue
                    if key.startswith('\x1b'):  # if key starts with escape character...
                        key = key[:3]  # arrow key
                    else:  # if some other character...
                        key = key[0]

                    is_neutral, current_leg = _execute_keyboard_commands(  # use keys to execute commands
                        key,
                        is_neutral,
                        current_leg,
                        intensity=10,
                        tune_mode=(MODE == 'ssh-tune'),
                    )

                except Exception as e:  # if there is an error reading from socket...
                    logging.error(f"(control_logic.py): Socket read error: {e}\n")

            elif MODE == 'web':
                if not COMMAND_QUEUE.empty():
                    command = COMMAND_QUEUE.get()
                    logging.info(f"(control_logic.py): Received command from web: {command}\n")

                    # Execute the command using existing keyboard command logic
                    #is_neutral, current_leg = _execute_keyboard_commands(
                        #command.strip(),  # Remove any whitespace/newlines
                        #is_neutral,
                        #current_leg,
                        #intensity=10,
                        #tune_mode=False
                    #)

            ##### wait to maintain global action rate and not outpace the camera #####

            # TODO get rid of these lines if delay unbearable
            # elapsed = time.time() - start_time # calculate elapsed time for actions
            # sleep_time = max(0, (1 / LOOP_RATE) - elapsed) # calculate time to sleep to maintain loop rate
            # if sleep_time > 0:  # if sleep time is greater than 0...
            # time.sleep(sleep_time) # only sleep if outpacing the camera

    except KeyboardInterrupt:  # if user ends program...
        logging.info("(control_logic.py): KeyboardInterrupt received, exiting.\n")

    except Exception as e:  # if something breaks and only God knows what it is...
        logging.error(f"(control_logic.py): Unexpected exception in main loop: {e}\n")
        exit(1)


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

_run_robot(CHANNEL_DATA)  # run robot process
