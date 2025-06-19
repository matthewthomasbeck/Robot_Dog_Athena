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

import RPi.GPIO as GPIO # import GPIO library for pin control
import pigpio # import pigpio library for PWM control
import logging # import logging library for debugging
import os # import os library for system commands and log files
import sys
import termios
import tty
import socket

##### import initialization functions #####

from utilities.receiver import * # import PWMDecoder class from initialize_receiver along with functions
from utilities.servos import * # import servo initialization functions and maestro object
from utilities.camera import * # import camera initialization functions
from utilities.opencv import * # import opencv initialization functions
from utilities.internet import * # import internet control functionality

##### import movement functions #####

from movement.standing.standing import * # import standing functions
from movement.walking.forward import * # import walking functions
from movement.fundamental_movement import * # import fundamental movement functions


########## CREATE DEPENDENCIES ##########

##### initialize GPIO and pigpio #####

GPIO.setmode(GPIO.BCM) # set gpio mode to bcm so pins a referred to the same way as the processor refers them
pi = pigpio.pi() # set pigpio object to pi so it can be referred to as pi throughout the script

##### set virtual environment path #####

venv_path = "/home/matthewthomasbeck/.virtualenvs/openvino/bin/activate_this.py" # activate the virtual environment

with open(venv_path) as f: # open the virtual environment path

    exec(f.read(), {'__file__': venv_path}) # execute the virtual environment path

##### set up logging #####

logFile = "/home/matthewthomasbeck/Projects/Robot_Dog/robot_dog.log"

# Rename old log file *after* confirming logger setup
if os.path.exists(logFile):
    os.rename(logFile, f"{logFile}.bak")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# File handler
file_handler = logging.FileHandler(logFile, mode='w')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(file_handler)

# Console handler (optional, for debugging via systemd logs)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

logger.info("Logging setup complete.\n")

logging.info("Starting control_logic.py script...\n") # log the start of the script

##### create different control mode #####

MODE = 'radio'





#########################################
############### RUN ROBOT ###############
#########################################


########## RUN ROBOTIC PROCESS ##########

def runRobot():  # central function that runs the robot

    ##### set vairables #####

    global CHANNEL_DATA
    CHANNEL_DATA = {pin: 1500 for pin in PWM_PINS}  # initialize with neutral values
    decoders = []  # define decoders as empty list
    CURRENT_LEG = 'FL' # default current leg for tuning mode
    IS_NEUTRAL = False # assume robot is not in neutral standing position until neutralStandingPosition() is called

    ##### initialize camera #####

    camera_process = start_camera_process(width=640, height=480, framerate=30)

    if camera_process is None:

        logging.error("(control_logic.py): Failed to start camera process. Exiting...\n")
        exit(1)

    ##### initialize PWM decoders #####

    for pin in PWM_PINS:

        decoder = PWMDecoder(pi, pin, pwmCallback)
        decoders.append(decoder)

    ##### run robotic logic #####

    try: # try to put the robot in a neutral standing position

        set_leg_neutral('FL')
        set_leg_neutral('BR')
        set_leg_neutral('FR')
        set_leg_neutral('BL')
        IS_NEUTRAL = True # set IS_NEUTRAL to True
        time.sleep(3) # wait for 3 seconds

    except Exception as e: # if there is an error, log the error

        logging.error(f"(control_logic.py): Failed to move to neutral standing position in runRobot: {e}\n")

    mjpeg_buffer = b''  # Initialize buffer for MJPEG frames

    try: # try to run the main robotic process

        # Detect mode and maybe start socket server
        #MODE = detect_ssh_and_prompt_mode() # TODO comment this out whenever I don't need to tune

        if MODE.startswith("ssh"):
            server = setup_unix_socket()
            logging.info("Waiting for SSH control client to connect to socket...\n")
            conn, _ = server.accept()
            conn.setblocking(True)
            logging.info("SSH client connected.")

        while True:
            # Read chunk of data from the camera process
            chunk = camera_process.stdout.read(4096)
            if not chunk:
                logging.error("(control_logic.py): Camera process stopped sending data.\n")
                break

            mjpeg_buffer += chunk

            # Attempt to decode a single frame and display
            prev_len = len(mjpeg_buffer)
            mjpeg_buffer = decode_and_show_frame(mjpeg_buffer)

            if mjpeg_buffer is None:
                # If something went very wrong, stop
                logging.warning("(control_logic.py): decode_and_show_frame returned None. Stopping...\n")
                break
            elif len(mjpeg_buffer) == prev_len:
                # Means no complete JPEG was found (incomplete frame)
                # If the buffer grows too large, reset it
                if len(mjpeg_buffer) > 65536:
                    logging.warning("(control_logic.py): MJPEG buffer overflow. Resetting buffer.\n")
                    mjpeg_buffer = b''

            # Check if 'q' was pressed in the imshow window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exiting camera feed display.\n")
                break

            # TODO OLD CODE Handle commands
            #commands = interpretCommands(CHANNEL_DATA)
            #for channel, (action, intensity) in commands.items():
                #IS_NEUTRAL = executeRadioCommands(channel, action, intensity, IS_NEUTRAL)

            if MODE == 'radio':
                commands = interpretCommands(CHANNEL_DATA)
                for channel, (action, intensity) in commands.items():
                    IS_NEUTRAL = executeRadioCommands(channel, action, intensity, IS_NEUTRAL)

            elif MODE.startswith("ssh"):
                try:
                    key = conn.recv(3).decode()
                    if not key:
                        continue
                    if key.startswith('\x1b'):
                        key = key[:3]  # arrow key
                    else:
                        key = key[0]
                    IS_NEUTRAL, CURRENT_LEG = executeKeyboardCommands(
                        key, IS_NEUTRAL, CURRENT_LEG, intensity=5, tune_mode=(MODE == 'ssh-tune'),
                    )
                except Exception as e:
                    logging.error(f"(control_logic.py): Socket read error: {e}\n")

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Exiting...\n")

    except Exception as e:
        logging.error(f"(control_logic.py): Unexpected exception in main loop: {e}\n")
        exit(1)

    finally:
        ##### clean up servos and decoders #####
        disableAllServos()
        for decoder in decoders:
            decoder.cancel()

        ##### close camera #####
        if camera_process.poll() is None:
            camera_process.terminate()
            camera_process.wait()

        cv2.destroyAllWindows()

        ##### clean up GPIO and pigpio #####
        pi.stop()
        GPIO.cleanup()
        closeMaestroConnection(MAESTRO)


########## PWM CALLBACK ##########

def pwmCallback(gpio, pulseWidth): # function to set pulse width to channel data

    CHANNEL_DATA[gpio] = pulseWidth # set channel data to pulse width


########## INTERPRET COMMANDS ##########

def executeRadioCommands(channel, action, intensity, IS_NEUTRAL): # function to interpret commands from channel data and do things

    ##### squat channel 2 #####

    if channel == 'channel-2':
        if action == 'NEUTRAL' or action == 'SQUAT DOWN':
            if action == 'SQUAT DOWN':
                pass
                # function to squat
        elif action == 'SQUAT UP':
                # function to neutral
                print(f"{channel}: {action}")

    ##### tilt channel 0 #####

    if channel == 'channel-0':
        if action == 'TILT DOWN':
            print(f"{channel}: {action}")
        elif action == 'TILT UP':
            print(f"{channel}: {action}")

    ##### trigger channel 1 #####

    elif channel == 'channel-1':

        if action == 'TRIGGER SHOOT':
            print(f"{channel}: {action}")

    ##### rotation channel 3 #####

    elif channel == 'channel-3':

        if action == 'ROTATE LEFT':

            print(f"{channel}: {action}")

            try:
                IS_NEUTRAL = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to rotate left in executeCommands: {e}\n")

        elif action == 'NEUTRAL':

            IS_NEUTRAL = True

        elif action == 'ROTATE RIGHT':

            print(f"{channel}: {action}")

            try:
                IS_NEUTRAL = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to rotate right in executeCommands: {e}\n")

    ##### look channel 4 #####

    elif channel == 'channel-4':

        if action == 'LOOK DOWN':

            logging.info(f"{channel}: {action}")

            try:
                IS_NEUTRAL = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to look down in executeCommands: {e}\n")

        elif action == 'NEUTRAL':

            IS_NEUTRAL = True

        elif action == 'LOOK UP':

            logging.info(f"{channel}: {action}")

            try:
                IS_NEUTRAL = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to look up in executeCommands: {e}\n")

    ##### move channel 5 #####

    elif channel == 'channel-5':

        if action == 'MOVE FORWARD':

            logging.info(f"{channel}: {action}")

            try:
                trot_forward(intensity)
                IS_NEUTRAL = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to move forward in executeCommands: {e}\n")

        elif action == 'NEUTRAL':

            try:

                if IS_NEUTRAL == False:

                    set_leg_neutral('FL')
                    set_leg_neutral('BR')
                    set_leg_neutral('FR')
                    set_leg_neutral('BL')
                    IS_NEUTRAL = True

            except Exception as e:

                logging.error(f"(control_logic.py): Failed to move to neutral standing position in executeCommands: {e}\n")

        elif action == 'MOVE BACKWARD':

            logging.info(f"{channel}: {action}")

            try:
                IS_NEUTRAL = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to move backward in executeCommands: {e}\n")

    ##### shift channel 6 #####

    elif channel == 'channel-6':

        if action == 'SHIFT LEFT':

            logging.info(f"{channel}: {action}")

            try:
                IS_NEUTRAL = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to shift left in executeCommands: {e}\n")

        elif action == 'NEUTRAL':

            IS_NEUTRAL = True

        elif action == 'SHIFT RIGHT':

            logging.info(f"{channel}: {action}")

            try:
                IS_NEUTRAL = False

            except Exception as e:
                logging.error(f"(control_logic.py): Failed to shift right in executeCommands: {e}\n")

    ##### extra channel 7 #####

    elif channel == 'channel-7':
        if action == '+':
            print(f"{channel}: {action}")
        elif action == '-':
            print(f"{channel}: {action}")

    ##### update is neutral standing #####

    return IS_NEUTRAL # return neutral standing boolean for position awareness


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

def executeKeyboardCommands(key, IS_NEUTRAL, CURRENT_LEG, intensity=5, tune_mode=False):

    if tune_mode:

        if key == 'q': # x axis positive
            ADJUSTMENT_FUNCS[CURRENT_LEG]['x+']()
            IS_NEUTRAL = False

        elif key == 'a': # x axis negative
            ADJUSTMENT_FUNCS[CURRENT_LEG]['x-']()
            IS_NEUTRAL = False

        elif key == 'w': # y axis positive
            ADJUSTMENT_FUNCS[CURRENT_LEG]['y+']()
            IS_NEUTRAL = False

        elif key == 's': # y axis negative
            ADJUSTMENT_FUNCS[CURRENT_LEG]['y-']()
            IS_NEUTRAL = False

        elif key == 'e': # z axis positive
            ADJUSTMENT_FUNCS[CURRENT_LEG]['z+']()
            IS_NEUTRAL = False

        elif key == 'd': # z axis negative
            ADJUSTMENT_FUNCS[CURRENT_LEG]['z-']()
            IS_NEUTRAL = False

        elif key == '1': # set current leg to front left

            CURRENT_LEG = 'FL'  # Set current leg to front left
            IS_NEUTRAL = False

        elif key == '2': # set current leg to front right

            CURRENT_LEG = 'FR'  # Set current leg to front right
            IS_NEUTRAL = False

        elif key == '3': # set current leg to back left

            CURRENT_LEG = 'BL'  # Set current leg to back left
            IS_NEUTRAL = False

        elif key == '4': # set current leg to back right

            CURRENT_LEG = 'BR'  # Set current leg to back right
            IS_NEUTRAL = False

        elif key == 'r': # right-leading swing


            IS_NEUTRAL = False

        elif key == 'l': # left-leading swing


            IS_NEUTRAL = False

        elif key == 'n':
            if not IS_NEUTRAL:
                set_leg_neutral('FL')
                set_leg_neutral('BR')
                set_leg_neutral('FR')
                set_leg_neutral('BL')
                IS_NEUTRAL = True

    else:  # Normal operation mode

        if key == 'q':
            logging.info("Exiting control logic.")
            return IS_NEUTRAL  # Exit condition

        elif key == 'w':  # Move forward
            trot_forward(intensity)
            IS_NEUTRAL = False

        elif key == 's':  # Move backward
            # trotBackward(intensity)
            IS_NEUTRAL = False

        elif key == 'a':  # Shift left
            # trotLeft(intensity)
            IS_NEUTRAL = False

        elif key == 'd':  # Shift right
            # trotRight(intensity)
            IS_NEUTRAL = False

        elif key == '\x1b[C':  # Rotate right
            # rotateRight(intensity)
            IS_NEUTRAL = False

        elif key == '\x1b[D':  # Rotate left
            # rotateLeft(intensity)
            IS_NEUTRAL = False

        elif key == '\x1b[A':  # Look up
            # adjustBL_Z(up=False)
            IS_NEUTRAL = False

        elif key == '\x1b[B':  # Look down
            # adjustBL_Z(up=True)
            IS_NEUTRAL = False

        elif key == 'i':  # Tilt up
            # adjustBL_Y(left=False)
            IS_NEUTRAL = False

        elif key == 'k':  # Tilt down
            # adjustBL_Y(left=True)
            IS_NEUTRAL = False

        elif key == 'n':  # Neutral position
            if not IS_NEUTRAL:
                set_leg_neutral('FL')
                set_leg_neutral('BR')
                set_leg_neutral('FR')
                set_leg_neutral('BL')
                IS_NEUTRAL = True

    return IS_NEUTRAL, CURRENT_LEG  # Return updated neutral standing state


########## RUN ROBOTIC PROCESS ##########

runRobot() # run the robot process
