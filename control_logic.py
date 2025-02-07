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


########## CREATE DEPENDENCIES ##########

##### initialize GPIO and pigpio #####

GPIO.setmode(GPIO.BCM) # set gpio mode to bcm so pins a referred to the same way as the processor refers them
pi = pigpio.pi() # set pigpio object to pi so it can be referred to as pi throughout the script

##### set virtual environment path #####

venv_path = "/home/matthewthomasbeck/.virtualenvs/openvino/bin/activate_this.py" # activate the virtual environment

with open(venv_path) as f: # open the virtual environment path

    exec(f.read(), {'__file__': venv_path}) # execute the virtual environment path

##### set up logging #####

logFile = "/home/matthewthomasbeck/Projects/Robot_Dog/robot_dog.log" # set the log file path

if os.path.exists(logFile): # if the old log file exists...

    os.rename(logFile, f"{logFile}.bak") # rename the old log file with a timestamp

logging.basicConfig( # configure the logging module to write mode, overwriting the old log file

    filename=logFile,
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Starting control_logic.py script...\n") # log the start of the script


########## IMPORT DEPENDENCIES ##########

##### import initialization functions #####

from initialize.initialize_receiver import * # import PWMDecoder class from initialize_receiver along with functions
from initialize.initialize_servos import * # import servo initialization functions and maestro object
from initialize.initialize_camera import * # import camera initialization functions
from initialize.initialize_opencv import * # import opencv initialization functions

##### import movement functions #####

from movement.standing.standing_inplace import * # import standing functions
from movement.walking.manual_walking import * # import walking functions





################################################
############### COMMAND REQUESTS ###############
################################################


########## RUN ROBOTIC PROCESS ##########

def runRobot():  # central function that runs the robot

    ##### set vairables #####

    global CHANNEL_DATA
    CHANNEL_DATA = {pin: 1500 for pin in PWM_PINS}  # initialize with neutral values
    decoders = []  # define decoders as empty list
    IS_NEUTRAL = False # assume robot is not in neutral standing position until neutralStandingPosition() is called

    ##### initialize camera #####

    camera_process = start_camera_process(width=640, height=480, framerate=30)

    if camera_process is None:

        logging.error("ERROR (control_logic.py): Failed to start camera process. Exiting...\n")
        exit(1)

    ##### initialize PWM decoders #####

    for pin in PWM_PINS:

        decoder = PWMDecoder(pi, pin, pwmCallback)
        decoders.append(decoder)

    ##### run robotic logic #####

    try: # try to put the robot in a neutral standing position

        neutralStandingPosition() # move to neutral standing position
        IS_NEUTRAL = True # set IS_NEUTRAL to True
        time.sleep(3) # wait for 3 seconds

    except Exception as e: # if there is an error, log the error

        logging.error(f"ERROR (control_logic.py): Failed to move to neutral standing position in runRobot: {e}\n")

    mjpeg_buffer = b''  # Initialize buffer for MJPEG frames

    try:
        while True:
            # Read chunk of data from the camera process
            chunk = camera_process.stdout.read(4096)
            if not chunk:
                logging.error("ERROR (control_logic.py): Camera process stopped sending data.")
                break

            mjpeg_buffer += chunk

            # Attempt to decode a single frame and display
            prev_len = len(mjpeg_buffer)
            mjpeg_buffer = decode_and_show_frame(mjpeg_buffer)

            if mjpeg_buffer is None:
                # If something went very wrong, stop
                logging.warning("WARNING (control_logic.py): decode_and_show_frame returned None. Stopping.")
                break
            elif len(mjpeg_buffer) == prev_len:
                # Means no complete JPEG was found (incomplete frame)
                # If the buffer grows too large, reset it
                if len(mjpeg_buffer) > 65536:
                    logging.warning("WARNING (control_logic.py): MJPEG buffer overflow. Resetting buffer.")
                    mjpeg_buffer = b''

            # Check if 'q' was pressed in the imshow window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exiting camera feed display.")
                break

            # Handle commands
            commands = interpretCommands(CHANNEL_DATA)
            for channel, (action, intensity) in commands.items():
                IS_NEUTRAL = executeCommands(channel, action, intensity, IS_NEUTRAL)

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Exiting...\n")

    except Exception as e:
        logging.error(f"ERROR (control_logic.py): Unexpected exception in main loop: {e}\n")
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

def executeCommands(channel, action, intensity, IS_NEUTRAL): # function to interpret commands from channel data and do things

    ##### squat channel 2 #####

    if channel == 'channel-2':
        if action == 'NEUTRAL' or action == 'SQUAT DOWN':
            if action == 'SQUAT DOWN':
                pass
                #disableAllServos()
        elif action == 'SQUAT UP':
                #eutralStandingPosition()
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
        elif action == 'ROTATE RIGHT':
            print(f"{channel}: {action}")

    ##### look channel 4 #####

    elif channel == 'channel-4':
        if action == 'LOOK DOWN':
            print(f"{channel}: {action}")
        elif action == 'LOOK UP':
            print(f"{channel}: {action}")

    ##### move channel 5 #####

    elif channel == 'channel-5':

        if action == 'MOVE FORWARD':

            print(f"{channel}: {action}")

            try:

                oscillateLegs(intensity)
                IS_NEUTRAL = False

            except Exception as e:

                logging.error(f"ERROR (control_logic.py): Failed to move forward in executeCommands: {e}\n")

        elif action == 'NEUTRAL':

            try:

                if IS_NEUTRAL == False:

                    neutralStandingPosition()
                    time.sleep(0.1)
                    IS_NEUTRAL = True

            except Exception as e:

                logging.error(f"ERROR (control_logic.py): Failed to move to neutral standing position in executeCommands: {e}\n")

        elif action == 'MOVE BACKWARD':

            print(f"{channel}: {action}")

    ##### shift channel 6 #####

    elif channel == 'channel-6':
        if action == 'SHIFT LEFT':
            print(f"{channel}: {action}")
        elif action == 'SHIFT RIGHT':
            print(f"{channel}: {action}")

    ##### extra channel 7 #####

    elif channel == 'channel-7':
        if action == '+':
            print(f"{channel}: {action}")
        elif action == '-':
            print(f"{channel}: {action}")

    ##### update is neutral standing #####

    return IS_NEUTRAL # return neutral standing boolean for position awareness


########## RUN ROBOTIC PROCESS ##########

##### complete all initialization and begin robotic process #####

runRobot() # initialize receiver
