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
    global CHANNEL_DATA
    CHANNEL_DATA = {pin: 1500 for pin in PWM_PINS}  # initialize with neutral values
    decoders = []  # define decoders as empty list

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
    neutralStandingPosition()
    time.sleep(3)

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
            for command, action in commands.items():
                executeCommands(command, action)

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

arc_reduction = 0

def executeCommands(command, action): # function to interpret commands from channel data and do things

    ##### squat channel 2 #####

    if command == 'channel-2':
        if action == 'NEUTRAL' or action == 'SQUAT DOWN':
            if action == 'SQUAT DOWN':
                pass
                #disableAllServos()
        elif action == 'SQUAT UP':
                #eutralStandingPosition()
                print(f"{command}: {action}")

    ##### tilt channel 0 #####

    if command == 'channel-0':
        if action == 'TILT DOWN':
            print(f"{command}: {action}")
        elif action == 'TILT UP':
            print(f"{command}: {action}")

    ##### trigger channel 1 #####

    elif command == 'channel-1':
        if action == 'TRIGGER SHOOT':
            print(f"{command}: {action}")

    ##### rotation channel 3 #####

    elif command == 'channel-3':
        if action == 'ROTATE LEFT':
            print(f"{command}: {action}")
        elif action == 'ROTATE RIGHT':
            print(f"{command}: {action}")

    ##### look channel 4 #####

    elif command == 'channel-4':
        if action == 'LOOK DOWN':
            print(f"{command}: {action}")
        elif action == 'LOOK UP':
            print(f"{command}: {action}")

    ##### move channel 5 #####

    elif command == 'channel-5':

        if action == 'MOVE FORWARD':

            print(f"{command}: {action}")

            try:

                oscillateOneServo(arc_reduction)

                logging.debug("Attempting to move forward...\n")

            except Exception as e:

                logging.error(f"ERROR (control_logic.py): Failed to move forward in executeCommands: {e}\n")

        elif action == 'MOVE BACKWARD':

            print(f"{command}: {action}")

    ##### shift channel 6 #####

    elif command == 'channel-6':
        if action == 'SHIFT LEFT':
            print(f"{command}: {action}")
        elif action == 'SHIFT RIGHT':
            print(f"{command}: {action}")

    ##### extra channel 7 #####

    elif command == 'channel-7':
        if action == '+':
            print(f"{command}: {action}")
        elif action == '-':
            print(f"{command}: {action}")


########## RUN ROBOTIC PROCESS ##########

##### complete all initialization and begin robotic process #####

runRobot() # initialize receiver
