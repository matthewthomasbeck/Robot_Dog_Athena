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

import subprocess # import subprocess to run rpicam command
import os # import os to check if rpicam instances exists
import signal # import signal to send signals to processes
import logging # import logging for logging messages

##### import config #####

from utilities.config import LOOP_RATE_HZ, CAMERA_CONFIG # import config to get camera settings





#################################################
############### INITIALIZE CAMERA ###############
#################################################


########## INITIALIZE CAMERA ##########

# function to initialize camera
def initialize_camera(
        width=CAMERA_CONFIG['WIDTH'],
        height=CAMERA_CONFIG['HEIGHT'],
        frame_rate=30 #LOOP_RATE_HZ TODO removed to see if camera bug
):

    ##### initialize camera by killing old processes and starting a new one #####

    logging.debug("(camera.py): Initializing camera...\n")
    _kill_existing_camera_processes() # kill existing camera processes
    camera_process = _start_camera_process(width, height, frame_rate) # start new camera process

    if camera_process is None: # if camera process failed to start...
        logging.error("(camera.py): Camera initialization failed, no camera process started.\n")

    else: # if camera process started successfully...
        logging.info(f"(camera.py): Camera initialized successfully with PID {camera_process.pid}.\n")
        return camera_process


########## TERMINATE EXISTING CAMERA PIPELINES ##########

def _kill_existing_camera_processes(): # function to kill existing camera processes if they exist

    try:

        logging.debug("(camera.py): Checking for existing camera processes...\n")

        # use pgrep to find existing camera processes
        result = subprocess.run(["pgrep", "-f", "rpicam-jpeg|rpicam-vid|libcamera"], stdout=subprocess.PIPE, text=True)

        pids = result.stdout.splitlines() # get the process IDs of existing camera processes

        if pids: # if there are any existing camera processes...
            logging.warning(f"(camera.py): Existing camera processes found: {pids}. Terminating them.\n")

            for pid in pids: # iterate through each process ID and kill it
                os.kill(int(pid), signal.SIGKILL)

            logging.info("(camera.py): Successfully killed existing camera processes.\n")

    except Exception as e:

        logging.error(f"(camera.py): Failed to terminate existing camera processes: {e}\n")


########## CREATE CAMERA PIPELINE ##########

def _start_camera_process(width, height, frame_rate): # function to start camera process for opencv

    try:

        camera_process = subprocess.Popen( # open an rpicam vid process
            [
                "rpicam-vid",
                "--width", str(width),
                "--height", str(height),
                "--framerate", str(frame_rate),
                "--timeout", "0",
                "--output", "-",
                "--codec", "mjpeg",
                "--nopreview"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )

        return camera_process

    except Exception as e:

        logging.error(f"(camera.py): Failed to start camera process: {e}\n")

        return None