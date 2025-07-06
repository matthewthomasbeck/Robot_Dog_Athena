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
import time # <-- add this import

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
        frame_rate=LOOP_RATE_HZ
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

        # Use pkill for each process type
        subprocess.run(["pkill", "-9", "-f", "rpicam-vid"])
        subprocess.run(["pkill", "-9", "-f", "rpicam-jpeg"])
        subprocess.run(["pkill", "-9", "-f", "libcamera"])
        logging.info("(camera.py): Successfully killed existing camera processes.\n")
        time.sleep(0.5)  # Give time for processes to exit

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
        # Optionally, check if process started successfully
        time.sleep(0.2)
        if camera_process.poll() is not None:
            # Process exited immediately
            stderr = camera_process.stderr.read().decode()
            logging.error(f"(camera.py): Camera process failed to start. Stderr: {stderr}\n")
            return None
        return camera_process

    except Exception as e:

        logging.error(f"(camera.py): Failed to start camera process: {e}\n")

        return None