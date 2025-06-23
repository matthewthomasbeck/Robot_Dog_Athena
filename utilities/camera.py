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

import subprocess # import subprocess to run rpicam command
import os # import os to check if rpicam instances exists
import signal # import signal to send signals to processes
import logging # import logging for logging messages





#################################################
############### INITIALIZE CAMERA ###############
#################################################


########## TERMINATE EXISTING CAMERA PIPELINES ##########

def kill_existing_camera_processes(): # function to kill existing camera processes if they exist

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

def start_camera_process(width=640, height=480, framerate=30): # function to start camera process for opencv

    try:

        camera_process = subprocess.Popen( # open an rpicam vid process
            [
                "rpicam-vid",
                "--width", str(width),
                "--height", str(height),
                "--framerate", str(framerate),
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