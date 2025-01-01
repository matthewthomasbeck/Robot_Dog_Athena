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

import cv2
import subprocess
import numpy as np
import os
import signal
import logging


#################################################
############### INITIALIZE CAMERA ###############
#################################################


########## TERMINATE EXISTING CAMERA PIPELINES ##########

def kill_existing_camera_processes():
    try:
        # List all processes using the camera
        result = subprocess.run(
            ["pgrep", "-f", "rpicam-jpeg|rpicam-vid|libcamera"],
            stdout=subprocess.PIPE,
            text=True
        )
        pids = result.stdout.splitlines()
        if pids:
            logging.warning(f"Existing camera processes found: {pids}. Terminating them.\n")
            for pid in pids:
                os.kill(int(pid), signal.SIGKILL)
            logging.info("Successfully killed existing camera processes.\n")
    except Exception as e:
        logging.error(f"Failed to terminate existing camera processes: {e}\n")


########## CREATE CAMERA PIPELINE ##########

def start_camera_process():
    try:
        kill_existing_camera_processes()

        # Optimize rpicam-vid settings
        camera_process = subprocess.Popen(
            ["rpicam-vid", "--width", "1280", "--height", "720", "--framerate", "30", "--timeout", "0", "--output", "-", "--codec", "mjpeg", "--nopreview"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        logging.info("Camera process started successfully with rpicam-vid.")
        return camera_process
    except Exception as e:
        logging.error(f"ERROR (initialize_camera.py): Failed to start camera process: {e}")
        return None
