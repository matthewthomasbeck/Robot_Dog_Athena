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


########## PROCESS A FRAME ##########

def process_frame(camera_process):
    if camera_process is None:
        logging.error("ERROR (initialize_camera.py): Camera process is None.\n")
        return None

    jpg = b''
    try:
        # Check for errors in stderr
        if camera_process.stderr:
            err_output = camera_process.stderr.read(1024).decode('utf-8', errors='replace').strip()
            if err_output:
                if "error" in err_output.lower():
                    logging.error(f"ERROR (initialize_camera.py): rpicam-vid error: {err_output}\n")
                    return None
                else:
                    logging.warning(f"WARNING (initialize_camera.py): rpicam-vid warning: {err_output}\n")

        # Heartbeat check: ensure camera process is running
        if camera_process.poll() is not None:
            logging.error("ERROR (initialize_camera.py): Camera process has stopped unexpectedly (heartbeat check failed).\n")
            return None

        # Read a single frame from the stdout stream
        chunk = camera_process.stdout.read(4096)
        if not chunk:
            logging.error("ERROR (initialize_camera.py): Camera process stopped sending data.\n")
            return None

        jpg += chunk
        a = jpg.find(b'\xff\xd8')  # JPEG start marker
        b = jpg.find(b'\xff\xd9')  # JPEG end marker
        if a != -1 and b != -1:
            frame = jpg[a:b+2]
            jpg = jpg[b+2:]  # Update buffer with remaining data
            decoded_frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded_frame is None:
                logging.error("ERROR (initialize_camera.py): Failed to decode frame.\n")
                return None
            return decoded_frame
    except Exception as e:
        logging.error(f"ERROR (initialize_camera.py): Unexpected exception in process_frame: {e}\n")
        return None
