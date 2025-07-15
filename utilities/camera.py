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

##### import config #####

from utilities.config import CAMERA_CONFIG, RL_NOT_CNN, USE_SIMULATION

##### import necessary libraries #####

import subprocess # import subprocess to run rpicam command
#import os # import os to check if rpicam instances exists
#import signal # import signal to send signals to processes
import logging # import logging for logging messages
import time # add time for waiting
import numpy # add numpy for decoding frames
if not USE_SIMULATION:
    import cv2  # add cv2 for decoding frames





#################################################
############### INITIALIZE CAMERA ###############
#################################################


########## INITIALIZE CAMERA ##########

# function to initialize camera
def initialize_camera(
        width=CAMERA_CONFIG['WIDTH'],
        height=CAMERA_CONFIG['HEIGHT'],
        frame_rate=CAMERA_CONFIG['FRAME_RATE']
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


########## DECODE FRAME ##########

def decode_frame(camera_process, mjpeg_buffer):

    try:
        chunk = camera_process.stdout.read(4096)
        if not chunk:
            return mjpeg_buffer, None, None

        mjpeg_buffer += chunk
        start_idx = mjpeg_buffer.find(b'\xff\xd8')
        end_idx = mjpeg_buffer.find(b'\xff\xd9')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            streamed_frame = mjpeg_buffer[start_idx:end_idx + 2]
            mjpeg_buffer = mjpeg_buffer[end_idx + 2:]

            # Decode JPEG to image first
            inference_frame = cv2.imdecode(numpy.frombuffer(streamed_frame, dtype=numpy.uint8), cv2.IMREAD_COLOR)

            if RL_NOT_CNN:  # if running RL model for movement...
                # 1. Crop
                h = inference_frame.shape[0]
                crop_start = int(h * (1 - CAMERA_CONFIG['CROP_FRACTION']))
                cropped = inference_frame[crop_start:, :, :]
                # 2. Grayscale
                gray_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                # 3. Resize
                output_size = (CAMERA_CONFIG['OUTPUT_WIDTH'], CAMERA_CONFIG['OUTPUT_HEIGHT'])
                resized_frame = cv2.resize(gray_frame, output_size)
                return mjpeg_buffer, streamed_frame, resized_frame

            else:  # CNN or other use
                return mjpeg_buffer, streamed_frame, inference_frame

        if len(mjpeg_buffer) > 65536: # if buffer overflow...
            mjpeg_buffer = b''
        return mjpeg_buffer, None, None

    except Exception as e:
        logging.error(f"(camera.py): Failed to decode frame: {e}\n")
        return mjpeg_buffer, None, None
