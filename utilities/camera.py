##################################################################################
# Copyright (c) 2025 Matthew Thomas Beck                                         #
#                                                                                #
# Licensed under the Creative Commons Attribution-NonCommercial 4.0              #
# International (CC BY-NC 4.0). Personal and educational use is permitted.       #
# Commercial use by companies or for-profit entities is prohibited.              #
##################################################################################





############################################################
############### IMPORT / CREATE DEPENDENCIES ###############
############################################################


########## IMPORT DEPENDENCIES ##########

##### import config #####

import utilities.config as config

##### import necessary libraries #####

import subprocess # import subprocess to run rpicam command
import logging # import logging for logging messages
import time # add time for waiting
import numpy # add numpy for decoding frames
import cv2  # add cv2 for decoding frames for isaac sim and the real robot





################################################
############### CAMERA FUNCTIONS ###############
################################################


########## INITIALIZE CAMERA ##########

def initialize_camera( # function to initialize camera
        robot_id=0, # add robot_id parameter for multi-robot support (from sim days)
        width=config.CAMERA_CONFIG['WIDTH'],
        height=config.CAMERA_CONFIG['HEIGHT'],
        frame_rate=config.CAMERA_CONFIG['FRAME_RATE']
):

    ##### initialize camera by killing old processes and starting a new one #####

    logging.debug("(camera.py): Initializing camera...\n")
    _kill_existing_camera_processes() # kill existing camera processes
    camera_process = _start_camera_process(robot_id, width, height, frame_rate) # start new camera process

    if camera_process is None: # if camera process failed to start...
        logging.error("(camera.py): Camera initialization failed, no camera process started.\n")
        return None

    else: # if camera process started successfully...
        return camera_process


########## TERMINATE EXISTING CAMERA PIPELINES ##########

def _kill_existing_camera_processes(): # function to kill existing camera processes if they exist

    try:
        logging.debug("(camera.py): Checking for existing camera processes...\n")
        subprocess.run(["pkill", "-9", "-f", "rpicam-vid"]) # use pkill for each process type
        subprocess.run(["pkill", "-9", "-f", "rpicam-jpeg"])
        subprocess.run(["pkill", "-9", "-f", "libcamera"])
        logging.info("(camera.py): Successfully killed existing camera processes.\n")
        time.sleep(0.5)  # give time for processes to exit

    except Exception as e:
        logging.error(f"(camera.py): Failed to terminate existing camera processes: {e}\n")


########## CREATE CAMERA PIPELINE ##########

def _start_camera_process(robot_id, width, height, frame_rate): # function to start camera process for opencv

    try:
        real_camera = subprocess.Popen(  # open an rpicam vid process
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
        time.sleep(0.2)
        if real_camera.poll() is not None:
            stderr = real_camera.stderr.read().decode()
            logging.error(f"(camera.py): Camera process failed to start. Stderr: {stderr}\n")
            return None
        logging.info(f"(camera.py): Camera initialized successfully with PID {real_camera.pid}.\n")
        return real_camera

    except Exception as e:
        logging.error(f"(camera.py): Failed to start camera process: {e}\n")
        return None


########## DECODE FRAME ##########

def decode_real_frame(camera_process, mjpeg_buffer):

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
            inference_frame = cv2.imdecode(numpy.frombuffer(streamed_frame, dtype=numpy.uint8), cv2.IMREAD_COLOR)

            if config.RL_NOT_CNN:  # if running RL model for movement...

                ##### crop #####

                h = inference_frame.shape[0]
                crop_start = int(h * (1 - config.CAMERA_CONFIG['CROP_FRACTION']))
                cropped = inference_frame[crop_start:, :, :]

                ##### grayscale #####

                gray_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

                ##### resize #####

                output_size = (config.CAMERA_CONFIG['OUTPUT_WIDTH'], config.CAMERA_CONFIG['OUTPUT_HEIGHT'])
                resized_frame = cv2.resize(gray_frame, output_size)
                return mjpeg_buffer, streamed_frame, resized_frame

            else: # cnn or other use
                return mjpeg_buffer, streamed_frame, inference_frame

        if len(mjpeg_buffer) > 65536: # if buffer overflow...
            mjpeg_buffer = b'' # reset buffer to avoid overflow
        return mjpeg_buffer, None, None

    except Exception as e:
        logging.error(f"(camera.py): Failed to decode frame: {e}\n")
        return mjpeg_buffer, None, None
