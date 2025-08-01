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

import utilities.config as config

##### import necessary libraries #####

import subprocess # import subprocess to run rpicam command
import logging # import logging for logging messages
import time # add time for waiting
import numpy # add numpy for decoding frames
import cv2  # add cv2 for decoding frames for isaac sim and the real robot

##### import isaac sim dependencies #####

if config.USE_SIMULATION and config.USE_ISAAC_SIM:

    from isaacsim.sensors.camera import Camera
    import isaacsim.core.utils.numpy.rotations as rot_utils
    from isaacsim.core.utils.stage import get_current_stage





#################################################
############### INITIALIZE CAMERA ###############
#################################################


########## INITIALIZE CAMERA ##########

def initialize_camera( # function to initialize camera
        width=config.CAMERA_CONFIG['WIDTH'],
        height=config.CAMERA_CONFIG['HEIGHT'],
        frame_rate=config.CAMERA_CONFIG['FRAME_RATE']
):

    ##### initialize camera by killing old processes and starting a new one #####

    logging.debug("(camera.py): Initializing camera...\n")
    _kill_existing_camera_processes() # kill existing camera processes
    camera_process = _start_camera_process(width, height, frame_rate) # start new camera process

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

def _start_camera_process(width, height, frame_rate): # function to start camera process for opencv

    if not config.USE_SIMULATION and not config.USE_ISAAC_SIM:  # if physical robot...
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

    elif config.USE_SIMULATION and config.USE_ISAAC_SIM: # if isaac sim...
        try: # try to make camera object for isaac sim

            logging.debug("(camera.py) Computing aperture size for isaac camera...\n")
            horizontal_aperture = config.CAMERA_CONFIG['CAMERA_WIDTH'] * config.CAMERA_CONFIG['PIXEL_SIZE_UM'] * 1e-6
            vertical_aperture = config.CAMERA_CONFIG['CAMERA_HEIGHT'] * config.CAMERA_CONFIG['PIXEL_SIZE_UM'] * 1e-6
            logging.info("(camera.py) Computed aperture size for isaac camera.\n")

            logging.debug("(camera.py) Computing focal length for isaac camera...\n")
            focal_length_x = (horizontal_aperture / 2) / numpy.tan(numpy.radians(config.CAMERA_CONFIG['FOV_HORIZONTAL'] / 2))
            focal_length_y = (vertical_aperture / 2) / numpy.tan(numpy.radians(config.CAMERA_CONFIG['FOV_VERTICAL'] / 2))
            focal_length = (focal_length_x + focal_length_y) / 2
            logging.info("(camera.py) Computed focal length for isaac camera.\n")

            logging.debug("(camera.py) Creating isaac camera object...\n")
            isaac_camera = Camera(
                prim_path="/World/robot_dog/athena_front_face/camera_sensor",
                resolution=(config.CAMERA_CONFIG['WIDTH'], config.CAMERA_CONFIG['HEIGHT']),
                frequency=config.CAMERA_CONFIG['FRAME_RATE'],
                orientation=rot_utils.euler_angles_to_quats(numpy.array([0, 0, 0]), degrees=True),  # change if needed
                position=numpy.array([0.0, 0.0, 0.01]),  # a tiny offset if needed
            )
            logging.debug("(camera.py) Created isaac camera object.\n")

            #logging.debug("(camera.py) Attaching isaac camera to parent with offset...\n")
            #stage = get_current_stage()
            #UsdGeom.Xformable(stage.GetPrimAtPath(isaac_camera.prim_path)).AddTransformOp().Set(
                #Gf.Matrix4d().SetTranslate((0, 0, 0.01)))
            #isaac_camera.set_parent("/World/robot_dog/athena_front_face")
            #logging.info("(camera.py) Attached isaac camera to parent with offset.\n")

            logging.debug("(camera.py) Applying optics to isaac camera...\n")
            isaac_camera.set_focal_length(focal_length)
            isaac_camera.set_horizontal_aperture(horizontal_aperture)
            isaac_camera.set_vertical_aperture(vertical_aperture)
            isaac_camera.set_focus_distance(config.CAMERA_CONFIG['DEPTH_OF_FIELD'])
            isaac_camera.set_lens_aperture(config.CAMERA_CONFIG['APERTURE_RATIO'])
            isaac_camera.set_clipping_range(0.05, 100000.0)
            logging.info("(camera.py) Applied optics to isaac camera.\n")

            logging.debug("(camera.py) Initializing isaac camera...\n")
            isaac_camera.initialize()
            logging.info("(camera.py) Initialized isaac camera.\n")

            for _ in range(10):
                rgba = isaac_camera.get_rgba()
                if rgba is not None and rgba.size > 0:
                    break
                time.sleep(0.1)
                if config.USE_SIMULATION and config.USE_ISAAC_SIM:
                    config.ISAAC_WORLD.step(render=True)
            else:
                logging.warning("(camera.py): Isaac camera failed to warm up with a valid frame.\n")

            return isaac_camera

        except Exception as e: # if camera object creation fails...
            logging.error(f"(camera.py): Failed to create camera object in Isaac Sim: {e}\n")


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
                # 1. Crop
                h = inference_frame.shape[0]
                crop_start = int(h * (1 - config.CAMERA_CONFIG['CROP_FRACTION']))
                cropped = inference_frame[crop_start:, :, :]
                # 2. Grayscale
                gray_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                # 3. Resize
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


########## DECODE ISAAC SIM FRAME ##########

def decode_isaac_frame(camera_object):
    try:
        rgba = camera_object.get_rgba()  # Shape: (H, W, 4), floats 0-1
        image = (rgba[..., :3] * 255).astype(numpy.uint8)  # Strip alpha, scale to 0–255
        inference_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h = inference_frame.shape[0]
        crop_start = int(h * (1 - config.CAMERA_CONFIG['CROP_FRACTION']))
        cropped = inference_frame[crop_start:, :, :]
        gray_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(
            gray_frame,
            (config.CAMERA_CONFIG['OUTPUT_WIDTH'], config.CAMERA_CONFIG['OUTPUT_HEIGHT'])
        )
        return None, inference_frame, resized_frame
    except Exception as e:
        logging.error(f"(camera.py): Failed to decode Isaac Sim frame: {e}\n")
        return None, None, None
