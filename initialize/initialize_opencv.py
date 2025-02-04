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

from openvino.runtime import Core
import numpy as np
import cv2
import subprocess


########## FUNCTION DEFINITIONS ##########

def start_camera_process(width=640, height=480, framerate=30):
    """
    Starts the rpicam-vid process to output MJPEG data via stdout.
    Returns the Popen object (camera_process).
    """
    try:
        camera_process = subprocess.Popen(
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
        print(f"ERROR 15 (initialize_opencv.py): Failed to start camera process: {e}\n")
        return None


def load_and_compile_model(model_xml_path, device_name="MYRIAD"):
    """
    Loads and compiles an OpenVINO model.
    Returns compiled_model, input_layer, output_layer.
    """
    ie = Core()
    try:
        model_bin_path = model_xml_path.replace(".xml", ".bin")
        model = ie.read_model(model=model_xml_path)
        compiled_model = ie.compile_model(model=model, device_name=device_name)
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        print(f"Model loaded and compiled on {device_name}.")
        print(f"Model input shape: {input_layer.shape}")
        return compiled_model, input_layer, output_layer
    except Exception as e:
        print(f"ERROR 16 (initialize_opencv.py): Failed to load/compile model: {e}\n")
        return None, None, None


def test_with_dummy_input(compiled_model, input_layer, output_layer):
    """
    Sends dummy input to the compiled model to ensure it works.
    """
    if compiled_model is None or input_layer is None or output_layer is None:
        print("ERROR 17 (initialize_opencv.py): Model is not properly initialized.\n")
        return

    try:
        dummy_input_shape = input_layer.shape
        dummy_input = np.ones(dummy_input_shape, dtype=np.float32)
        _ = compiled_model([dummy_input])[output_layer]  # Just run inference to test
        print("Dummy input test passed!")
    except Exception as e:
        print(f"ERROR 18 (initialize_opencv.py): Dummy input test failed: {e}\n")


def decode_and_show_frame(mjpeg_buffer):
    """
    Attempts to extract a JPEG from mjpeg_buffer, decode it, and display it.
    Returns the updated buffer after extracting a frame, or None if decoding fails.
    """
    # Look for JPEG frame markers
    start_idx = mjpeg_buffer.find(b'\xff\xd8')  # JPEG start
    end_idx = mjpeg_buffer.find(b'\xff\xd9')    # JPEG end

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        # Extract the JPEG data
        frame_data = mjpeg_buffer[start_idx:end_idx + 2]
        updated_buffer = mjpeg_buffer[end_idx + 2:]  # remove the processed frame
        # Decode
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            # Display
            cv2.imshow("Robot Camera", frame)
            return updated_buffer
        else:
            # Decoding failed, return updated buffer but log a warning
            print("WARNING (initialize_opencv.py): Failed to decode frame.")
            return updated_buffer
    else:
        # No complete frame found; return existing buffer
        return mjpeg_buffer
