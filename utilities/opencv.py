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

from utilities.config import INFERENCE_CONFIG, USE_SIMULATION

##### import necessary libraries #####

import numpy as np # import NumPy for array manipulation
import logging # import logging for logging messages
if not USE_SIMULATION: # if not using simulation...
    from openvino.runtime import Core  # import OpenVINO runtime
    import cv2 # import OpenCV for image processing

########## CREATE DEPENDENCIES ##########

##### all commands #####

# used to encode commands as a fixed-length vector
ALL_COMMANDS = ['a', 'd', 's', 'w', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright']





############################################################
############### IMPORT / CREATE DEPENDENCIES ###############
############################################################


########## HELPER FUNCTIONS ##########

def encode_commands(commands):
    """Multi-hot encode commands as a fixed-length vector."""
    vec = np.zeros(len(ALL_COMMANDS), dtype=np.float32)
    for i, cmd in enumerate(ALL_COMMANDS):
        if cmd in commands:
            vec[i] = 1.0
    return vec

def normalize_scalar(value, min_val, max_val):
    """Normalize a scalar to [0, 1]."""
    return (value - min_val) / (max_val - min_val)

def normalize_feet_positions(feet_dict, min_xyz, max_xyz):
    """Flatten and normalize all foot positions."""
    arr = []
    for leg in ['FL', 'FR', 'BL', 'BR']:
        for axis in ['x', 'y', 'z']:
            v = feet_dict[leg][axis]
            arr.append(normalize_scalar(v, min_xyz[axis], max_xyz[axis]))
    return np.array(arr, dtype=np.float32)


########## LOAD AND COMPILE MODEL ##########

# function to load and compile an OpenVINO model
def load_and_compile_model(
        model_path, # path to the model file
        device_name=INFERENCE_CONFIG['TPU_NAME'] # device name for inference (e.g., "CPU", "GPU", "MYRIAD")
):

    ##### clean up OpenCV windows from last run #####

    try: # try to destroy any lingering OpenCV windows from previous runs
        cv2.destroyAllWindows()
        logging.info("(opencv.py): Closed lingering OpenCV windows.\n")

    except Exception as e:
        logging.warning(f"(opencv.py): Failed to destroy OpenCV windows: {e}\n")

    ##### compile model #####

    logging.debug("(opencv.py): Loading and compiling model...\n")

    try: # try to load and compile the model

        ie = Core() # check for devices
        #model_bin_path = model_path.replace(".xml", ".bin") # get binary path from XML path (incase needed)
        model = ie.read_model(model=model_path) # read model from XML file
        compiled_model = ie.compile_model(model=model, device_name=device_name) # compile model for specified device
        input_layer = compiled_model.input(0) # get input layer of compiled model
        output_layer = compiled_model.output(0) # get output layer of compiled model
        logging.info(f"(opencv.py): Model loaded and compiled on {device_name}.\n")
        logging.debug(f"(opencv.py): Model input shape: {input_layer.shape}\n")

        try: # try to test model with dummy input

            test_with_dummy_input(compiled_model, input_layer, output_layer) # test model with dummy input
        except Exception as e: # if dummy input test fails...
            logging.warning(f"(opencv.py): Dummy input test failed: {e}\n")
            return None, None, None

        return compiled_model, input_layer, output_layer

    except Exception as e:
        logging.error(f"(opencv.py): Failed to load/compile model: {e}\n")
        return None, None, None


########## TEST MODEL ##########

def test_with_dummy_input(compiled_model, input_layer, output_layer): # function to test the model with a dummy input

    ##### check if model/layers are properly initialized #####

    logging.debug("(opencv.py): Testing model with dummy input...\n")

    # if model/layers not properly initialized...
    if compiled_model is None or input_layer is None or output_layer is None:
        logging.error("(opencv.py): Model is not properly initialized.\n")
        return

    ##### run dummy input through the model #####

    try: # try to run a dummy input through the model

        dummy_input_shape = input_layer.shape # get the shape of the input layer
        dummy_input = np.ones(dummy_input_shape, dtype=np.float32) # create a dummy input with ones
        _ = compiled_model([dummy_input])[output_layer] # run the model but don't use output
        logging.info("(opencv.py): Dummy input test passed.\n")

    except Exception as e:
        logging.error(f"(opencv.py): Dummy input test failed: {e}\n")


########## RUN GAIT ADJUSTMENT RL MODEL STANDARD ##########

def run_gait_adjustment_standard(  # function to run gait adjustment RL model without images for all terrain
        model,
        input_layer,
        output_layer,
        commands,
        frame,
        intensity,
        current_feet_positions
):
    """
    Run RL model with vision.
    - model: OpenVINO compiled model
    - input_layer, output_layer: OpenVINO layers
    - commands: list of str
    - frame: preprocessed np.ndarray (H, W) or (H, W, 1)
    - speed, acceleration: scalars
    - current_feet_positions: dict of dicts
    Returns: target_positions, mid_positions
    """
    # --- Normalize/encode inputs ---
    cmd_vec = encode_commands(commands)
    intensity_norm = normalize_scalar(intensity, 1, 10)  # adjust min/max as needed
    feet_vec = normalize_feet_positions(current_feet_positions, {'x':-0.2,'y':-0.2,'z':-0.2}, {'x':0.2,'y':0.2,'z':0.2})  # adjust min/max as needed

    # Flatten frame and normalize to [0,1]
    frame_flat = frame.astype(np.float32) / 255.0
    frame_flat = frame_flat.flatten()

    # --- Assemble input ---
    input_vec = np.concatenate([cmd_vec, intensity_norm, feet_vec, frame_flat])
    input_vec = input_vec.reshape(1, -1)  # batch dimension

    # --- Run inference ---
    result = model([input_vec])[output_layer]
    # Assume result shape: (1, 24) for 4 legs * 2 points * 3 coords

    # --- Parse output ---
    result = result.reshape(4, 2, 3)  # (leg, point, xyz)
    legs = ['FL', 'FR', 'BL', 'BR']
    target_positions = {}
    mid_positions = {}
    for i, leg in enumerate(legs):
        mid_positions[leg] = {'x': result[i,0,0], 'y': result[i,0,1], 'z': result[i,0,2]}
        target_positions[leg] = {'x': result[i,1,0], 'y': result[i,1,1], 'z': result[i,1,2]}
    return target_positions, mid_positions


########## RUN GAIT ADJUSTMENT RL MODEL WITHOUT IMAGES ##########

def run_gait_adjustment_blind( # function to run gait adjustment RL model without images for speedy processing
        model,
        input_layer,
        output_layer,
        commands,
        intensity,
        current_feet_positions
):
    """
    Run RL model without vision.
    """
    cmd_vec = encode_commands(commands)
    intensity_norm = normalize_scalar(intensity, 1, 10)  # adjust min/max as needed
    feet_vec = normalize_feet_positions(current_feet_positions, {'x':-0.2,'y':-0.2,'z':-0.2}, {'x':0.2,'y':0.2,'z':0.2})

    input_vec = np.concatenate([cmd_vec, intensity_norm, feet_vec])
    input_vec = input_vec.reshape(1, -1)

    result = model([input_vec])[output_layer]
    result = result.reshape(4, 2, 3)
    legs = ['FL', 'FR', 'BL', 'BR']
    target_positions = {}
    mid_positions = {}
    for i, leg in enumerate(legs):
        mid_positions[leg] = {'x': result[i,0,0], 'y': result[i,0,1], 'z': result[i,0,2]}
        target_positions[leg] = {'x': result[i,1,0], 'y': result[i,1,1], 'z': result[i,1,2]}
    return target_positions, mid_positions


########## RUN PERSON DETECTION CNN MODEL AND SHOW FRAME ##########

def run_person_detection(compiled_model, input_layer, output_layer, frame, run_inference):

    if frame is None:
        logging.warning("(opencv.py): Frame is None.\n")
        return

    try:
        if not run_inference:

            logging.debug("(opencv.py): Not running inference, passing...\n")
            #cv2.imshow("video (standard)", frame)
            #cv2.waitKey(1)
            return

        if compiled_model is not None and input_layer is not None and output_layer is not None:

            logging.debug("(opencv.py): Running inference...\n")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_blob = cv2.resize(frame_rgb, (256, 256)).transpose(2, 0, 1)
            input_blob = np.expand_dims(input_blob, axis=0).astype(np.float32)
            results = compiled_model([input_blob])[output_layer]

            for detection in results[0][0]:
                confidence = detection[2]
                if confidence > 0.5:
                    xmin, ymin, xmax, ymax = map(
                        int, detection[3:7] * [
                            frame.shape[1], frame.shape[0],
                            frame.shape[1], frame.shape[0]
                        ]
                    )
                    label = f"ID {int(detection[1])}: {confidence:.2f}"
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            logging.info("(opencv.py): Inference complete.\n")
            #cv2.imshow("video (inference)", frame)
            #cv2.waitKey(1)
            
        else:
            logging.warning("(opencv.py): Inference requested but model is not loaded.\n")

    except Exception as e:
        logging.error(f"(opencv.py): Inference error: {e}\n")