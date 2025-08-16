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

##### get physical robot dependencies #####

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
        current_servo_config
):
    """
    Run RL model with vision.
    - model: OpenVINO compiled model
    - input_layer, output_layer: OpenVINO layers
    - commands: list of str
    - frame: preprocessed np.ndarray (H, W) or (H, W, 1)
    - intensity: scalar
    - current_servo_config: dict of dicts with CURRENT_ANGLE values
    Returns: target_angles, mid_angles
    """
    # --- Normalize/encode inputs ---
    cmd_vec = encode_commands(commands)
    intensity_norm = normalize_scalar(intensity, 1, 10)  # adjust min/max as needed
    
    # Flatten current joint angles and normalize
    current_angles_vec = []
    for leg in ['FL', 'FR', 'BL', 'BR']:
        for joint in ['hip', 'upper', 'lower']:
            current_angle = current_servo_config[leg][joint]['CURRENT_ANGLE']
            # Normalize angle to [0, 1] assuming range [-pi, pi]
            angle_norm = (current_angle + np.pi) / (2 * np.pi)
            current_angles_vec.append(angle_norm)
    current_angles_vec = np.array(current_angles_vec, dtype=np.float32)

    # Flatten frame and normalize to [0,1]
    frame_flat = frame.astype(np.float32) / 255.0
    frame_flat = frame_flat.flatten()

    # --- Assemble input ---
    input_vec = np.concatenate([cmd_vec, intensity_norm, current_angles_vec, frame_flat])
    input_vec = input_vec.reshape(1, -1)  # batch dimension

    # --- Run inference ---
    result = model([input_vec])[output_layer]
    # Assume result shape: (1, 24) for 4 legs * 2 points * 3 joints

    # --- Parse output ---
    result = result.reshape(4, 2, 3)  # (leg, point, joint)
    legs = ['FL', 'FR', 'BL', 'BR']
    joints = ['hip', 'upper', 'lower']
    
    target_angles = {}
    mid_angles = {}
    for i, leg in enumerate(legs):
        target_angles[leg] = {}
        mid_angles[leg] = {}
        for j, joint in enumerate(joints):
            # Denormalize angles from [0, 1] back to [-pi, pi]
            mid_angle_norm = result[i, 0, j]
            target_angle_norm = result[i, 1, j]
            mid_angles[leg][joint] = (mid_angle_norm * 2 * np.pi) - np.pi
            target_angles[leg][joint] = (target_angle_norm * 2 * np.pi) - np.pi
    
    return target_angles, mid_angles


########## RUN GAIT ADJUSTMENT RL MODEL WITHOUT IMAGES ##########

def run_gait_adjustment_blind( # function to run gait adjustment RL model without images for speedy processing
        model,
        input_layer,
        output_layer,
        commands,
        intensity,
        current_servo_config
):
    """
    Run RL model without vision.
    EXACTLY matches data treatment from get_rl_action_blind in training.py
    """
    try:
        logging.debug(f"(inference.py): Starting gait adjustment inference for commands: {commands}, intensity: {intensity}...\n")
        
        # 1. Extract joint angles (12D) - EXACTLY like training (NO NORMALIZATION)
        current_angles_vec = []
        for leg_id in ['FL', 'FR', 'BL', 'BR']:
            for joint_name in ['hip', 'upper', 'lower']:
                angle = current_servo_config[leg_id][joint_name]['CURRENT_ANGLE']
                current_angles_vec.append(float(angle))  # Raw angle, no normalization
        
        current_angles_vec = np.array(current_angles_vec, dtype=np.float32)
        logging.info(f"(inference.py): Extracted {len(current_angles_vec)} joint angles, range: [{min(current_angles_vec):.3f}, {max(current_angles_vec):.3f}] rad\n")

        # 2. Encode commands (8D one-hot) - EXACTLY like training
        if isinstance(commands, list):
            command_list = commands
        elif isinstance(commands, str):
            command_list = commands.split('+') if commands else []
        else:
            command_list = []
        
        logging.debug(f"(inference.py): Processing commands: {commands} -> command_list: {command_list}...\n")

        command_encoding = [
            1.0 if 'w' in command_list else 0.0,
            1.0 if 's' in command_list else 0.0,
            1.0 if 'a' in command_list else 0.0,
            1.0 if 'd' in command_list else 0.0,
            1.0 if 'arrowleft' in command_list else 0.0,
            1.0 if 'arrowright' in command_list else 0.0,
            1.0 if 'arrowup' in command_list else 0.0,
            1.0 if 'arrowdown' in command_list else 0.0
        ]
        command_encoding = np.array(command_encoding, dtype=np.float32)
        logging.info(f"(inference.py): Command encoding: {command_encoding}\n")

        # 3. Normalize intensity (1D) - EXACTLY like training
        intensity_normalized = float(intensity) / 10.0
        intensity_normalized = np.array([intensity_normalized], dtype=np.float32)
        logging.info(f"(inference.py): Intensity {intensity} -> normalized: {intensity_normalized[0]:.3f}\n")

        # Build input vector in EXACT same order as training: [joint_angles, commands, intensity]
        input_vec = np.concatenate([current_angles_vec, command_encoding, intensity_normalized])
        input_vec = input_vec.reshape(1, -1)
        logging.info(f"(inference.py): Built input vector: {input_vec.shape}, total elements: {len(input_vec[0])}\n")

        # Run inference
        logging.info(f"(inference.py): Running model inference...\n")
        result = model([input_vec])[output_layer]
        logging.info(f"(inference.py): Model output shape: {result.shape}\n")
        
        result = result.reshape(4, 2, 3)
        logging.info(f"(inference.py): Reshaped output: {result.shape} (4 legs, 2 angles per leg, 3 joints per leg)\n")
        
        legs = ['FL', 'FR', 'BL', 'BR']
        joints = ['hip', 'upper', 'lower']
        
        target_angles = {}
        mid_angles = {}
        for i, leg in enumerate(legs):
            target_angles[leg] = {}
            mid_angles[leg] = {}
            for j, joint in enumerate(joints):
                # Denormalize angles from [0, 1] back to [-pi, pi]
                mid_angle_norm = result[i, 0, j]
                target_angle_norm = result[i, 1, j]
                mid_angles[leg][joint] = (mid_angle_norm * 2 * np.pi) - np.pi
                target_angles[leg][joint] = (target_angle_norm * 2 * np.pi) - np.pi
                
                logging.debug(f"(inference.py): {leg}_{joint}: mid={mid_angles[leg][joint]:.3f} rad, target={target_angles[leg][joint]:.3f} rad\n")
        
        logging.info(f"(inference.py): Generated angles - Mid range: [{min([min(angles.values()) for angles in mid_angles.values()]):.3f}, {max([max(angles.values()) for angles in mid_angles.values()]):.3f}] rad\n")
        logging.info(f"(inference.py): Generated angles - Target range: [{min([min(angles.values()) for angles in target_angles.values()]):.3f}, {max([max(angles.values()) for angles in target_angles.values()]):.3f}] rad\n")
        
        # Add movement_rates to match expected return format
        movement_rates = {
            'FL': {'speed': 16383, 'acceleration': 255},
            'FR': {'speed': 16383, 'acceleration': 255},
            'BL': {'speed': 16383, 'acceleration': 255},
            'BR': {'speed': 16383, 'acceleration': 255}
        }
        
        logging.info(f"(inference.py): Gait adjustment inference completed successfully.\n")
        return target_angles, mid_angles, movement_rates
        
    except Exception as e:
        logging.error(f"(inference.py): Error during gait adjustment inference: {e}\n")
        logging.error(f"(inference.py): Commands: {commands}, Intensity: {intensity}\n")
        logging.error(f"(inference.py): Current servo config keys: {list(current_servo_config.keys()) if current_servo_config else 'None'}\n")
        raise


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