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
        compiled_model = ie.compile_model(model, device_name=device_name) # compile model for specified device
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


########## LOAD AND COMPILE ONNX MODEL ##########

# function to load and compile an ONNX model
def load_and_compile_onnx_model(
        model_path, # path to the ONNX model file
        device_name=INFERENCE_CONFIG['TPU_NAME'] # device name for inference (e.g., "CPU", "GPU", "MYRIAD")
):
    """
    Load and compile an ONNX model for inference.
    This function is specifically designed for ONNX files, not OpenVINO .xml/.bin files.
    """
    
    logging.debug(f"(inference.py): Loading and compiling ONNX model: {model_path}\n")

    try:
        # Load the ONNX model
        ie = Core()
        model = ie.read_model(model=model_path)
        
        # Compile the model for the specified device
        compiled_model = ie.compile_model(model, device_name=device_name)
        
        # Get input and output layers
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        
        logging.info(f"(inference.py): ONNX model loaded and compiled on {device_name}.\n")
        
        # Test with dummy input
        try:
            test_with_dummy_input(compiled_model, input_layer, output_layer)
            logging.info(f"(inference.py): ONNX model dummy input test passed.\n")
        except Exception as e:
            logging.warning(f"(inference.py): ONNX model dummy input test failed: {e}\n")
            # Don't fail completely - some models work fine without dummy input test
        
        return compiled_model, input_layer, output_layer

    except Exception as e:
        logging.error(f"(inference.py): Failed to load/compile ONNX model: {e}\n")
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
        
        # 1. Extract and normalize joint angles (12D) - EXACTLY like training
        # Normalize angles to [-1, 1] range for better training stability with Adam optimizer
        current_angles_vec = []
        for leg_id in ['FL', 'FR', 'BL', 'BR']:
            for joint_name in ['hip', 'upper', 'lower']:
                angle = current_servo_config[leg_id][joint_name]['CURRENT_ANGLE']
                
                # Get joint limits for normalization - EXACTLY like training
                servo_data = current_servo_config[leg_id][joint_name]
                min_angle = servo_data['FULL_BACK_ANGLE']
                max_angle = servo_data['FULL_FRONT_ANGLE']
                
                # Ensure correct order - EXACTLY like training
                if min_angle > max_angle:
                    min_angle, max_angle = max_angle, min_angle
                
                # Normalize to [-1, 1] range - EXACTLY like training
                angle_range = max_angle - min_angle
                if angle_range > 0:
                    normalized_angle = 2.0 * (float(angle) - min_angle) / angle_range - 1.0
                    normalized_angle = np.clip(normalized_angle, -1.0, 1.0)
                else:
                    normalized_angle = 0.0  # Fallback if range is zero
                    
                current_angles_vec.append(normalized_angle)
        
        current_angles_vec = np.array(current_angles_vec, dtype=np.float32)
        logging.info(f"(inference.py): Extracted {len(current_angles_vec)} joint angles, normalized to [-1, 1] range: [{min(current_angles_vec):.3f}, {max(current_angles_vec):.3f}]\n")

        # 2. Encode commands (6D one-hot) - EXACTLY like training
        if isinstance(commands, list):
            command_list = commands
        elif isinstance(commands, str):
            command_list = commands.split('+') if commands else []
        else:
            command_list = []
        
        logging.debug(f"(inference.py): Processing commands: {commands} -> command_list: {command_list}...\n")

        # CRITICAL: Match training.py exactly - 6D command encoding, not 8D
        command_encoding = [
            1.0 if 'w' in command_list else 0.0,
            1.0 if 's' in command_list else 0.0,
            1.0 if 'a' in command_list else 0.0,
            1.0 if 'd' in command_list else 0.0,
            1.0 if 'arrowleft' in command_list else 0.0,
            1.0 if 'arrowright' in command_list else 0.0
        ]
        command_encoding = np.array(command_encoding, dtype=np.float32)
        logging.info(f"(inference.py): Command encoding: {command_encoding}\n")

        # 3. Normalize intensity (1D) - EXACTLY like training
        # Map intensity 1-10 to range [-1.0, 1.0] preserving all 10 distinct levels
        intensity_normalized = (float(intensity) - 5.5) / 4.5  # Maps 1->-1.0, 10->1.0
        intensity_normalized = np.array([intensity_normalized], dtype=np.float32)
        logging.info(f"(inference.py): Intensity {intensity} -> normalized: {intensity_normalized[0]:.3f}\n")

        # Build input vector in EXACT same order as training: [joint_angles, commands, intensity]
        # State size: 12 (joints) + 6 (commands) + 1 (intensity) = 19
        input_vec = np.concatenate([current_angles_vec, command_encoding, intensity_normalized])
        input_vec = input_vec.reshape(1, -1)
        logging.info(f"(inference.py): Built input vector: {input_vec.shape}, total elements: {len(input_vec[0])}\n")

        # Run inference
        logging.info(f"(inference.py): Running model inference...\n")
        result = model([input_vec])
        
        # Extract the actual numpy array from OpenVINO result
        # OpenVINO returns an OVDict with complex keys, but the value is a numpy array
        logging.info(f"(inference.py): Raw result type: {type(result)}\n")
        
        if hasattr(result, 'values'):
            # If result is a dict-like object, get the first value
            logging.info(f"(inference.py): Result has .values() method, extracting first value\n")
            result_array = list(result.values())[0]
            logging.info(f"(inference.py): First value type: {type(result_array)}\n")
        elif isinstance(result, dict):
            # If result is a regular dict, get the first value
            logging.info(f"(inference.py): Result is dict, extracting first value\n")
            result_array = list(result.values())[0]
            logging.info(f"(inference.py): First value type: {type(result_array)}\n")
        elif isinstance(result, list):
            # If result is a list, get the first element
            logging.info(f"(inference.py): Result is list, extracting first element\n")
            result_array = result[0]
            logging.info(f"(inference.py): First element type: {type(result_array)}\n")
        else:
            # If result is already the array
            logging.info(f"(inference.py): Result is direct type: {type(result)}\n")
            result_array = result
        
        # Ensure we have a numpy array
        if hasattr(result_array, 'numpy'):
            logging.info(f"(inference.py): Converting to numpy array\n")
            result_array = result_array.numpy()
        else:
            logging.info(f"(inference.py): Result array already numpy compatible\n")
        
        logging.info(f"(inference.py): Final result array type: {type(result_array)}\n")
        logging.info(f"(inference.py): Extracted result array shape: {result_array.shape}\n")
        logging.info(f"(inference.py): Result array range: [{result_array.min():.3f}, {result_array.max():.3f}]\n")
        
        # Convert action output from [-1, 1] range to actual joint angles - EXACTLY like training
        result_reshaped = result_array.reshape(4, 2, 3)  # 4 legs, 2 angles per leg, 3 joints per leg
        logging.info(f"(inference.py): Reshaped output: {result_reshaped.shape} (4 legs, 2 angles per leg, 3 joints per leg)\n")
        
        legs = ['FL', 'FR', 'BL', 'BR']
        joints = ['hip', 'upper', 'lower']
        
        target_angles = {}
        mid_angles = {}
        action_idx = 0
        
        for i, leg in enumerate(legs):
            target_angles[leg] = {}
            mid_angles[leg] = {}
            for j, joint in enumerate(joints):
                # Get joint limits for denormalization - EXACTLY like training
                servo_data = current_servo_config[leg][joint]
                min_angle = servo_data['FULL_BACK_ANGLE']
                max_angle = servo_data['FULL_FRONT_ANGLE']
                
                # CRITICAL: Handle servo inversion EXACTLY like training
                # The training code ensures correct order by swapping if needed
                if min_angle > max_angle:
                    min_angle, max_angle = max_angle, min_angle
                
                # Convert mid action (-1 to 1) to joint angle - EXACTLY like training
                mid_action = result_reshaped[i, 0, j]  # First angle is mid angle
                # Use EXACTLY the same formula as training
                mid_angle = min_angle + (mid_action + 1.0) * 0.5 * (max_angle - min_angle)
                mid_angle = np.clip(mid_angle, min_angle, max_angle)
                mid_angles[leg][joint] = float(mid_angle)
                
                # Convert target action (-1 to 1) to joint angle - EXACTLY like training
                target_action = result_reshaped[i, 1, j]  # Second angle is target angle
                # Use EXACTLY the same formula as training
                target_angle = min_angle + (target_action + 1.0) * 0.5 * (max_angle - min_angle)
                target_angle = np.clip(target_angle, min_angle, max_angle)
                target_angles[leg][joint] = float(target_angle)
                
                action_idx += 1
                
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