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

import numpy as np # import NumPy for array manipulation
import logging # import logging for logging messages

##### get physical robot dependencies #####

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

    vec = np.zeros(len(ALL_COMMANDS), dtype=np.float32)
    for i, cmd in enumerate(ALL_COMMANDS):
        if cmd in commands:
            vec[i] = 1.0
    return vec

def normalize_scalar(value, min_val, max_val):

    return (value - min_val) / (max_val - min_val)

def normalize_feet_positions(feet_dict, min_xyz, max_xyz):

    arr = []
    for leg in ['FL', 'FR', 'BL', 'BR']:
        for axis in ['x', 'y', 'z']:
            v = feet_dict[leg][axis]
            arr.append(normalize_scalar(v, min_xyz[axis], max_xyz[axis]))
    return np.array(arr, dtype=np.float32)


########## LOAD AND COMPILE MODEL ##########

def load_and_compile_model( # function to load and compile an OpenVINO model
        model_path, # path to the model file
        device_name=config.INFERENCE_CONFIG['TPU_NAME'] # device name for inference (e.g., "CPU", "GPU", "MYRIAD")
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

def load_and_compile_onnx_model( # function to load and compile model
        model_path, # path to the ONNX model file
        device_name=config.INFERENCE_CONFIG['TPU_NAME'] # device name for inference (e.g., "CPU", "GPU", "MYRIAD")
):
    
    logging.debug(f"(inference.py): Loading and compiling ONNX model: {model_path}\n")

    try: # try to load and compile the ONNX model

        ie = Core() # import runtime
        model = ie.read_model(model=model_path) # load the model
        compiled_model = ie.compile_model(model, device_name=device_name) # compile the model for the specified device
        input_layer = compiled_model.input(0) # get input and output layers
        output_layer = compiled_model.output(0) # get input and output layers
        
        logging.info(f"(inference.py): ONNX model loaded and compiled on {device_name}.\n")

        try: # attempt to run with dummy input
            test_with_dummy_input(compiled_model, input_layer, output_layer)
            logging.info(f"(inference.py): ONNX model dummy input test passed.\n")

        except Exception as e: # if dummy input test fails...
            logging.warning(f"(inference.py): ONNX model dummy input test failed: {e}\n")
        
        return compiled_model, input_layer, output_layer # return compiled model and layers (some models may still work)

    except Exception as e: # if loading/compiling fails...
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

def run_gait_adjustment_standard(  # function to run gait adjustment RL model with vision
        model,
        input_layer,
        output_layer,
        commands,
        frame,
        intensity
):

    try: # try to run the model with vision

        ##### set variables #####

        state = [] # build state vector
        target_angles = {}
        mid_angles = {}

        logging.debug(f"(inference.py): Starting standard gait adjustment inference for commands: {commands}, intensity: {intensity}...\n")

        ##### add last 5 target angle sets (60D total: 12 * 5) #####

        if not config.PREVIOUS_POSITIONS or len(config.PREVIOUS_POSITIONS) == 0: # if prev pos initialized...
            logging.warning("PREVIOUS_POSITIONS not initialized, initializing for physical robot...\n")
            config.PREVIOUS_POSITIONS = []
            robot_history = deque(maxlen=5)
            for _ in range(5):
                robot_history.append(np.zeros(12, dtype=np.float32))
            config.PREVIOUS_POSITIONS.append(robot_history)

        for historical_angles in config.PREVIOUS_POSITIONS[0]: # add angle sets
            state.extend(historical_angles)

        ##### encode commands (6D one-hot for movement commands only) #####

        if isinstance(commands, list):
            command_list = commands
        elif isinstance(commands, str):
            command_list = commands.split('+') if commands else []
        else:
            command_list = []
        
        command_encoding = [
            1.0 if 'w' in command_list else 0.0,
            1.0 if 's' in command_list else 0.0,
            1.0 if 'a' in command_list else 0.0,
            1.0 if 'd' in command_list else 0.0,
            1.0 if 'arrowleft' in command_list else 0.0,
            1.0 if 'arrowright' in command_list else 0.0
        ]
        
        state.extend(command_encoding)

        ##### prepare fields #####

        intensity_normalized = (float(intensity) - 5.5) / 4.5 # normalize intensity (1D)
        state.append(intensity_normalized)
        frame_flat = frame.astype(np.float32) / 255.0 # normalize frame to [0, 1]
        frame_flat = frame_flat.flatten()
        input_vec = np.concatenate([state, frame_flat]) # assemble input
        input_vec = input_vec.reshape(1, -1)  # batch dimension

        ##### run inference #####

        result = model([input_vec])[output_layer] #assume result shape: (1, 24) for 4 legs * 2 points * 3 joints

        ##### parse output #####

        result = result.reshape(4, 2, 3)  # (leg, point, joint)
        legs = ['FL', 'FR', 'BL', 'BR']
        joints = ['hip', 'upper', 'lower']

        ##### process output #####

        for i, leg in enumerate(legs):
            target_angles[leg] = {}
            mid_angles[leg] = {}
            for j, joint in enumerate(joints):
                # Denormalize angles from [0, 1] back to [-pi, pi]
                mid_angle_norm = result[i, 0, j]
                target_angle_norm = result[i, 1, j]
                mid_angles[leg][joint] = (mid_angle_norm * 2 * np.pi) - np.pi
                target_angles[leg][joint] = (target_angle_norm * 2 * np.pi) - np.pi

        ##### update movement history #####

        current_target_array = np.zeros(12, dtype=np.float32)
        action_idx = 0
        for leg_id in ['FL', 'FR', 'BL', 'BR']:
            for joint_name in ['hip', 'upper', 'lower']:
                # Store the raw action value (normalized to [-1, 1])
                current_target_array[action_idx] = result[0, 1, action_idx % 3]  # target angles
                action_idx += 1

        config.PREVIOUS_POSITIONS[0].append(current_target_array)
        
        logging.debug(f"(inference.py): Standard inference completed successfully.\n")

        return target_angles, mid_angles
        
    except Exception as e: # if anything fails...
        logging.error(f"(inference.py): Failed to run standard gait adjustment: {e}\n")
        return {}, {}


########## RUN GAIT ADJUSTMENT RL MODEL WITHOUT IMAGES ########## TODO finish me

def run_gait_adjustment_blind( # function to run gait adjustment RL model without images for speedy processing
        model,
        input_layer,
        output_layer,
        commands,
        intensity
):

    try: # try to run the model without images

        ##### set variables #####

        state = []
        target_angles = {}
        movement_rates = {}

        logging.debug(f"(inference.py): Starting blind gait adjustment inference for commands: {commands}, intensity: {intensity}...\n")

        ##### initialize previous positions #####

        if not config.PREVIOUS_POSITIONS or len(config.PREVIOUS_POSITIONS) == 0:
            logging.warning("PREVIOUS_POSITIONS not initialized, initializing for physical robot...")
            config.PREVIOUS_POSITIONS = []
            robot_history = deque(maxlen=5)
            for _ in range(5):
                robot_history.append(np.zeros(12, dtype=np.float32))
            config.PREVIOUS_POSITIONS.append(robot_history)

        for historical_angles in config.PREVIOUS_POSITIONS[0]:  # Physical robot is always index 0
            state.extend(historical_angles)
        
        ##### encode commands (6D one-hot for movement commands only) #####

        if isinstance(commands, list):
            command_list = commands
        elif isinstance(commands, str):
            command_list = commands.split('+') if commands else []
        else:
            command_list = []
        
        command_encoding = [
            1.0 if 'w' in command_list else 0.0,
            1.0 if 's' in command_list else 0.0,
            1.0 if 'a' in command_list else 0.0,
            1.0 if 'd' in command_list else 0.0,
            1.0 if 'arrowleft' in command_list else 0.0,
            1.0 if 'arrowright' in command_list else 0.0
        ]
        
        state.extend(command_encoding)
        
        ##### prepare fields #####

        intensity_normalized = (float(intensity) - 5.5) / 4.5
        state.append(intensity_normalized)
        state = np.array(state, dtype=np.float32)
        expected_state_size = 67
        if len(state) != expected_state_size:
            raise ValueError(f"State size mismatch: expected {expected_state_size}, got {len(state)}")
        input_vec = state.reshape(1, -1)  # batch dimension

        ##### run inference #####

        result = model([input_vec])[output_layer]

        ##### parse output #####

        action_idx = 0
        for leg_id in ['FL', 'FR', 'BL', 'BR']:
            target_angles[leg_id] = {}
            movement_rates[leg_id] = {}
            
            for joint_name in ['hip', 'upper', 'lower']:
                servo_data = config.SERVO_CONFIG[leg_id][joint_name]
                min_angle = servo_data['FULL_BACK_ANGLE']
                max_angle = servo_data['FULL_FRONT_ANGLE']

                if min_angle > max_angle:
                    min_angle, max_angle = max_angle, min_angle

                target_action = result[0, action_idx]  # get from model output
                target_angle = min_angle + (target_action + 1.0) * 0.5 * (max_angle - min_angle)
                target_angle = np.clip(target_angle, min_angle, max_angle)
                target_angles[leg_id][joint_name] = float(target_angle)
                
                movement_rates[leg_id][joint_name] = 1.0  # legacy support
                
                action_idx += 1
        
        ##### update movement history #####

        current_target_array = np.zeros(12, dtype=np.float32)
        action_idx = 0
        for leg_id in ['FL', 'FR', 'BL', 'BR']:
            for joint_name in ['hip', 'upper', 'lower']:
                # Store the raw action value (already normalized to [-1, 1])
                current_target_array[action_idx] = result[0, action_idx]
                action_idx += 1

        config.PREVIOUS_POSITIONS[0].append(current_target_array)
        
        logging.debug(f"(inference.py): Blind inference completed successfully")
        return target_angles, movement_rates
        
    except Exception as e:
        logging.error(f"(inference.py): Failed to run blind gait adjustment: {e}")
        return {}, {}


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