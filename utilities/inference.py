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
        intensity,
        orientation
):

    pass # TODO get working with camera in future, but focus on basics first


########## RUN GAIT ADJUSTMENT RL MODEL WITHOUT IMAGES ##########

def run_gait_adjustment_blind( # function to run gait adjustment RL model without images for speedy processing
        model,
        input_layer,
        output_layer,
        commands,
        intensity,
        orientation
):

    try: # try to run the model without images

        ##### set variables #####

        state = []
        target_angles = {}
        movement_rates = {}

        # `orientation` is now expected to be a dict produced by accelerometer.get_orientation_vectors()
        # containing: base_lin_vel (3), base_ang_vel (3), projected_gravity (3).
        logging.info(f"(inference) orientation vectors: {orientation}")
        logging.debug(f"(inference.py): Starting blind gait adjustment inference for commands: {commands}, intensity: {intensity}...\n")

        ##### build Isaac Lab 48-dim observation #####

        # 1) base_lin_vel (3)
        base_lin_vel = np.array(orientation.get("base_lin_vel", [0.0, 0.0, 0.0]), dtype=np.float32)
        if base_lin_vel.shape != (3,):
            base_lin_vel = base_lin_vel.reshape(-1)[:3].astype(np.float32)

        # 2) base_ang_vel (3) (rad/s)
        base_ang_vel = np.array(orientation["base_ang_vel"], dtype=np.float32)
        if base_ang_vel.shape != (3,):
            base_ang_vel = base_ang_vel.reshape(-1)[:3].astype(np.float32)

        # 3) projected_gravity (3) (unit vector)
        projected_gravity = np.array(orientation["projected_gravity"], dtype=np.float32)
        if projected_gravity.shape != (3,):
            projected_gravity = projected_gravity.reshape(-1)[:3].astype(np.float32)

        # 4) velocity_commands (3) from discrete commands (w/s/a/d/arrowleft/arrowright)
        if isinstance(commands, list):
            command_list = commands
        elif isinstance(commands, str):
            command_list = commands.split('+') if commands else []
        else:
            command_list = []

        lin_vel_x = 0.0
        lin_vel_y = 0.0
        ang_vel_z = 0.0

        # Forward/backward
        if 'w' in command_list:
            lin_vel_x = 0.4
        elif 's' in command_list:
            lin_vel_x = -0.4

        # Left/right strafe
        if 'a' in command_list:
            lin_vel_y = 0.3
        elif 'd' in command_list:
            lin_vel_y = -0.3

        # Rotation
        if 'arrowleft' in command_list:
            ang_vel_z = 0.5
        elif 'arrowright' in command_list:
            ang_vel_z = -0.5

        # Clamp to Isaac Lab typical ranges
        lin_vel_x = float(np.clip(lin_vel_x, -0.6, 0.6))
        lin_vel_y = float(np.clip(lin_vel_y, -0.6, 0.6))
        ang_vel_z = float(np.clip(ang_vel_z, -0.8, 0.8))
        velocity_commands = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)

        # 5) joint_pos (12): joint positions relative to default positions (radians)
        joint_pos_abs = []
        for leg_id in ['FL', 'FR', 'BL', 'BR']:
            for joint_name in ['hip', 'upper', 'lower']:
                joint_pos_abs.append(float(config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE']))
        joint_pos_abs = np.array(joint_pos_abs, dtype=np.float32)

        default_positions = np.array([
            0.1465, -0.1465, 0.0,    # FL
            -0.1465, 0.1465, 0.0,    # FR
            -0.1465, 0.1465, 0.0,    # BL
            0.1465, -0.1465, 0.0     # BR
        ], dtype=np.float32)
        joint_pos = joint_pos_abs - default_positions

        # 6) joint_vel (12): finite-difference of commanded positions
        if not hasattr(config, "PREV_JOINT_POS_ABS") or config.PREV_JOINT_POS_ABS is None:
            config.PREV_JOINT_POS_ABS = joint_pos_abs.copy()
        prev_joint_pos_abs = np.array(config.PREV_JOINT_POS_ABS, dtype=np.float32)
        dt = 0.033  # ~30 Hz
        joint_vel = (joint_pos_abs - prev_joint_pos_abs) / dt

        # 7) actions (12): previous action output in [-1, 1]
        if not hasattr(config, "LAST_ACTION") or config.LAST_ACTION is None:
            config.LAST_ACTION = np.zeros(12, dtype=np.float32)
        last_action = np.array(config.LAST_ACTION, dtype=np.float32)
        if last_action.shape != (12,):
            last_action = last_action.reshape(-1)[:12].astype(np.float32)

        # Concatenate into 48-dim observation
        obs = np.concatenate([
            base_lin_vel,       # 3
            base_ang_vel,       # 3
            projected_gravity,  # 3
            velocity_commands,  # 3
            joint_pos,          # 12
            joint_vel,          # 12
            last_action,        # 12
        ]).astype(np.float32)

        expected_state_size = 48
        if obs.shape[0] != expected_state_size:
            raise ValueError(f"State size mismatch: expected {expected_state_size}, got {obs.shape[0]}")

        input_vec = obs.reshape(1, -1)  # batch dimension

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

        ##### update state memory for next step #####

        # Store last action (raw policy output in [-1, 1]) and previous joint positions for velocity.
        config.LAST_ACTION = np.array(result[0, :12], dtype=np.float32).copy()
        config.PREV_JOINT_POS_ABS = joint_pos_abs.copy()
        
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