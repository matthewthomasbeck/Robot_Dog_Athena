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

from openvino.runtime import Core # import OpenVINO runtime
import numpy as np # import NumPy for array manipulation
import cv2 # import OpenCV for image processing
import logging # import logging for logging messages





############################################################
############### IMPORT / CREATE DEPENDENCIES ###############
############################################################


########## LOAD AND COMPILE MODEL ##########

def load_and_compile_model(model_xml_path, device_name="MYRIAD"): # function to load and compile an OpenVINO model

    try: # try to load and compile the model

        ie = Core() # check for devices
        model_bin_path = model_xml_path.replace(".xml", ".bin") # get binary path from XML path (incase needed)
        model = ie.read_model(model=model_xml_path) # read model from XML file
        compiled_model = ie.compile_model(model=model, device_name=device_name) # compile model for specified device
        input_layer = compiled_model.input(0) # get input layer of compiled model
        output_layer = compiled_model.output(0) # get output layer of compiled model

        logging.info(f"Model loaded and compiled on {device_name}.\n")
        logging.info(f"Model input shape: {input_layer.shape}\n")

        return compiled_model, input_layer, output_layer

    except Exception as e:
        print(f"ERROR (initialize_opencv.py): Failed to load/compile model: {e}\n")
        return None, None, None


########## TEST MODEL ##########

def test_with_dummy_input(compiled_model, input_layer, output_layer): # function to test the model with a dummy input

    ##### check if model/layers are properly initialized #####

    # if model/layers not properly initialized...
    if compiled_model is None or input_layer is None or output_layer is None:

        print("ERROR (initialize_opencv.py): Model is not properly initialized.\n")

        return

    ##### run dummy input through the model #####

    try: # try to run a dummy input through the model

        dummy_input_shape = input_layer.shape # get the shape of the input layer
        dummy_input = np.ones(dummy_input_shape, dtype=np.float32) # create a dummy input with ones
        _ = compiled_model([dummy_input])[output_layer] # run the model but don't use output

        logging.info("Dummy input test passed.\n")

    except Exception as e:

        logging.error(f"(opencv.py): Dummy input test failed: {e}\n")


########## RUN MODEL AND SHOW FRAME ##########

def decode_and_show_frame(mjpeg_buffer): # function to run model and show a frame from an MJPEG buffer

    ##### find JPEG markers #####

    start_idx = mjpeg_buffer.find(b'\xff\xd8') # JPEG start
    end_idx = mjpeg_buffer.find(b'\xff\xd9') # JPEG end

    ##### process frame if valid indices found #####

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx: # if indices are valid...

        frame_data = mjpeg_buffer[start_idx:end_idx + 2] # extract JPEG data
        updated_buffer = mjpeg_buffer[end_idx + 2:] # remove processed frame
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR) # decode JPEG data

        if frame is not None: # if decoding was successful...

            cv2.imshow("Robot Camera", frame) # show the frame

            return updated_buffer

        else:  # if decoding failed...

            print("WARNING (initialize_opencv.py): Failed to decode frame.")

            return updated_buffer
    else:

        return mjpeg_buffer # if no valid JPEG markers found, return buffer unchanged
