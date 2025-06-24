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

from utilities.config import INFERENCE_CONFIG





############################################################
############### IMPORT / CREATE DEPENDENCIES ###############
############################################################


########## LOAD AND COMPILE MODEL ##########

# function to load and compile an OpenVINO model
def load_and_compile_model(
        model_xml_path=INFERENCE_CONFIG['MODEL_PATH'], # path to model XML file
        device_name=INFERENCE_CONFIG['TPU_NAME'] # device name for inference (e.g., "CPU", "GPU", "MYRIAD", etc.)
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
        model_bin_path = model_xml_path.replace(".xml", ".bin") # get binary path from XML path (incase needed)
        model = ie.read_model(model=model_xml_path) # read model from XML file
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


########## RUN MODEL AND SHOW FRAME ##########

def run_inference(compiled_model, input_layer, output_layer, camera_process, mjpeg_buffer, run_inference):

    ##### check if model/layers are properly initialized #####

    #logging.debug("(opencv.py): Running inference on camera stream...\n") # very annoying, leave commented

    try: # try to run inference on the camera stream

        chunk = camera_process.stdout.read(4096) # read a chunk of data from camera process stdout

        if not chunk: # if no data is received...
            logging.error("(opencv.py): Camera process stopped sending data.\n")
            return mjpeg_buffer, None

        mjpeg_buffer += chunk # append chunk to buffer
        start_idx = mjpeg_buffer.find(b'\xff\xd8') # start of JPEG frame
        end_idx = mjpeg_buffer.find(b'\xff\xd9') # end of JPEG frame

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx: # if valid JPEG frame found...

            frame_data = mjpeg_buffer[start_idx:end_idx + 2] # extract frame data
            mjpeg_buffer = mjpeg_buffer[end_idx + 2:] # remove processed frame from buffer
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR) # decode frame from buffer

            if frame is not None: # if frame is successfully decoded...

                if not run_inference:  # if inference is not to be run...

                    ##### show frame without inference #####

                    cv2.imshow("video (standard)", frame) # show the frame in a window
                    if cv2.waitKey(1) & 0xFF == ord('q'): # exit on 'q' key press
                        return mjpeg_buffer

                    return mjpeg_buffer, frame_data # return frame_data and not frame to avoid re-encoding

                ##### run inference #####

                try: # if inference is to be run...

                    #logging.debug("(opencv.py): Running inference on frame...\n") # very annoying, leave commented
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert frame to RGB format
                    input_blob = cv2.resize(frame_rgb, (256, 256)).transpose(2, 0, 1) # resize/transpose to match shape
                    input_blob = np.expand_dims(input_blob, axis=0).astype(np.float32) # add batch dim and set float32
                    results = compiled_model([input_blob])[output_layer] # collect inference results

                    for detection in results[0][0]: # iterate through detections

                        confidence = detection[2] # get confidence score

                        if confidence > 0.5: # if confidence is above threshold...

                            xmin, ymin, xmax, ymax = map( # convert coordinates to integers
                                int, detection[3:7] * [
                                    frame.shape[1], frame.shape[0],
                                    frame.shape[1], frame.shape[0]
                                ]
                            )
                            label = f"ID {int(detection[1])}: {confidence:.2f}"
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # draw bounding box
                            cv2.putText( # put label on frame
                                frame,
                                label,
                                (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2
                            )

                    cv2.imshow("video (inference)", frame) # show frame with detections in a window
                    if cv2.waitKey(1) & 0xFF == ord('q'): # exit on 'q' key press
                        return mjpeg_buffer, frame_data # return frame_data and not frame to avoid re-encoding

                except Exception as e:
                    logging.error(f"(opencv.py): Inference error: {e}\n")

            else:
                logging.error("(opencv.py): Failed to decode frame.\n")

        else:

            if len(mjpeg_buffer) > 65536: # if buffer is too large...
                logging.warning("(opencv.py): MJPEG buffer overflow, resetting...\n")
                mjpeg_buffer = b''

    except Exception as err: # catch any unexpected errors
        logging.error(f"(opencv.py): Unexpected error in inference loop: {err}\n")

    return mjpeg_buffer, None
