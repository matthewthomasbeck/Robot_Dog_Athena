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

##### import necessary libraries #####

import cv2
import subprocess
import numpy as np
import os
import signal
import logging


########## CREATE DEPENDENCIES ##########

##### create camera object #####

camera_process = subprocess.Popen(["rpicam-jpeg", "-o", "-"], stdout=subprocess.PIPE)





#################################################
############### INITIALIZE CAMERA ###############
#################################################


########## INITIALIZE CAMERA ##########

def start_camera_process():

    return subprocess.Popen(["rpicam-jpeg", "-o", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


########## PROCESS A FRAME ##########

def process_frame(camera_process):  # Pass camera process explicitly

    jpg = b''
    while True:
        chunk = camera_process.stdout.read(1024)
        if not chunk:
            logging.error("ERROR 8 (initialize_camera.py): Camera process stopped.")
            # Handle zombie cleanup
            os.kill(camera_process.pid, signal.SIGKILL)
            camera_process.wait()
            return None
        jpg += chunk
        a = jpg.find(b'\xff\xd8')  # JPEG start marker
        b = jpg.find(b'\xff\xd9')  # JPEG end marker
        if a != -1 and b != -1:
            frame = jpg[a:b+2]
            jpg = jpg[b+2:]
            return cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    return None