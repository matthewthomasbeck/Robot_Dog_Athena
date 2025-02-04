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

import time # import time library for time functions
import os # import os library for system functions
import sys # import sys library for system functions
import logging # import logging for debugging

##### import necessary functions #####

from initialize.initialize_servos import * # import servo logic functions





#################################################
############### WALKING FUNCTIONS ###############
#################################################


########## MOVE LEG ##########

def moveLeg(leg, action): # function to move a leg to a desired position

    ##### check if leg is valid #####

    if leg not in LEG_CONFIG: # if leg is not in leg configuration...

        logging.error(f"ERROR 19 (manual_walking.py): Invalid leg {leg} found.\n") # print error message

        return # return to stop the function



    for joint, servo_info in LEG_CONFIG[leg].items():

        servo = servo_info['servo']
        full_back = servo_info['FULL_BACK']
        full_front = servo_info['FULL_FRONT']

        if action == 'LIFT':

            setTarget(servo, full_back)  # Move joint to LIFT position

        elif action == 'FORWARD':

            setTarget(servo, full_front)  # Move joint to FORWARD position

        elif action == 'DOWN':

            setTarget(servo, (full_front + full_back) / 2)  # Move joint to MID position


########## MANUAL FORWARD ##########

def manualForward(): # function to move forward manually

    ##### set variables #####

    gait_sequence = [ # define gait sequence for forward movement

        # step 1: lift front-left (FL) and back-right (BR)
        {'FL': 'LIFT', 'BR': 'LIFT'},
        {'FL': 'FORWARD', 'BR': 'FORWARD'},
        {'FL': 'DOWN', 'BR': 'DOWN'},

        # step 2: lift front-right (FR) and back-left (BL)
        {'FR': 'LIFT', 'BL': 'LIFT'},
        {'FR': 'FORWARD', 'BL': 'FORWARD'},
        {'FR': 'DOWN', 'BL': 'DOWN'},
    ]

    ##### move forward #####

    for step in gait_sequence: # iterate through each step in the gait sequence

        for leg, action in step.items(): # iterate through each leg in the step

            moveLeg(leg, action) # move leg to desired location

        time.sleep(0.2) # wait for smoother motion