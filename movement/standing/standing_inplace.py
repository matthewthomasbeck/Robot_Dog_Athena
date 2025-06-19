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
import logging # import logging for debugging

##### import necessary functions #####

import utilities.servos as initialize_servos # import servo logic functions
from utilities.config import * # import leg positions config





##########################################################
############### STANDING INPLACE MOVEMENTS ###############
##########################################################


########## NEUTRAL ##########

def neutral_standing_position(leg_id, stride_scalar):

    NEUTRAL_POSITIONS = {
        'FL': FL_NEUTRAL,
        'FR': FR_NEUTRAL,
        'BL': BL_NEUTRAL,
        'BR': BR_NEUTRAL
    }

    neutral = NEUTRAL_POSITIONS[leg_id]

    return {
        'x': neutral['x'] * stride_scalar,
        'y': neutral['y'] * stride_scalar,
        'z': neutral['z']
    }


########## TIPPY TOES ##########

def tippyToesStandingPosition(): # function to set all servos to tippy toes position

    ##### move to tippy toes position #####

    logging.debug("Moving to tippy toes position...\n") # print initialization message

    try: # attempt to move to tippy-toes position

        ##### tippy toes hips #####

        time.sleep(1) # wait a moment before proceeding incase there is movement before this

        logging.debug("Preparing legs...\n")  # print initialization message

        initialize_servos.setTarget(4, 1628) # set front left hip to tippy toes
        initialize_servos.setTarget(2, 1344.25) # set front right hip to tippy toes
        initialize_servos.setTarget(11, 1540) # set rear right hip to tippy toes
        initialize_servos.setTarget(8, 1378.50) # set rear left hip to tippy toes

        logging.info("Prepared legs.\n") # print success statement

        time.sleep(3) # stand by for movement

        logging.debug("Moving to 'tippy toes'...\n") # print initialization message

        ##### front legs #####

        initialize_servos.setTarget(5, 1344.25) # set front left knee to tippy toes
        initialize_servos.setTarget(1, 1902) # set front right knee to tippy toes
        initialize_servos.setTarget(0, 1505.75) # set front right ankle to tippy toes
        initialize_servos.setTarget(3, 1486) # set front left ankle to tippy toes

        ##### back legs #####

        initialize_servos.setTarget(7, 1456.75) # set rear left knee to tippy toes
        initialize_servos.setTarget(10, 1598.75) # set rear right knee to tippy toes
        initialize_servos.setTarget(6, 1730.75) # set rear left ankle to tippy toes
        initialize_servos.setTarget(9, 1393) # set rear right ankle to tippy toes

        ##### update config #####



        logging.info("Moved to 'tippy toes'.\n") # print success statement

    except: # if movement failed...

        # print failure statement
        logging.error("(standing_inplace.py): Failed to move to 'tippy toes' position.\n")


########## SQUATTING ##########

def squattingStandingPosition(): # function to set all servos to full forward position

    ##### move to full forward position #####

    logging.debug("Moving to full forward position...\n") # print initialization message

    try: # attempt to move to tippy toes position

        ##### full forward hips #####

        time.sleep(1) # wait a moment before proceeding incase there is movement before this

        logging.debug("Preparing legs...\n")  # print initialization message

        initialize_servos.setTarget(4, 1148.5) # set front left hip to full forward
        initialize_servos.setTarget(2, 992) # set front right hip to full forward
        initialize_servos.setTarget(11, 1848.25) # set rear right hip to full forward
        initialize_servos.setTarget(8, 1036) # set rear left hip to full forward

        logging.info("Prepared legs.\n") # print success statement

        time.sleep(3) # stand by for movement

        logging.debug("Moving to full forward...\n") # print initialization message

        ##### front legs #####

        initialize_servos.setTarget(5, 1266) # set front left knee to full forward
        initialize_servos.setTarget(1, 1921) # set front right knee to full forward
        initialize_servos.setTarget(3, 1892.25)  # set front left ankle to full forward
        initialize_servos.setTarget(0, 2000) # set front right ankle to full forward

        ##### back legs #####

        initialize_servos.setTarget(7, 1354) # set rear left knee to full forward
        initialize_servos.setTarget(10, 1701.5) # set rear right knee to full forward
        initialize_servos.setTarget(6, 1138.75) # set rear left ankle to full forward
        initialize_servos.setTarget(9, 2000) # set rear right ankle to full forward

        ##### update config #####



        logging.info("Moved to full forward.\n") # print success statement

    except: # if movement failed...

        # print failure statement
        logging.error("(standing_inplace.py): Failed to move to full forward position.\n")