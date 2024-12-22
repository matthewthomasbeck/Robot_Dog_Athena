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

##### import necessary functions #####

from initialize.initialize_servos import * # import servo logic functions





##########################################################
############### STANDING INPLACE MOVEMENTS ###############
##########################################################


########## NEUTRAL ##########

def neutralStandingPosition(): # function to set all servos to neutral standing position

    ##### move to neutral standing position #####

    logging.debug("Moving to neutral standing position...\n") # print initialization message

    try: # attempt to move to neutral standing position

        ##### neutral hips #####

        time.sleep(1) # wait a moment before proceeding incase there is movement before this

        logging.debug("Preparing legs...\n")  # print initialization message

        setTarget(4, 1628) # set front left hip to neutral
        setTarget(2, 1344.25) # set front right hip to neutral
        setTarget(11, 1540) # set rear right hip to neutral
        setTarget(8, 1378.50) # set rear left hip to neutral

        logging.info("Prepared legs.\n") # print success statement

        time.sleep(3) # stand by for movement

        logging.debug("Standing up...\n") # print initialization message

        ##### front legs #####

        setTarget(5, 1344.25) # set front left knee to neutral
        setTarget(1, 1902) # set front right knee to neutral
        setTarget(0, 1505.75) # set front right ankle to neutral
        setTarget(3, 1486) # set front left ankle to neutral

        ##### back legs #####

        setTarget(7, 1456.75) # set rear left knee to neutral
        setTarget(10, 1598.75) # set rear right knee to neutral
        setTarget(6, 1730.75) # set rear left ankle to neutral
        setTarget(9, 1393) # set rear right ankle to neutral

        logging.info("Stood up.\n") # print success statement

    except: # if movement failed...

        # print failure statement
        logging.error("ERROR 6 (standing_inplace.py) Failed to move to neutral standing position.\n")


########## TIPPY TOES ##########

def tippyToesStandingPosition(): # function to set all servos to tippy toes position

    ##### move to tippy toes position #####

    logging.debug("Moving to tippy toes position...\n") # print initialization message

    try: # attempt to move to tippy toes position

        ##### tippy toes hips #####

        time.sleep(1) # wait a moment before proceeding incase there is movement before this

        logging.debug("Preparing legs...\n")  # print initialization message

        setTarget(4, 1628) # set front left hip to tippy toes
        setTarget(2, 1344.25) # set front right hip to tippy toes
        setTarget(11, 1540) # set rear right hip to tippy toes
        setTarget(8, 1378.50) # set rear left hip to tippy toes

        logging.info("Prepared legs.\n") # print success statement

        time.sleep(3) # stand by for movement

        logging.debug("Standing up...\n") # print initialization message

        ##### front legs #####

        setTarget(5, 1344.25) # set front left knee to tippy toes
        setTarget(1, 1902) # set front right knee to tippy toes
        setTarget(0, 1505.75) # set front right ankle to tippy toes
        setTarget(3, 1486) # set front left ankle to tippy toes

        ##### back legs #####

        setTarget(7, 1456.75) # set rear left knee to tippy toes
        setTarget(10, 1598.75) # set rear right knee to tippy toes
        setTarget(6, 1730.75) # set rear left ankle to tippy toes
        setTarget(9, 1393) # set rear right ankle to tippy toes

        logging.info("Stood up.\n") # print success statement

    except: # if movement failed...

        # print failure statement
        logging.error("ERROR 7 (standing_inplace.py) Failed to move to tippy toes position.\n")


########## SQUAT UP ##########



########## SQUAT DOWN ##########
