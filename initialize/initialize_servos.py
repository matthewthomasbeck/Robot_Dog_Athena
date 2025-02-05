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

import logging # import logging for debugging

##### import necessary functions #####

from initialize.initialize_maestro import * # import maestro initialization functions


########## CREATE DEPENDENCIES ##########

##### create maestro object #####

MAESTRO = createMaestroConnection() # create maestro connection

##### set servo number count for maestro #####

#NUM_SERVOS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # number of servos as a list of their names

##### set dictionary of servos and their ranges #####

LEG_CONFIG = { # dictionary of leg configurations

    'FL': {'hip': {'servo': 3, 'FULL_BACK': 1236.50, 'FULL_FRONT': 1892.25, 'CUR_POS': 1564.375},
           'upper': {'servo': 5, 'FULL_BACK': 1921.50, 'FULL_FRONT': 1266.00, 'CUR_POS': 1593.75},
           'lower': {'servo': 4, 'FULL_BACK': 1872.75, 'FULL_FRONT': 1148.50, 'CUR_POS': 1510.625}},

    'FR': {'hip': {'servo': 2, 'FULL_BACK': 1613.25, 'FULL_FRONT': 992.00, 'CUR_POS': 1302.625},
           'upper': {'servo': 1, 'FULL_BACK': 1310.00, 'FULL_FRONT': 1921.50, 'CUR_POS': 1615.75},
           'lower': {'servo': 0, 'FULL_BACK': 1231.75, 'FULL_FRONT': 2000.00, 'CUR_POS': 1615.875}},

    'BL': {'hip': {'servo': 8, 'FULL_BACK': 1623.00, 'FULL_FRONT': 1036.00, 'CUR_POS': 1329.5},
           'upper': {'servo': 7, 'FULL_BACK': 2000.00, 'FULL_FRONT': 1354.00, 'CUR_POS': 1677.0},
           'lower': {'servo': 6, 'FULL_BACK': 2000.00, 'FULL_FRONT': 1138.75, 'CUR_POS': 1569.375}},

    'BR': {'hip': {'servo': 11, 'FULL_BACK': 1261.00, 'FULL_FRONT': 1848.25, 'CUR_POS': 1554.625},
           'upper': {'servo': 10, 'FULL_BACK': 1065.25, 'FULL_FRONT': 1701.50, 'CUR_POS': 1383.375},
           'lower': {'servo': 9, 'FULL_BACK': 1221.75, 'FULL_FRONT': 2000.00, 'CUR_POS': 1610.875}},
}





#############################################################
############### FUNDAMENTAL MOVEMENT FUNCTION ###############
#############################################################


########## MOVE A SINGLE SERVO ##########

def setTarget(channel, target): # function to set target position of a singular servo

    ##### move a servo to a desired position using its number and said position #####

    try: # attempt to move desired servo

        target = int(round(target * 4)) # adjust target from microseconds to quarter-microseconds

        command = bytearray([0x84, channel, target & 0x7F, (target >> 7) & 0x7F]) # create command to send to maestro

        MAESTRO.write(command) # write command to maestro
        
        logging.info(f"Moved servo {channel}.\n") # see if servo moved to a position

    except: # if movement failed...

        logging.error("ERROR (initialize_servos.py): Failed to move servo.\n") # print failure statement





####################################################
############### DESTROY DEPENDENCIES ###############
####################################################


########## DISABLE ALL SERVOS ##########

def disableAllServos(): # function to disable servos via code

    ##### make all servos go limp for easy reinitialization #####

    logging.debug("Attempting to disable all servos...\n") # print initialization statement

    ##### old try statement #####

    #try:  # attempt to disable all servos

        #for servo in range(len(NUM_SERVOS)):  # loop through all available servos

            #setTarget(servo, 0)  # set target to 0 to disable the servo

            #logging.info(f"Disabled servo {servo}.")  # print success statement

        #logging.info("\nSuccessfully disabled all servos.\n")  # print success statement

    ##### new try statement #####

    try: # attempt to disable all servos

        for leg, joints in LEG_CONFIG.items(): # loop through each leg

            for joint, config in joints.items(): # loop through each joint

                servo = config['servo'] # get the servo number

                setTarget(servo, 0) # set target to 0 to disable the servo

                logging.info(f"Disabled servo {servo} ({leg} - {joint}).") # print success statement

        logging.info("\nSuccessfully disabled all servos.\n") # print success statement

    ##### exception incase failure to disable servos #####

    except: # if failure to disable any or all servos...

        logging.error("ERROR (initialize_servos.py): Failed to disable servo(s).\n") # print failure statement
