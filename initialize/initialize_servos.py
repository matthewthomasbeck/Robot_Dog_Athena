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

##### set dictionary of servos and their ranges #####

LEG_CONFIG = { # dictionary of leg configurations

    'FL': {'hip': {'servo': 3, 'FULL_BACK': 1236.50, 'FULL_FRONT': 1892.25, 'NEUTRAL': 1564.375, 'CUR_POS': 1564.375, 'DIR': 0, 'MOVED': False},
           'upper': {'servo': 5, 'FULL_BACK': 1921.50, 'FULL_FRONT': 1266.00, 'NEUTRAL': 1593.75, 'CUR_POS': 1593.75, 'DIR': 0, 'MOVED': False},
           'lower': {'servo': 4, 'FULL_BACK': 1872.75, 'FULL_FRONT': 1148.50, 'NEUTRAL': 1510.625, 'CUR_POS': 1510.625, 'DIR': 0, 'MOVED': False}},

    'FR': {'hip': {'servo': 2, 'FULL_BACK': 1613.25, 'FULL_FRONT': 992.00, 'NEUTRAL': 1302.625, 'CUR_POS': 1302.625, 'DIR': 0, 'MOVED': False},
           'upper': {'servo': 1, 'FULL_BACK': 1310.00, 'FULL_FRONT': 1921.50, 'NEUTRAL': 1615.75, 'CUR_POS': 1615.75, 'DIR': 0, 'MOVED': False},
           'lower': {'servo': 0, 'FULL_BACK': 1231.75, 'FULL_FRONT': 2000.00, 'NEUTRAL': 1615.875, 'CUR_POS': 1615.875, 'DIR': 0, 'MOVED': False}},

    'BL': {'hip': {'servo': 8, 'FULL_BACK': 1623.00, 'FULL_FRONT': 1036.00, 'NEUTRAL': 1329.5, 'CUR_POS': 1329.5, 'DIR': 0, 'MOVED': False},
           'upper': {'servo': 7, 'FULL_BACK': 2000.00, 'FULL_FRONT': 1354.00, 'NEUTRAL': 1677.0, 'CUR_POS': 1677.0, 'DIR': 0, 'MOVED': False},
           'lower': {'servo': 6, 'FULL_BACK': 2000.00, 'FULL_FRONT': 1138.75, 'NEUTRAL': 1569.375, 'CUR_POS': 1569.375, 'DIR': 0, 'MOVED': False}},

    'BR': {'hip': {'servo': 11, 'FULL_BACK': 1261.00, 'FULL_FRONT': 1848.25, 'NEUTRAL': 1554.625, 'CUR_POS': 1554.625, 'DIR': 0, 'MOVED': False},
           'upper': {'servo': 10, 'FULL_BACK': 1065.25, 'FULL_FRONT': 1701.50, 'NEUTRAL': 1383.375, 'CUR_POS': 1383.375, 'DIR': 0, 'MOVED': False},
           'lower': {'servo': 9, 'FULL_BACK': 1221.75, 'FULL_FRONT': 2000.00, 'NEUTRAL': 1610.875, 'CUR_POS': 1610.875, 'DIR': 0, 'MOVED': False}},
}





#############################################################
############### FUNDAMENTAL MOVEMENT FUNCTION ###############
#############################################################


########## MOVE A SINGLE SERVO ##########

def setTarget(channel, target, speed, acceleration): # function to set target position of a singular servo

    ##### move a servo to a desired position using its number and said position #####

    try: # attempt to move desired servo

        target = int(round(target * 4)) # convert target from microseconds to quarter-microseconds

        # ensure speed and acceleration are within valid ranges
        speed = max(0, min(16383, speed))
        acceleration = max(0, min(255, acceleration))

        # create speed command
        speed_command = bytearray([0x87, channel, speed & 0x7F, (speed >> 7) & 0x7F])
        MAESTRO.write(speed_command)

        # create acceleration command
        accel_command = bytearray([0x89, channel, acceleration & 0x7F, (acceleration >> 7) & 0x7F])
        MAESTRO.write(accel_command)

        # create and send target position command
        command = bytearray([0x84, channel, target & 0x7F, (target >> 7) & 0x7F])
        MAESTRO.write(command)
        
        #logging.info(f"Moved servo {channel} to {target}.\n") # see if servo moved to a position

    except: # if movement failed...

        logging.error("ERROR (initialize_servos.py): Failed to move servo.\n") # print failure statement





####################################################
############### DESTROY DEPENDENCIES ###############
####################################################


########## DISABLE ALL SERVOS ##########

def disableAllServos(): # function to disable servos via code

    ##### make all servos go limp for easy reinitialization #####

    logging.debug("Attempting to disable all servos...\n") # print initialization statement

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
