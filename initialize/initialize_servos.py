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

##### import necessary functions #####

from initialize.initialize_maestro import * # import maestro initialization functions


########## CREATE DEPENDENCIES ##########

##### create maestro object #####

MAESTRO = createMaestroConnection() # create maestro connection

##### set servo number count for maestro #####

NUM_SERVOS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # number of servos as a list of their names





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

        logging.error("ERROR 4 (initialize_servos.py): Failed to move servo.\n") # print failure statement





####################################################
############### DESTROY DEPENDENCIES ###############
####################################################


########## DISABLE ALL SERVOS ##########

def disableAllServos(): # function to disable servos via code

    ##### make all servos go limp for easy reinitialization #####

    logging.debug("Attempting to disable all servos...\n") # print initialization statement

    try: # attempt to disable all servos

        for servo in range(len(NUM_SERVOS)): # loop through all available servos

            setTarget(servo, 0) # set target to 0 to disable the servo

            logging.info(f"Disabled servo {servo}.") # print success statement

        logging.info("\nSuccessfully disabled all servos.\n") # print success statement


    except: # if failure to disable any or all servos...

        logging.error("ERROR 5 (initialize_servos.py): Failed to disable servo(s).\n") # print failure statement


