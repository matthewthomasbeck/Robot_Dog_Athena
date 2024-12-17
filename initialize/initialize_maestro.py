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
import serial # import serial for maestro control
import logging # import logging library for debugging


########## CREATE DEPENDENCIES ##########

##### set serial connection #####

serialPortName = '/dev/serial0' # set serial port name to first available
serialBaudRate = 9600 # set baud rate for serial connection
serialTimeout = 1 # set timeout for serial connection





##################################################
############### INITIALIZE MAESTRO ###############
##################################################


########## ESTABLISH SERIAL CONNECTION ##########

# function to establish serial connection to maestro
def establishSerialConnection(serialPortName, serialBaudRate, serialTimeout):

    ##### attempt to establish serial connection and return maestro object #####

    logging.debug("Establishing serial connection with maestro...\n") # print initialization statement

    try: # try to establish serial connection

        maestro = serial.Serial( # set maestro connection object

            serialPortName, # port to connect to
            baudrate=serialBaudRate, # baud rate for serial connection
            timeout=serialTimeout # amount of time to wait for response
        )

        logging.info("Successfully established connection with maestro.\n") # print success statement

        return maestro # return maestro connection object

    ##### throw error and return error code if conntection fails #####

    except: # if connection failed...

        # print failure statement
        logging.error("ERROR 1 (initialize_maestro.py): Failed to establish serial connection to maestro.\n")

        return 1 # return error 1


########## SEND BAUD RATE ##########

def sendBaudRateIndication(maestro): # function to send baud rate indication to Maestro to establish communication

    ##### attempt to send baud rate indication #####

    logging.debug("Sending baud rate initializer...\n") # print initialization statement

    try: # try to send baud rate indication

        maestro.write(bytearray([0xAA])) # send 0xAA byte to indicate baud rate

        time.sleep(0.1) # give time for maestro to process baud rate indication

        logging.info("Successfully initialized maestro with baud rate.\n") # print success statement

        return 0 # return success code

    ##### throw error and return error code if baud rate indication fails #####

    except: # if baud rate indication failed...

        logging.error("ERROR 2 (initialize_maestro.py): Failed to send baud rate indication to maestro.\n") # print failure

        return 2 # return error 2


########## CREATE MAESTRO CONNECTION ##########

def createMaestroConnection(): # function to create maestro connection

    ##### establish serial connection to maestro #####

    # establish maestro serial connection
    MAESTRO = establishSerialConnection(serialPortName, serialBaudRate, serialTimeout)

    if MAESTRO == 1: # if maestro connection failed...

        raise SystemExit(1) # kill process

    ##### send baud rate indication to maestro #####

    if sendBaudRateIndication(MAESTRO) == 2: # send baud rate indication to maestro to establish communication

        raise SystemExit(2) # kill process

    return MAESTRO # return maestro connection object


########## CLOSE MAESTRO CONNECTION ##########

def closeMaestroConnection(maestro): # function to close serial connection to maestro

    ##### attempt to close maestro connection #####

    logging.debug("Attempting to close connection with maestro...\n") # print initialization statement

    try: # try to close maestro connection

        maestro.close() # close maestro connection

        logging.info("Successfully closed connection with maestro.\n") # print success statement

    ##### throw error and return error code if closing connection fails #####

    except: # if closing connection failed...

        # print failure statement
        logging.error("ERROR 3 (initialize_maestro.py): Failed to close serial connection to maestro.\n")
