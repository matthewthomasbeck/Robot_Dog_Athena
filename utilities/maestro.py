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
def establish_serial_connection(serial_port_name, serial_baud_rate, serial_timeout):

    ##### attempt to establish serial connection and return maestro object #####

    logging.debug("(maestro.py): Establishing serial connection with maestro...\n")

    try: # try to establish serial connection

        maestro = serial.Serial( # set maestro connection object

            serial_port_name, # port to connect to
            baudrate=serial_baud_rate, # baud rate for serial connection
            timeout=serial_timeout # amount of time to wait for response
        )

        logging.info("(maestro.py): Successfully established connection with maestro.\n")

        return maestro # return maestro connection object

    except:
        logging.error("(maestro.py): Failed to establish serial connection to maestro.\n")
        return 1


########## SEND BAUD RATE ##########

def sendBaudRateIndication(maestro): # function to send baud rate indication to Maestro to establish communication

    ##### attempt to send baud rate indication #####

    logging.debug("(maestro.py): Sending baud rate initializer...\n") # print initialization statement

    try: # try to send baud rate indication

        maestro.write(bytearray([0xAA])) # send 0xAA byte to indicate baud rate
        time.sleep(0.1) # give time for maestro to process baud rate indication
        logging.info("(maestro.py): Successfully initialized maestro with baud rate.\n")
        return 0

    except: # if baud rate indication failed...

        logging.error("(maestro.py): Failed to send baud rate indication to maestro.\n") # print failure
        return 1


########## CREATE MAESTRO CONNECTION ##########

def createMaestroConnection(): # function to create maestro connection

    ##### establish serial connection to maestro #####

    logging.debug("(maestro.py): Attempting to establish connection with maestro...\n")

    # establish maestro serial connection
    MAESTRO = establish_serial_connection(serialPortName, serialBaudRate, serialTimeout)

    if MAESTRO == 1: # if maestro connection failed...
        raise SystemExit(1) # kill process

    ##### send baud rate indication to maestro #####

    if sendBaudRateIndication(MAESTRO) == 2: # if baud rate indication failed...
        raise SystemExit(2) # kill process

    return MAESTRO


########## CLOSE MAESTRO CONNECTION ##########

def closeMaestroConnection(maestro): # function to close serial connection to maestro

    ##### attempt to close maestro connection #####

    logging.debug("(maestro.py): Attempting to close connection with maestro...\n")

    try: # try to close maestro connection

        maestro.close() # close maestro connection
        logging.info("(maestro.py): Successfully closed connection with maestro.\n")

    except:
        logging.error("(maestro.py): Failed to close serial connection to maestro.\n")
