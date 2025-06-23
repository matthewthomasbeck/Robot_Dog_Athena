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

from utilities.maestro import * # import maestro initialization functions
from utilities.config import SERVO_CONFIG # import servo configuration data


########## CREATE DEPENDENCIES ##########

##### create maestro object #####

MAESTRO = createMaestroConnection() # create maestro connection





#############################################################
############### FUNDAMENTAL MOVEMENT FUNCTION ###############
#############################################################


########## MOVE A SINGLE SERVO ##########

def setTarget(channel, target, speed, acceleration): # function to set target position of a singular servo

    ##### move a servo to a desired position using its number and said position #####

    logging.debug(f"(servos.py): Attempting to move servo {channel} to target {target} with speed {speed} and acceleration {acceleration}...\n")

    try: # attempt to move desired servo

        target = int(round(target * 4)) # convert target from microseconds to quarter-microseconds
        speed = max(0, min(16383, speed)) # ensure speed is within valid range
        acceleration = max(0, min(255, acceleration)) # ensure acceleration is within valid range
        speed_command = bytearray([0x87, channel, speed & 0x7F, (speed >> 7) & 0x7F]) # create speed command
        MAESTRO.write(speed_command) # send speed command to maestro

        # create acceleration command
        accel_command = bytearray([0x89, channel, acceleration & 0x7F, (acceleration >> 7) & 0x7F])
        MAESTRO.write(accel_command) # send acceleration command to maestro
        command = bytearray([0x84, channel, target & 0x7F, (target >> 7) & 0x7F]) # create target position command
        MAESTRO.write(command) # send target position command to maestro

    except:
        logging.error("(servos.py): Failed to move servo.\n") # print failure statement


########## ANGLE TO TARGET ##########

# function to map an angle to a servo pulse width
def map_angle_to_servo_position(angle, joint_data, angle_neutral, is_inverted=False):

    ##### map angle to servo pulse width #####

    logging.debug(f"(servos.py): Mapping angle {angle} to servo position...\n")
    full_back = joint_data['FULL_BACK'] # get full back pulse width from joint data
    full_front = joint_data['FULL_FRONT'] # get full front pulse width from joint data
    neutral_pulse = joint_data['NEUTRAL'] # get neutral pulse width from joint data
    pulse_range = full_front - full_back # calculate pulse width range
    direction = -1 if is_inverted else 1 # determine direction based on inversion flag
    pulse = neutral_pulse + direction * ((angle - angle_neutral) / 90) * pulse_range # pulse width based on angle

    return int(round(pulse)) # return calculated pulse width





####################################################
############### DESTROY DEPENDENCIES ###############
####################################################


########## DISABLE ALL SERVOS ##########

def disableAllServos(): # function to disable servos via code

    ##### make all servos go limp for easy reinitialization #####

    logging.debug("(servos.py): Attempting to disable all servos...\n")

    try: # attempt to disable all servos

        for leg, joints in SERVO_CONFIG.items(): # loop through each leg

            for joint, config in joints.items(): # loop through each joint
                servo = config['servo'] # get the servo number
                setTarget(servo, 0) # set target to 0 to disable the servo
                logging.info(f"(servos.py) Disabled servo {servo} ({leg} - {joint}).\n")

        logging.info("(servos.py): Successfully disabled all servos.\n")

    except:
        logging.error("(servos.py): Failed to disable servo(s).\n")
