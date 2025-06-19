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


########## ANGLE TO TARGET ##########

def map_angle_to_servo_position(angle, joint_data, angle_neutral, is_inverted=False):
    """
    Maps an angle (in degrees) to a servo pulse width using joint calibration data.
    - angle: target angle in degrees
    - joint_data: dict with FULL_BACK, FULL_FRONT, NEUTRAL
    - angle_neutral: the angle (in degrees) that corresponds to the NEUTRAL pulse
    - is_inverted: True if increasing angle lowers pulse
    """
    full_back = joint_data['FULL_BACK']
    full_front = joint_data['FULL_FRONT']
    neutral_pulse = joint_data['NEUTRAL']

    pulse_range = full_front - full_back
    direction = -1 if is_inverted else 1

    # map angle to pulse
    pulse = neutral_pulse + direction * ((angle - angle_neutral) / 90) * pulse_range
    return int(round(pulse))





####################################################
############### DESTROY DEPENDENCIES ###############
####################################################


########## DISABLE ALL SERVOS ##########

def disableAllServos(): # function to disable servos via code

    ##### make all servos go limp for easy reinitialization #####

    logging.debug("Attempting to disable all servos...\n") # print initialization statement

    try: # attempt to disable all servos

        for leg, joints in SERVO_CONFIG.items(): # loop through each leg

            for joint, config in joints.items(): # loop through each joint

                servo = config['servo'] # get the servo number

                setTarget(servo, 0) # set target to 0 to disable the servo

                logging.info(f"Disabled servo {servo} ({leg} - {joint}).") # print success statement

        logging.info("\nSuccessfully disabled all servos.\n") # print success statement

    ##### exception incase failure to disable servos #####

    except: # if failure to disable any or all servos...

        logging.error("ERROR (initialize_servos.py): Failed to disable servo(s).\n") # print failure statement
