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

from utilities.maestro import initialize_maestro # import maestro initialization functions


########## CREATE DEPENDENCIES ##########

##### create maestro object #####

MAESTRO = initialize_maestro() # create maestro object





#############################################################
############### FUNDAMENTAL MOVEMENT FUNCTION ###############
#############################################################


########## MOVE A SINGLE SERVO ##########

def set_target(channel, target, speed, acceleration): # function to set target position of a singular servo

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


def pwm_to_angle(pwm, joint_data):
    """
    Map a PWM value to a servo angle in radians using the joint's config.
    Args:
        pwm (float): The PWM value (microseconds)
        joint_data (dict): The joint's config dict, must include FULL_FRONT, FULL_BACK, FULL_FRONT_ANGLE, FULL_BACK_ANGLE
    Returns:
        float: The angle in radians corresponding to the PWM value
    """
    full_front_pwm = joint_data['FULL_FRONT']
    full_back_pwm = joint_data['FULL_BACK']
    full_front_angle = joint_data.get('FULL_FRONT_ANGLE', 0)
    full_back_angle = joint_data.get('FULL_BACK_ANGLE', 0)
    # Linear interpolation
    angle = full_front_angle + (full_back_angle - full_front_angle) * ((pwm - full_front_pwm) / (full_back_pwm - full_front_pwm))
    return angle
