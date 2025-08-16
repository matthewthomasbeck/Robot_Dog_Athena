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

def map_angle_to_servo_position(angle, joint_data): # map radian to pwm

    ##### map angle to servo pulse width #####

    logging.debug(f"(servos.py): Mapping radian {angle} to servo position...\n")
    
    # Get angle limits and PWM limits from joint data
    full_back_angle = joint_data['FULL_BACK_ANGLE']  # radian position for FULL_BACK PWM
    full_front_angle = joint_data['FULL_FRONT_ANGLE']  # radian position for FULL_FRONT PWM
    full_back_pwm = joint_data['FULL_BACK']  # PWM value for full back position
    full_front_pwm = joint_data['FULL_FRONT']  # PWM value for full front position
    
    # Calculate the angle range and PWM range (handle any order)
    angle_range = full_front_angle - full_back_angle
    pwm_range = full_front_pwm - full_back_pwm
    
    # Ensure we don't divide by zero
    if abs(angle_range) < 1e-6:
        logging.error(f"(servos.py): Invalid angle range: {angle_range}")
        return full_back_pwm
    
    # Map the target angle to PWM using linear interpolation
    # Formula: pwm = full_back_pwm + (angle - full_back_angle) * (pwm_range / angle_range)
    pwm = full_back_pwm + (angle - full_back_angle) * (pwm_range / angle_range)
    
    # Clamp PWM to valid range (handle any order of PWM values)
    if full_back_pwm < full_front_pwm:
        pwm = max(full_back_pwm, min(full_front_pwm, pwm))
    else:
        pwm = max(full_front_pwm, min(full_back_pwm, pwm))
    
    logging.debug(f"(servos.py): Angle {angle:.3f} rad -> PWM {pwm:.1f} (range: {full_back_pwm} to {full_front_pwm})\n")
    logging.debug(f"(servos.py): Angle range: {full_back_angle:.3f} to {full_front_angle:.3f} rad\n")
    
    return int(round(pwm)) # return calculated pulse width
