##################################################################################
# Copyright (c) 2025 Matthew Thomas Beck                                         #
#                                                                                #
# Personal and educational use only. This code and its associated files may be   #
# copied, modified, and distributed by individuals for non-commercial purposes. #
# Commercial use by companies or for-profit entities is prohibited.              #
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


########## RADIAN TO SERVO SPEED ##########

def map_radian_to_servo_speed(radian_speed):
    """
    Map radian velocity (0-9.52 rad/s) to servo speed (0-16383).
    
    Args:
        radian_speed: Velocity in radians per second (0.0 to 9.52)
    
    Returns:
        int: Servo speed value from 0 to 16383
    """
    logging.debug(f"(servos.py): Mapping radian speed {radian_speed} to servo speed...\n")
    
    # Clamp input to valid range
    radian_speed = max(0.0, min(9.52, radian_speed))
    
    # Map 0-9.52 rad/s to 0-16383 servo speed
    # Linear mapping: servo_speed = (radian_speed / 9.52) * 16383
    servo_speed = (radian_speed / 9.52) * 16383
    
    # Convert to integer and clamp to valid range
    servo_speed = int(round(servo_speed))
    servo_speed = max(0, min(16383, servo_speed))
    
    logging.debug(f"(servos.py): Radian speed {radian_speed:.3f} rad/s -> Servo speed {servo_speed}\n")
    
    return servo_speed
