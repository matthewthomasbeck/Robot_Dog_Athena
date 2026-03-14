##################################################################################
# Copyright (c) 2025 Matthew Thomas Beck                                         #
#                                                                                #
# Licensed under the Creative Commons Attribution-NonCommercial 4.0              #
# International (CC BY-NC 4.0). Personal and educational use is permitted.       #
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

    ##### set constants #####

    rad_per_microsecond = 0.001997 # radian change per microsecond of pwm

    ##### set variables #####

    neutral_pwm = joint_data['NEUTRAL'] # pwm value for neutral position
    neutral_angle = joint_data.get('NEUTRAL_ANGLE', 0.0) # radian position for neutral pwm

    full_back_angle = joint_data['FULL_BACK_ANGLE'] # radian position for FULL_BACK pwm
    full_front_angle = joint_data['FULL_FRONT_ANGLE'] # radian position for FULL_FRONT pwm
    full_back_pwm = joint_data['FULL_BACK'] # value for full back position
    full_front_pwm = joint_data['FULL_FRONT'] # value for full front position

    logging.debug(f"(servos.py): mapping angle {angle:.4f} rad using neutral "
                  f"({neutral_angle:.4f} rad -> {neutral_pwm:.1f} us) "
                  f"and rate {rad_per_microsecond} rad/us...\n")

    ##### infer servo direction from calibration #####

    sign = 1.0 # default sign
    slope_ref = None # reference slope for angle vs pwm near neutral

    if (abs(full_front_pwm - neutral_pwm) > 1e-3 and
        abs(full_front_angle - neutral_angle) > 1e-6): # prefer front side if valid

        slope_ref = (full_front_angle - neutral_angle) / (full_front_pwm - neutral_pwm)

    elif (abs(full_back_pwm - neutral_pwm) > 1e-3 and
          abs(full_back_angle - neutral_angle) > 1e-6): # otherwise use back side

        slope_ref = (full_back_angle - neutral_angle) / (full_back_pwm - neutral_pwm)

    if slope_ref is not None: # if able to infer direction...
        sign = 1.0 if slope_ref > 0 else -1.0 # set sign based on calibration
    else:
        logging.warning("(servos.py): could not infer servo direction from calibration; defaulting to positive direction.\n")

    ##### map angle to pwm using neutral anchor and fixed rate #####

    angle_delta = angle - neutral_angle # angle offset from neutral

    if abs(rad_per_microsecond) < 1e-9: # if dividing by zero...
        logging.error("(servos.py): invalid radian rate; defaulting to neutral pwm.\n")
        pwm = neutral_pwm
    else:
        pwm = neutral_pwm + (angle_delta / (sign * rad_per_microsecond)) # map via fixed rad/us rate

    ##### clamp pwm to calibrated bounds #####

    if full_back_pwm < full_front_pwm: # if back value scalar less than front value scalar...
        pwm = max(full_back_pwm, min(full_front_pwm, pwm)) # clamp pwm to valid range

    else: # if back value scalar greater than front value scalar...
        pwm = max(full_front_pwm, min(full_back_pwm, pwm)) # clamp pwm to valid range

    logging.debug(f"(servos.py): angle {angle:.3f} rad -> pwm {pwm:.1f} us "
                  f"(range: {min(full_back_pwm, full_front_pwm)} to {max(full_back_pwm, full_front_pwm)})\n")
    logging.debug(f"(servos.py): angle range (config): {full_back_angle:.3f} to {full_front_angle:.3f} rad\n")

    return int(round(pwm)) # return calculated pulse width


########## RADIAN TO SERVO SPEED ##########

def map_radian_to_servo_speed(radian_speed): # function to map radian speed to servo speed

    ##### mao radian speed to servo speed #####

    logging.debug(f"(servos.py): Mapping radian speed {radian_speed} to servo speed...\n")

    radian_speed = max(0.0, min(9.52, radian_speed)) # clamp radian speed to valid range (0 to 9.52 rad/s)
    servo_speed = (radian_speed / 9.52) * 16383 # map radian speed to servo speed (0 to 16383)
    servo_speed = int(round(servo_speed)) # round servo speed to nearest integer
    servo_speed = max(0, min(16383, servo_speed)) # ensure servo speed is within valid range
    
    logging.debug(f"(servos.py): Radian speed {radian_speed:.3f} rad/s -> Servo speed {servo_speed}\n")
    
    return servo_speed
