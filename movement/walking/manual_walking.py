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
import os # import os library for system functions
import sys # import sys library for system functions
import logging # import logging for debugging

##### import necessary functions #####

import initialize.initialize_servos as initialize_servos # import servo logic functions





#################################################
############### WALKING FUNCTIONS ###############
#################################################


########## CALCULATE INTENSITY ##########

def interpretIntensity(intensity, full_back, full_front): # function to interpret intensity

    ##### find intensity value to calculate arc later #####

    # find intensity by dividing the difference between full_back and full front,
    # converting to positive, dividing by 10, and multiplying by intensity
    arc_length = (abs(full_back - full_front) / 10) * intensity

    ##### find speed and acceleration #####

    if intensity == 1 or intensity == 2:
        speed = int(((16383 / 5) / 10) * intensity)
        acceleration = int(((255 / 5) / 10) * intensity)
    elif intensity == 3 or intensity == 4:
        speed = int(((16383 / 4) / 10) * intensity)
        acceleration = int(((255 / 4) / 10) * intensity)
    elif intensity == 5 or intensity == 6:
        speed = int(((16383 / 3) / 10) * intensity)
        acceleration = int(((255 / 3) / 10) * intensity)
    elif intensity == 7 or intensity == 8:
        speed = int(((16383 / 2) / 10) * intensity)
        acceleration = int(((255 / 2) / 10) * intensity)
    else:
        speed = int((16383 / 10) * intensity)
        acceleration = int((255 / 10) * intensity)

    ##### return arc length speed and acceleration #####

    return arc_length, speed, acceleration # return movement parameters


########## OSCILLATE LEGS ##########

def oscillateLegs(intensity): # function to oscillate one servo

    ##### set vairables #####

    upper_leg_servos = { # define upper leg servos

        "FL": initialize_servos.LEG_CONFIG['FL']['upper'], # front left
        "FR": initialize_servos.LEG_CONFIG['FR']['upper'], # front right
        "BL": initialize_servos.LEG_CONFIG['BL']['upper'], # back left
        "BR": initialize_servos.LEG_CONFIG['BR']['upper'], # back right
    }

    arc_lengths = [] # store all arc lengths for uniform movement distance
    speeds = [] # store all speeds for uniform movement speed
    accelerations = [] # store all accelerations for uniform movement acceleration

    ##### oscillate upper legs #####

    for leg, servo_data in upper_leg_servos.items():

        full_back = servo_data['FULL_BACK'] # get full back position
        full_front = servo_data['FULL_FRONT'] # get full front position
        arc_length, speed, acceleration = interpretIntensity(intensity, full_back, full_front) # get movement parameters
        arc_lengths.append(arc_length) # append arc length to list
        speeds.append(speed) # append speed to list
        accelerations.append(acceleration) # append acceleration to list
        servo_data['MOVED'] = False

    min_arc_length = min(arc_lengths) # get minimum arc length
    min_speed = min(speeds) # get minimum speed
    min_acceleration = min(accelerations) # get minimum acceleration

    for leg, servo_data in upper_leg_servos.items():

        full_back = servo_data['FULL_BACK']  # get full back position
        full_front = servo_data['FULL_FRONT'] # get full front position
        neutral_position = servo_data['NEUTRAL'] # get neutral position

        if full_back < full_front: # if back position greater number...

            max_limit = neutral_position + (min_arc_length / 2)
            min_limit = neutral_position - (min_arc_length / 2)

        else: # if front position greater number...

            max_limit = neutral_position - (min_arc_length / 2)
            min_limit = neutral_position + (min_arc_length / 2)

        # Ensure DIR is set correctly at the start
        if servo_data['DIR'] == 0:
            if leg in ["FL", "BR"]:  # FL & BR move forward first
                servo_data['DIR'] = 1
            else:  # FR & BL move backward first
                servo_data['DIR'] = -1

        # Move in the current direction
        new_pos = servo_data['CUR_POS'] + (servo_data['DIR'] * abs(max_limit - min_limit))

        # Change direction at limits
        if new_pos >= max_limit:
            new_pos = max_limit
            servo_data['DIR'] = -1  # Move backward next cycle
        elif new_pos <= min_limit:
            new_pos = min_limit
            servo_data['DIR'] = 1  # Move forward next cycle

        # Update CUR_POS in leg_config
        servo_data['CUR_POS'] = new_pos
        initialize_servos.LEG_CONFIG[leg]['upper']['CUR_POS'] = new_pos  # Ensure the original dictionary is updated

        # Send the updated position
        initialize_servos.setTarget(servo_data['servo'], servo_data['CUR_POS'], min_speed, min_acceleration)

        servo_data['MOVED'] = True

        # Log the movement
        logging.info(f"{leg} Upper Leg: Moved servo {servo_data['servo']} to {servo_data['CUR_POS']} with DIR={servo_data['DIR']}, Arc Length={min_arc_length}")

    ##### ensure all servos have moved before new oscillation cycle #####

    while not all(servo_data['MOVED'] for servo_data in upper_leg_servos.values()):

        time.sleep(0.05) # wait for servo to reach target