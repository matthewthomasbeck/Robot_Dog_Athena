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

    ##### find speed #####

    speed = int(((16383 / 5) / 10) * intensity) # map intensity (1-10) to valid speed range (0-16383)

    ##### find acceleration #####

    acceleration = int(((255 / 4) / 10) * intensity) # map intensity (1-10) to valid acceleration range (0-255)

    ##### return arc length and speed #####

    return arc_length, speed, acceleration # return movement parameters


########## OSCILLATE LEGS ##########

def oscillateLegs(intensity): # function to oscillate one servo

    # Define upper leg servos
    upper_leg_servos = {
        "FL": initialize_servos.LEG_CONFIG['FL']['upper'],  # Front Left
        "FR": initialize_servos.LEG_CONFIG['FR']['upper'],  # Front Right
        "BL": initialize_servos.LEG_CONFIG['BL']['upper'],  # Back Left
        "BR": initialize_servos.LEG_CONFIG['BR']['upper'],  # Back Right
    }

    for leg, servo_data in upper_leg_servos.items():

        full_back = servo_data['FULL_BACK']
        full_front = servo_data['FULL_FRONT']
        neutral_position = servo_data['NEUTRAL']
        arc_length, speed, acceleration = interpretIntensity(intensity, full_back, full_front)

        if full_back < full_front: # if back position greater number...

            max_limit = neutral_position + (arc_length / 2)
            min_limit = neutral_position - (arc_length / 2)

        else: # if front position greater number...

            max_limit = neutral_position - (arc_length / 2)
            min_limit = neutral_position + (arc_length / 2)

        #max_limit = servo_data['FULL_FRONT']
        #min_limit = servo_data['FULL_BACK']

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
        initialize_servos.setTarget(servo_data['servo'], servo_data['CUR_POS'], speed, acceleration)

        # Log the movement
        logging.info(f"{leg} Upper Leg: Moved servo {servo_data['servo']} to {servo_data['CUR_POS']} with DIR={servo_data['DIR']}, Arc Length={arc_length}")
