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
import math # import math library for pi, used with elliptical movement
import logging # import logging for debugging

##### import necessary functions #####

import initialize.initialize_servos as initialize_servos # import servo logic functions


########## CREATE DEPENDENCIES ##########

##### define servos #####

upper_leg_servos = { # define upper leg servos

    "FL": initialize_servos.LEG_CONFIG['FL']['upper'],  # front left
    "FR": initialize_servos.LEG_CONFIG['FR']['upper'],  # front right
    "BL": initialize_servos.LEG_CONFIG['BL']['upper'],  # back left
    "BR": initialize_servos.LEG_CONFIG['BR']['upper'],  # back right
}

lower_leg_servos = { # define lower leg servos

    "FL": initialize_servos.LEG_CONFIG['FL']['lower'],  # front left
    "FR": initialize_servos.LEG_CONFIG['FR']['lower'],  # front right
    "BL": initialize_servos.LEG_CONFIG['BL']['lower'],  # back left
    "BR": initialize_servos.LEG_CONFIG['BR']['lower'],  # back right
}





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


########## CORRECT LEG SYNC ##########

def correctLegSync(upper_leg_servos):

    # Get movement directions
    dir_FL = upper_leg_servos["FL"]["DIR"]
    dir_FR = upper_leg_servos["FR"]["DIR"]
    dir_BL = upper_leg_servos["BL"]["DIR"]
    dir_BR = upper_leg_servos["BR"]["DIR"]

    # Fix diagonal pairs (FL & BR) and (FR & BL)
    if dir_FL != dir_BR:
        logging.warning("FL & BR were out of sync. Correcting.")
        upper_leg_servos["BR"]["DIR"] = dir_FL  # Sync BR to FL

    if dir_FR != dir_BL:
        logging.warning("FR & BL were out of sync. Correcting.")
        upper_leg_servos["BL"]["DIR"] = dir_FR  # Sync BL to FR

    # Fix adjacent legs to ensure they are NOT in sync
    if dir_FL == dir_FR:  # If FL & FR are in sync, flip FR
        logging.warning("FL & FR were in sync. Correcting FR.")
        upper_leg_servos["FR"]["DIR"] *= -1

    if dir_BL == dir_BR:  # If BL & BR are in sync, flip BR
        logging.warning("BL & BR were in sync. Correcting BR.")
        upper_leg_servos["BR"]["DIR"] *= -1

    logging.info("Leg synchronization corrected.")


########## OSCILLATE LEGS ##########

def manualTrot(intensity): # function to oscillate one servo

    ##### set vairables #####

    diagonal_pairs = [("FL", "BR"), ("FR", "BL")]  # Trot pairings
    arc_lengths = []  # Store all arc lengths for uniform movement distance
    speeds = []  # Store all speeds for uniform movement speed
    accelerations = []  # Store all accelerations for uniform movement acceleration

    ##### Find movement parameters #####
    for leg, servo_data in upper_leg_servos.items():  # Loop through upper leg servos to get parameters with intensity
        full_back = servo_data['FULL_BACK']  # Get full back position
        full_front = servo_data['FULL_FRONT']  # Get full front position
        arc_length, speed, acceleration = interpretIntensity(intensity, full_back, full_front)  # Get movement parameters
        arc_lengths.append(arc_length)  # Append arc length to list
        speeds.append(speed)  # Append speed to list
        accelerations.append(acceleration)  # Append acceleration to list
        servo_data['MOVED'] = False

    min_arc_length = min(arc_lengths)  # Get minimum arc length
    min_speed = min(speeds)  # Get minimum speed
    min_acceleration = min(accelerations)  # Get minimum acceleration

    ##### move upper legs #####

    for pair in diagonal_pairs:

        move_commands = []  # Store commands to execute simultaneously

        for leg in pair:

            servo_data = upper_leg_servos[leg]
            full_back = servo_data['FULL_BACK']
            full_front = servo_data['FULL_FRONT']
            neutral_position = servo_data['NEUTRAL']

            if full_back < full_front:
                max_limit = neutral_position + (min_arc_length / 2)
                min_limit = neutral_position - (min_arc_length / 2)
            else:
                max_limit = neutral_position - (min_arc_length / 2)
                min_limit = neutral_position + (min_arc_length / 2)

            # Initialize movement direction
            if servo_data['DIR'] == 0:
                if leg in ["FL", "BR"]:
                    servo_data['DIR'] = 1  # Move forward
                else:
                    servo_data['DIR'] = -1  # Move backward

            # Compute new position
            new_pos = servo_data['CUR_POS'] + (servo_data['DIR'] * abs(max_limit - min_limit))

            # Change direction at limits
            if servo_data['DIR'] == 1:
                new_pos = max_limit
                servo_data['DIR'] = -1  # Move backward next cycle
            elif servo_data['DIR'] == -1:
                new_pos = min_limit
                servo_data['DIR'] = 1  # Move forward next cycle

            # Update servo data
            servo_data['CUR_POS'] = new_pos
            initialize_servos.LEG_CONFIG[leg]['upper']['CUR_POS'] = new_pos

            # Store movement command for simultaneous execution
            move_commands.append((servo_data['servo'], new_pos, min_speed, min_acceleration))
            servo_data['MOVED'] = True

            logging.info(
                f"{leg} Upper Leg: Moved servo {servo_data['servo']} to {new_pos} with DIR={servo_data['DIR']}, Arc Length={min_arc_length}")

        # Send all move commands for the pair at once
        for servo, pos, speed, acc in move_commands:
            initialize_servos.setTarget(servo, pos, speed, acc)