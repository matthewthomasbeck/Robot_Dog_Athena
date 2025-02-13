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

def manualTrot(intensity): # function to oscillate one servo

    ##### set vairables #####

    upper_leg_servos = { # define upper leg servos

        "FL": initialize_servos.LEG_CONFIG['FL']['upper'], # front left
        "FR": initialize_servos.LEG_CONFIG['FR']['upper'], # front right
        "BL": initialize_servos.LEG_CONFIG['BL']['upper'], # back left
        "BR": initialize_servos.LEG_CONFIG['BR']['upper'], # back right
    }

    lower_leg_servos = { # define lower leg servos

        "FL": initialize_servos.LEG_CONFIG['FL']['lower'], # front left
        "FR": initialize_servos.LEG_CONFIG['FR']['lower'], # front right
        "BL": initialize_servos.LEG_CONFIG['BL']['lower'], # back left
        "BR": initialize_servos.LEG_CONFIG['BR']['lower'], # back right
    }

    arc_lengths = [] # store all arc lengths for uniform movement distance
    speeds = [] # store all speeds for uniform movement speed
    accelerations = [] # store all accelerations for uniform movement acceleration

    ##### find movement parameters #####

    for leg, servo_data in upper_leg_servos.items(): # loop through upper leg servos to get parameters with intensity

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

    ##### oscillate upper legs #####

    diagonal_pairs = [("FL", "BR"), ("FR", "BL")]  # Trot pairings

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
            if new_pos >= max_limit:
                new_pos = max_limit
                servo_data['DIR'] = -1  # Move backward next cycle
            elif new_pos <= min_limit:
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

    ##### ensure all servos have moved before new oscillation cycle #####

    #while not all(servo_data['MOVED'] for servo_data in upper_leg_servos.values()): # while not all servos have moved...

        #time.sleep(0.15) # wait for servo to reach target
