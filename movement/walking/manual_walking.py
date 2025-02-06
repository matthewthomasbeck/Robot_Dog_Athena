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

    speed = int((intensity / 10) * 16383) # map intensity (1-10) to valid speed range (0-16383)

    ##### find acceleration #####

    acceleration = int((intensity / 10) * 255) # map intensity (1-10) to valid acceleration range (0-255)

    ##### return arc length and speed #####

    return arc_length, speed, acceleration # return movement parameters


########## MOVE LEG ##########

def moveLeg(leg, action): # function to move a leg to a desired position

    ##### check if leg is valid #####

    if leg not in initialize_servos.LEG_CONFIG: # if leg is not in leg configuration...

        logging.error(f"ERROR (manual_walking.py): Invalid leg '{leg}' found.\n") # print error message

        return # return to stop the function

    ##### move leg to desired position #####

    try: # attempt to move leg to desired position

        for joint, servo_info in initialize_servos.LEG_CONFIG[leg].items():

            ##### set variables #####

            servo = servo_info['servo'] # get servo pin
            full_back = servo_info['FULL_BACK'] # get full back position
            full_front = servo_info['FULL_FRONT'] # get full front position

            ##### move leg to desired position #####

            if action == 'LIFT': # if action is to lift leg...

                initialize_servos.setTarget(servo, full_back) # move leg to full back position

            elif action == 'FORWARD': # if action is to move leg forward...

                initialize_servos.setTarget(servo, full_front) # move leg to full front position

            elif action == 'DOWN': # if action is to move leg down...

                mid_position = (full_front + full_back) / 2

                initialize_servos.setTarget(servo, mid_position) # move leg to center position

    except Exception as e: # if some error occurs...

        logging.error(f"ERROR (manual_walking.py): Failed to move {leg} in moveLeg: {e}\n")


########## OSCILLATE ONE SERVO ##########

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
        initialize_servos.setTarget(servo_data['servo'], servo_data['CUR_POS'])

        # Log the movement
        logging.info(f"{leg} Upper Leg: Moved servo {servo_data['servo']} to {servo_data['CUR_POS']} with DIR={servo_data['DIR']}, Arc Length={arc_length}")


########## MANUAL FORWARD ##########

def manualForward(): # function to move forward manually

    ##### set variables #####

    gait_sequence = [ # define gait sequence for forward movement

        # step 1: lift front-left (FL) and back-right (BR)
        {'FL': 'LIFT', 'BR': 'LIFT'},
        {'FL': 'FORWARD', 'BR': 'FORWARD'},
        {'FL': 'DOWN', 'BR': 'DOWN'},

        # step 2: lift front-right (FR) and back-left (BL)
        {'FR': 'LIFT', 'BL': 'LIFT'},
        {'FR': 'FORWARD', 'BL': 'FORWARD'},
        {'FR': 'DOWN', 'BL': 'DOWN'},
    ]

    ##### move forward #####

    try: # attempt to move forward

        for step in gait_sequence: # iterate through each step in the gait sequence

            for leg, action in step.items(): # iterate through each leg in the step

                moveLeg(leg, action) # move leg to desired location

            time.sleep(0.2) # wait for smoother motion

    except Exception as e: # if some error occurs...

        logging.error(f"ERROR (manual_walking.py): Failed to move forward in manualForward: {e}\n")