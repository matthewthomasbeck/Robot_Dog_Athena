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
from kinematics.kinematics import Kinematics # import kinematics functions


########## CREATE DEPENDENCIES ##########

##### initialize kinematics #####

k = Kinematics(initialize_servos.LINK_CONFIG) # use link lengths to initialize kinematic functions

##### define servos #####

upper_leg_servos = { # define upper leg servos

    "FL": initialize_servos.SERVO_CONFIG['FL']['upper'],  # front left
    "FR": initialize_servos.SERVO_CONFIG['FR']['upper'],  # front right
    "BL": initialize_servos.SERVO_CONFIG['BL']['upper'],  # back left
    "BR": initialize_servos.SERVO_CONFIG['BR']['upper'],  # back right
}

lower_leg_servos = { # define lower leg servos

    "FL": initialize_servos.SERVO_CONFIG['FL']['lower'],  # front left
    "FR": initialize_servos.SERVO_CONFIG['FR']['lower'],  # front right
    "BL": initialize_servos.SERVO_CONFIG['BL']['lower'],  # back left
    "BR": initialize_servos.SERVO_CONFIG['BR']['lower'],  # back right
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


########## MANUAL TROT ##########


def moveFrontLeftLeg(x, y, z, speed=100, acceleration=10):
    """
    Moves the front left leg to a specified (x, y, z) foot position in meters.
    Uses inverse kinematics and maps angles to servo positions.
    """
    from kinematics.kinematics import Kinematics

    # Run inverse kinematics
    hip_angle, upper_angle, lower_angle = k.inverse_kinematics(x, y, z)

    # Define neutral angle assumptions (you can tune these)
    hip_neutral_angle = 0
    upper_neutral_angle = 90
    lower_neutral_angle = 90

    # Move each joint
    for joint, angle, neutral in zip(
        ['hip', 'upper', 'lower'],
        [hip_angle, upper_angle, lower_angle],
        [hip_neutral_angle, upper_neutral_angle, lower_neutral_angle]
    ):
        servo_data = initialize_servos.SERVO_CONFIG['FL'][joint]
        is_inverted = servo_data['FULL_BACK'] > servo_data['FULL_FRONT']
        pwm = initialize_servos.map_angle_to_servo_position(angle, servo_data, neutral, is_inverted)
        initialize_servos.setTarget(servo_data['servo'], pwm, speed, acceleration)



def manualTrot(intensity): # function to oscillate one servo

    ##### set vairables #####

    diagonal_pairs = [("FL", "BR"), ("FR", "BL")]  # Trot pairings
    upper_arc_lengths = []  # Store all arc lengths for uniform movement distance
    speeds = []  # Store all speeds for uniform movement speed
    accelerations = []  # Store all accelerations for uniform movement acceleration

    ##### Find movement parameters #####
    for leg, upper_servo_data in initialize_servos.SERVO_CONFIG.items(): # Loop through upper leg servos to get parameters with intensity
        upper_servo_data = initialize_servos.SERVO_CONFIG[leg]['upper']
        full_back = upper_servo_data['FULL_BACK']  # Get full back position
        full_front = upper_servo_data['FULL_FRONT']  # Get full front position
        arc_length, speed, acceleration = interpretIntensity(intensity, full_back, full_front)  # Get movement parameters
        upper_arc_lengths.append(arc_length)  # Append arc length to list
        speeds.append(speed)  # Append speed to list
        accelerations.append(acceleration)  # Append acceleration to list
        upper_servo_data['MOVED'] = False

    min_upper_arc_length = min(upper_arc_lengths)  # Get minimum arc length
    min_speed = min(speeds)  # Get minimum speed
    min_acceleration = min(accelerations)  # Get minimum acceleration

    ##### move upper legs #####

    for pair in diagonal_pairs:

        for leg in pair:

            upper_servo_data = upper_leg_servos[leg]
            full_back = upper_servo_data['FULL_BACK']
            full_front = upper_servo_data['FULL_FRONT']
            neutral_position = upper_servo_data['NEUTRAL']

            lower_servo_data = lower_leg_servos[leg]

            if full_back < full_front:
                max_limit = neutral_position + (min_upper_arc_length / 2)
                min_limit = neutral_position - (min_upper_arc_length / 2)
            else:
                min_upper_arc_length = (-1 * min_upper_arc_length)

                max_limit = neutral_position + (min_upper_arc_length / 2)
                min_limit = neutral_position - (min_upper_arc_length / 2)

            # Initialize movement direction
            if upper_servo_data['DIR'] == 0:
                if leg in ["FL", "BR"]:
                    upper_servo_data['DIR'] = 1  # Move forward
                else:
                    upper_servo_data['DIR'] = -1  # Move backward

            # Compute new position
            #upper_new_pos = upper_servo_data['CUR_POS'] + (upper_servo_data['DIR'] * abs(max_limit - min_limit))

            logging.info(f"Moving {leg}.\n")

            # Change direction at limits
            if upper_servo_data['DIR'] == 1:
                upper_new_pos = max_limit
                upper_servo_data['CUR_POS'] = upper_new_pos
                initialize_servos.SERVO_CONFIG[leg]['upper']['CUR_POS'] = upper_new_pos

                logging.debug("Lifting up...")

                liftLowerLeg(  # lift-up lower leg

                    lower_servo_data['servo'],
                    min_upper_arc_length,
                    16383,
                    255
                )

                time.sleep(0.1)

                logging.debug("Swinging Leg...")

                initialize_servos.setTarget(upper_servo_data['servo'], upper_new_pos, min_speed, min_acceleration)

                time.sleep(0.05)

                logging.debug("Stepping down...")

                neutralLowerLeg(  # touch down lower leg

                    lower_servo_data['servo'],
                    lower_servo_data['NEUTRAL'],
                    16383,
                    255
                )

                time.sleep(0.05)

                upper_servo_data['DIR'] = -1  # Move backward next cycle

                logging.info("Stepped forward.\n")

            elif upper_servo_data['DIR'] == -1:
                upper_new_pos = min_limit
                upper_servo_data['CUR_POS'] = upper_new_pos
                initialize_servos.SERVO_CONFIG[leg]['upper']['CUR_POS'] = upper_new_pos

                logging.debug("Planting foot...")

                lowerLowerLeg( # touch down lower leg

                    lower_servo_data['servo'],
                    min_upper_arc_length,
                    16383,
                    255
                )

                time.sleep(0.1)

                logging.debug("Pushing back...")

                initialize_servos.setTarget(upper_servo_data['servo'], upper_new_pos, min_speed, min_acceleration)

                time.sleep(0.05)

                logging.info("pushed backwards.\n")

                upper_servo_data['DIR'] = 1  # Move forward next cycle

            upper_servo_data['MOVED'] = True


# function to oscillate lower leg
def liftLowerLeg(servo_name, arc_length, speed, acceleration):

    initialize_servos.setTarget(servo_name, (-1 * arc_length), speed, acceleration)

def neutralLowerLeg(servo_name, arc_length, speed, acceleration):

    initialize_servos.setTarget(servo_name, arc_length, speed, acceleration)

def lowerLowerLeg(servo_name, arc_length, speed, acceleration):

    initialize_servos.setTarget(servo_name, (1 * arc_length), speed, acceleration)