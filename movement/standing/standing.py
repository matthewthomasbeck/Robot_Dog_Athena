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
import logging # import logging for debugging

##### import necessary functions #####

import utilities.config as config # import leg positions config
from utilities.mathematics import interpret_intensity # import intensity interpretation function
from movement.fundamental_movement import move_foot_to_pos_OLD # import fundamental movement function to move foot to pos





##########################################################
############### STANDING INPLACE MOVEMENTS ###############
##########################################################


########## NEUTRAL POSITION ##########

def neutral_position(intensity): # function to set all legs to squatting position

    ##### get intensity #####

    try: # try to calculate intensity of the movement

        speed, acceleration = interpret_intensity(intensity)

    except Exception as e: # if interpretation fails...

        logging.error(f"(standing.py): Failed to interpret intensity in neutral_position(): {e}\n")
        return

    ##### move legs forward #####

    try: # try to update legs to neutral position

        set_leg_neutral('FL', {'FORWARD': True}, speed, acceleration)
        set_leg_neutral('BR', {'FORWARD': True}, speed, acceleration)
        set_leg_neutral('FR', {'FORWARD': True}, speed, acceleration)
        set_leg_neutral('BL', {'FORWARD': True}, speed, acceleration)

    except Exception as e: # if gait update fails...

        logging.error(f"(standing.py): Failed to move legs to neutral position: {e}\n")


########## SET LEG NEUTRAL ##########

def set_leg_neutral(leg_id, state, speed, acceleration): # function to return leg to neutral

    ##### set variables #####

    gait_states = {
        'FL': config.FL_GAIT_STATE, 'FR': config.FR_GAIT_STATE, 'BL': config.BL_GAIT_STATE, 'BR': config.BR_GAIT_STATE
    }
    neutral_positions = {
        'FL': config.FL_NEUTRAL, 'FR': config.FR_NEUTRAL, 'BL': config.BL_NEUTRAL, 'BR': config.BR_NEUTRAL
    }
    gait_state = gait_states[leg_id]
    neutral_position = neutral_positions[leg_id]

    ##### return legs to neutral #####

    if not gait_state['returned_to_neutral']: # if leg has not returned to neutral position...

        try: # try to move leg to neutral position

            move_foot_to_pos_OLD(
                leg_id,
                neutral_position,
                speed,
                acceleration,
                use_bezier=False
            )

            gait_state['returned_to_neutral'] = True

        except Exception as e: # if movement fails...

            logging.error(f"(standing.py): Failed to reset {leg_id} leg neutral position: {e}\n")
            return


########## TIPPYTOES POSITION ##########

def tippytoes_position(intensity): # function to set all legs to tippytoes position TODO TUNE ME!

    ##### get intensity #####

    try: # try to calculate intensity of the movement

        speed, acceleration = interpret_intensity(intensity)

    except Exception as e: # if interpretation fails...

        logging.error(f"(standing.py): Failed to interpret intensity in tippytoes_position(): {e}\n")
        return

    ##### move legs forward #####

    try: # try to update leg tippytoes

        set_leg_tippytoes('BR', {'FORWARD': True}, speed, acceleration)
        set_leg_tippytoes('BL', {'FORWARD': True}, speed, acceleration)
        time.sleep(0.1) # TODO figure this out once I get the robot walking
        set_leg_tippytoes('FL', {'FORWARD': True}, speed, acceleration)
        set_leg_tippytoes('FR', {'FORWARD': True}, speed, acceleration)
        #time.sleep(0.5)
        #set_leg_tippytoes('BR', {'FORWARD': True}, speed, acceleration)
        #set_leg_tippytoes('BL', {'FORWARD': True}, speed, acceleration)

    except Exception as e: # if gait update fails...

        logging.error(f"(standing.py): Failed to move legs to tippytoes position: {e}\n")


########## SET LEG TIPPYTOES ##########

def set_leg_tippytoes(leg_id, state, speed, acceleration): # function to move leg to tippytoes

    ##### set variables #####

    gait_states = {
        'FL': config.FL_GAIT_STATE, 'FR': config.FR_GAIT_STATE, 'BL': config.BL_GAIT_STATE, 'BR': config.BR_GAIT_STATE
    }
    tippytoes_positions = {
        'FL': config.FL_TIPPYTOES, 'FR': config.FR_TIPPYTOES, 'BL': config.BL_TIPPYTOES, 'BR': config.BR_TIPPYTOES
    }
    gait_state = gait_states[leg_id]
    tippytoes_position = tippytoes_positions[leg_id]

    ##### move leg to tippytoes #####

    if gait_state.get('last_position') != 'tippytoes': # if leg is not already at tippytoes...

        try: # try to move leg to tippytoes position

            move_foot_to_pos_OLD(
                leg_id,
                tippytoes_position,
                speed,
                acceleration,
                use_bezier=False
            )

            gait_state['last_position'] = 'tippytoes'
            gait_state['returned_to_neutral'] = False

        except Exception as e: # if movement fails...

            logging.error(f"(standing.py): Failed to move leg {leg_id} to tippytoes position: {e}\n")
            return


########## SQUATTING POSITION ##########

def squatting_position(intensity): # function to set all legs to squatting position

    ##### get intensity #####

    try: # try to calculate intensity of the movement

        speed, acceleration = interpret_intensity(intensity)

    except Exception as e: # if interpretation fails...

        logging.error(f"(standing.py): Failed to interpret intensity in squatting_position(): {e}\n")
        return

    ##### move legs forward #####

    try: # try to update legs to tippytoes

        set_leg_squatting('FL', {'FORWARD': True}, speed, acceleration)
        set_leg_squatting('BR', {'FORWARD': True}, speed, acceleration)
        set_leg_squatting('FR', {'FORWARD': True}, speed, acceleration)
        set_leg_squatting('BL', {'FORWARD': True}, speed, acceleration)

    except Exception as e: # if gait update fails...

        logging.error(f"(standing.py): Failed to move legs to squatting position: {e}\n")


########## SET LEG SQUATTING ##########

def set_leg_squatting(leg_id, state, speed, acceleration): # function to move leg to squatting position

    ##### set variables #####

    gait_states = {
        'FL': config.FL_GAIT_STATE, 'FR': config.FR_GAIT_STATE, 'BL': config.BL_GAIT_STATE, 'BR': config.BR_GAIT_STATE
    }
    squatting_positions = {
        'FL': config.FL_SQUATTING, 'FR': config.FR_SQUATTING, 'BL': config.BL_SQUATTING, 'BR': config.BR_SQUATTING
    }
    gait_state = gait_states[leg_id]
    squatting_position = squatting_positions[leg_id]

    ##### move leg to squatting #####

    if gait_state.get('last_position') != 'squatting': # if leg is not already squatting...

        try: # try to move leg to squatting position

            move_foot_to_pos_OLD(
                leg_id,
                squatting_position,
                speed,
                acceleration,
                use_bezier=False
            )

            gait_state['last_position'] = 'squatting'
            gait_state['returned_to_neutral'] = False

        except Exception as e: # if movement fails...

            logging.error(f"(standing.py): Failed to move leg {leg_id} to squatting position: {e}\n")
            return


def move_all_joints_full_front(intensity):
    """
    Move all joints of all legs to their FULL_FRONT positions for calibration.
    """
    import utilities.config as config
    from utilities.servos import set_target
    speed, acceleration = interpret_intensity(intensity)
    for leg in config.SERVO_CONFIG:
        for joint in config.SERVO_CONFIG[leg]:
            servo_data = config.SERVO_CONFIG[leg][joint]
            set_target(servo_data['servo'], servo_data['FULL_FRONT'], speed, acceleration)
    logging.info("(standing.py): All joints moved to FULL_FRONT for calibration.\n")


def move_all_joints_full_back(intensity):
    """
    Move all joints of all legs to their FULL_BACK positions for calibration.
    """
    import utilities.config as config
    from utilities.servos import set_target
    speed, acceleration = interpret_intensity(intensity)
    for leg in config.SERVO_CONFIG:
        for joint in config.SERVO_CONFIG[leg]:
            servo_data = config.SERVO_CONFIG[leg][joint]
            set_target(servo_data['servo'], servo_data['FULL_BACK'], speed, acceleration)
    logging.info("(standing.py): All joints moved to FULL_BACK for calibration.\n")