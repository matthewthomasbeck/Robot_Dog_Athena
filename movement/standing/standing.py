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
from movement.fundamental_movement import move_foot_to_pos # import fundamental movement function to move foot to pos





##########################################################
############### STANDING INPLACE MOVEMENTS ###############
##########################################################


########## NEUTRAL POSITION ##########

def neutral_position(intensity): # function to set all legs to squatting position

    ##### get intensity #####

    try: # try to calculate intensity of the movement

        speed, acceleration, _ = interpret_intensity(intensity) # TODO experiment with timing difference

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

            move_foot_to_pos(
                leg_id,
                neutral_position,
                speed,
                acceleration,
                stride_scalar=1,
                use_bezier=False
            )

            gait_state['returned_to_neutral'] = True

        except Exception as e: # if movement fails...

            logging.error(f"(manual_walking.py): Failed to reset {leg_id} leg neutral position: {e}\n")
            return


########## TIPPYTOES POSITION ##########

def tippytoes_position(intensity): # function to set all legs to tippytoes position

    ##### get intensity #####

    try: # try to calculate intensity of the movement

        speed, acceleration, _ = interpret_intensity(intensity) # TODO experiment with timing difference

    except Exception as e: # if interpretation fails...

        logging.error(f"(standing.py): Failed to interpret intensity in tippytoes_position(): {e}\n")
        return

    ##### move legs forward #####

    try: # try to update leg tippytoes

        set_leg_tippytoes('FL', {'FORWARD': True}, speed, acceleration)
        set_leg_tippytoes('BR', {'FORWARD': True}, speed, acceleration)
        set_leg_tippytoes('FR', {'FORWARD': True}, speed, acceleration)
        set_leg_tippytoes('BL', {'FORWARD': True}, speed, acceleration)

    except Exception as e: # if gait update fails...

        logging.error(f"(standing.py): Failed to move legs to tippytoes position: {e}\n")


########## SET LEG TIPPYTOES ##########

def set_leg_tippytoes(leg_id, state, speed, acceleration):

    ##### gait pre-check #####

    if not state.get('FORWARD', False): # if leg is moving forward...
        return  # skip if not moving forward

    ##### set variables #####

    tippytoes_positions = {
        'FL': config.FL_TIPPYTOES, 'FR': config.FR_TIPPYTOES, 'BL': config.BL_TIPPYTOES, 'BR': config.BR_TIPPYTOES
    }
    gait_states = {
        'FL': config.FL_GAIT_STATE, 'FR': config.FR_GAIT_STATE, 'BL': config.BL_GAIT_STATE, 'BR': config.BR_GAIT_STATE
    }
    gait_state = gait_states[leg_id]
    current_phase = gait_state.get('phase')
    tippytoes_pos = tippytoes_positions[leg_id]

    ##### check if already at tippytoes #####

    if gait_state.get('last_position') == 'tippytoes':
        return  # skip if already at tippytoes

    ##### move leg to tippytoes #####

    try: # try to move leg to tippytoes position
        move_foot_to_pos(leg_id, tippytoes_pos, speed, acceleration, stride_scalar=1, use_bezier=False)
        gait_state['last_position'] = 'tippytoes'
        gait_state['phase'] = 'stance'  # tippytoes is a grounded stance
        gait_state['returned_to_neutral'] = False

    except Exception as e: # if movement fails...
        logging.error(f"(standing.py): Failed to move leg {leg_id} to tippytoes position: {e}\n")
        return


########## SQUATTING POSITION ##########

def squatting_position(intensity): # function to set all legs to squatting position

    ##### get intensity #####

    try: # try to calculate intensity of the movement

        speed, acceleration, _ = interpret_intensity(intensity) # TODO experiment with timing difference

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

def set_leg_squatting(leg_id, state, speed, acceleration):

    ##### gait pre-check #####

    if not state.get('FORWARD', False): # if leg is moving forward...
        return  # skip if not moving forward

    ##### set variables #####

    squatting_positions = {
        'FL': config.FL_SQUATTING, 'FR': config.FR_SQUATTING, 'BL': config.BL_SQUATTING, 'BR': config.BR_SQUATTING
    }
    gait_states = {
        'FL': config.FL_GAIT_STATE, 'FR': config.FR_GAIT_STATE, 'BL': config.BL_GAIT_STATE, 'BR': config.BR_GAIT_STATE
    }
    gait_state = gait_states[leg_id]
    current_phase = gait_state.get('phase')
    squatting_pos = squatting_positions[leg_id]

    ##### check if already squatting #####

    if gait_state.get('last_position') == 'squatting':
        return  # skip if already squatting

    ##### move leg to squatting #####

    try: # try to move leg to squatting position
        move_foot_to_pos(leg_id, squatting_pos, speed, acceleration, stride_scalar=1, use_bezier=False)
        gait_state['last_position'] = 'squatting'
        gait_state['phase'] = 'stance'  # squatting is a grounded stance
        gait_state['returned_to_neutral'] = False

    except Exception as e: # if movement fails...
        logging.error(f"(standing.py): Failed to move leg {leg_id} to squatting position: {e}\n")
        return