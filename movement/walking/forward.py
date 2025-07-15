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

from movement.standing.standing import tippytoes_position
##### import necessary functions #####

from utilities.mathematics import interpret_intensity # import intensity interpretation function
import utilities.config as config # import leg positions config
from movement.fundamental_movement import move_foot_to_pos_OLD # import to move foot to pos





#################################################
############### WALKING FUNCTIONS ###############
#################################################


########## GAIT FUNCTIONS ##########

def trot_forward(intensity): # function to trot forward manually

    logging.debug(f"(forward.py): Running trot_forward() with intensity: {intensity}...\n")

    ##### set variables #####

    gait_states = {
        'FL': config.FL_GAIT_STATE, 'FR': config.FR_GAIT_STATE,
        'BL': config.BL_GAIT_STATE, 'BR': config.BR_GAIT_STATE
    }

    ##### get intensity #####

    try: # try to calculate intensity of the trot
        speed, acceleration = interpret_intensity(intensity)
    except Exception as e: # if interpretation fails...
        logging.error(f"(forward.py): Failed to interpret intensity in trot_forward(): {e}\n")
        return

    ##### move legs forward #####

    try: # try to update leg gait

        if gait_states['FL']['phase'] == 'swing': # if front left about to push...
            set_leg_phase('BL', {'FORWARD': True}, speed, acceleration)
            set_leg_phase('FR', {'FORWARD': True}, speed, acceleration)
            time.sleep(0.5)
            set_leg_phase('BR', {'FORWARD': True}, speed, acceleration)
            set_leg_phase('FL', {'FORWARD': True}, speed, acceleration)
            #set_leg_phase('FL', {'FORWARD': True}, speed, acceleration)
            #set_leg_phase('BR', {'FORWARD': True}, speed, acceleration)
            #set_leg_phase('FR', {'FORWARD': True}, speed, acceleration)
            #set_leg_phase('BL', {'FORWARD': True}, speed, acceleration)

        elif gait_states['FR']['phase'] == 'swing': # if front right about to push...
            set_leg_phase('BR', {'FORWARD': True}, speed, acceleration)
            set_leg_phase('FL', {'FORWARD': True}, speed, acceleration)
            time.sleep(0.5)
            set_leg_phase('BL', {'FORWARD': True}, speed, acceleration)
            set_leg_phase('FR', {'FORWARD': True}, speed, acceleration)
            #set_leg_phase('FR', {'FORWARD': True}, speed, acceleration)
            #set_leg_phase('BL', {'FORWARD': True}, speed, acceleration)
            #set_leg_phase('FL', {'FORWARD': True}, speed, acceleration)
            #set_leg_phase('BR', {'FORWARD': True}, speed, acceleration)
            logging.info(f"(forward.py): Executed trot_forward() with intensity: {intensity}\n")

        time.sleep(0.1) # wait for legs to reach positions

    except Exception as e: # if gait update fails...
        logging.error(f"(forward.py): Failed to gait-cycle legs in trot_forward(): {e}\n")


########## SET LEG PHASE ##########

def set_leg_phase(leg_id, state, speed, acceleration): # function to set leg phase manually

    ##### pre-check if moving forward #####

    if not state.get('FORWARD', False): # if not moving forward...
        return # ignore this function call

    ##### set variables #####
    gait_states = {
        'FL': config.FL_GAIT_STATE, 'FR': config.FR_GAIT_STATE,
        'BL': config.BL_GAIT_STATE, 'BR': config.BR_GAIT_STATE
    }
    swing_positions = {
        'FL': config.FL_SWING, 'FR': config.FR_SWING,
        'BL': config.BL_SWING, 'BR': config.BR_SWING
    }
    touchdown_positions = {
        'FL': config.FL_TOUCHDOWN, 'FR': config.FR_TOUCHDOWN,
        'BL': config.BL_TOUCHDOWN, 'BR': config.BR_TOUCHDOWN
    }
    midstance_positions = {
        'FL': config.FL_MIDSTANCE, 'FR': config.FR_MIDSTANCE,
        'BL': config.BL_MIDSTANCE, 'BR': config.BR_MIDSTANCE
    }
    neutral_positions = {
        'FL': config.FL_NEUTRAL, 'FR': config.FR_NEUTRAL,
        'BL': config.BL_NEUTRAL, 'BR': config.BR_NEUTRAL
    }
    tippytoes_positions = {
        'FL': config.FL_TIPPYTOES, 'FR': config.FR_TIPPYTOES,
        'BL': config.BL_TIPPYTOES, 'BR': config.BR_TIPPYTOES
    }
    stance_positions = {
        'FL': config.FL_STANCE, 'FR': config.FR_STANCE,
        'BL': config.BL_STANCE, 'BR': config.BR_STANCE
    }
    gait_state = gait_states[leg_id]
    phase = gait_state['phase']

    ##### move feet #####

    try: # try to move feet
        if phase == 'stance': # lift foot into swing
            move_foot_to_pos_OLD(leg_id, swing_positions[leg_id], speed, acceleration, use_bezier=False)
            time.sleep(0.2)
            move_foot_to_pos_OLD(leg_id, touchdown_positions[leg_id], speed, acceleration, use_bezier=False)
            time.sleep(0.2)
            gait_state['phase'] = 'swing'
        elif phase == 'swing': # move foot into stance
            move_foot_to_pos_OLD(leg_id, tippytoes_positions[leg_id], speed, acceleration, use_bezier=False)
            time.sleep(0.2)
            move_foot_to_pos_OLD(leg_id, stance_positions[leg_id], speed, acceleration, use_bezier=False)
            time.sleep(0.2)
            gait_state['phase'] = 'stance'
        gait_state['returned_to_neutral'] = False # reset neutral state

    except Exception as e:
        logging.error(f"(forward.py): Failed gait move for {leg_id} in phase {phase}: {e}\n")
