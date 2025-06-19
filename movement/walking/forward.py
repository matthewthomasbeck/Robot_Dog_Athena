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

from utilities.mathematics import interpret_intensity # import intensity interpretation function
import utilities.config as config # import leg positions config
from movement.fundamental_movement import move_foot_to_pos # import fundamental movement function to move foot to pos





#################################################
############### WALKING FUNCTIONS ###############
#################################################


########## GAIT FUNCTIONS ##########

def trot_forward(intensity): # function to trot forward

    ##### get intensity #####

    try: # try to calculate intensity of the trot

        speed, acceleration, stride_scalar = interpret_intensity(intensity) # TODO experiment with timing difference

    except Exception as e: # if interpretation fails...

        logging.error(f"(forward.py): Failed to interpret intensity in trot_forward(): {e}\n")
        return

    ##### move legs forward #####

    try: # try to update leg gait

        update_leg_gait('FL', {'FORWARD': True}, speed, acceleration, stride_scalar)
        time.sleep(.06)
        update_leg_gait('BR', {'FORWARD': True}, speed, acceleration, stride_scalar)
        time.sleep(.06)
        update_leg_gait('FR', {'FORWARD': True}, speed, acceleration, stride_scalar)
        time.sleep(.06)
        update_leg_gait('BL', {'FORWARD': True}, speed, acceleration, stride_scalar)
        time.sleep(.06)

    except Exception as e: # if gait update fails...

        logging.error(f"(forward.py): Failed to gait-cycle legs in trot_forward(): {e}\n")


########## UPDATE LEG GAITS ##########

def update_leg_gait(leg_id, state, speed, acceleration, stride_scalar):

    ##### gait pre-check #####

    if not state.get('FORWARD', False): # if leg is moving forward...
        return  # skip if not moving forward

    ##### set variables #####

    gait_states = {
        'FL': config.FL_GAIT_STATE, 'FR': config.FR_GAIT_STATE, 'BL': config.BL_GAIT_STATE, 'BR': config.BR_GAIT_STATE
    }
    swing_positions = {
        'FL': config.FL_SWING, 'FR': config.FR_SWING, 'BL': config.BL_SWING, 'BR': config.BR_SWING
    }
    stance_positions = {
        'FL': config.FL_STANCE, 'FR': config.FR_STANCE, 'BL': config.BL_STANCE, 'BR': config.BR_STANCE
    }
    gait_state = gait_states[leg_id]

    ##### update leg gait #####

    if gait_state['phase'] == 'stance': # if leg is in stance phase...

        try: # try to move leg to swing position
            move_foot_to_pos(leg_id, swing_positions[leg_id], speed, acceleration, stride_scalar, use_bezier=False)
            gait_state['phase'] = 'swing'

        except Exception as e: # if movement fails...
            logging.error(f"(forward.py): Failed to move leg {leg_id} to swing position: {e}\n")
            return

    else: # if leg is in swing phase...

        try: # try to move leg to stance position
            move_foot_to_pos(leg_id, stance_positions[leg_id], speed, acceleration, stride_scalar, use_bezier=False) #TODO enable bezier once fixed
            gait_state['phase'] = 'stance'

        except Exception as e: # if movement fails...
            logging.error(f"(forward.py): Failed to move leg {leg_id} to stance position: {e}\n")
            return

    gait_state['returned_to_neutral'] = False # reset neutral position flag