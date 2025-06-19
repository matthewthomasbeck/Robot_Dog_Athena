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

from mathematics.mathematics import interpret_intensity # import intensity interpretation function
from movement.positions_config import * # import leg positions config
import movement.fundamental_movement as fundamental_movement # import fundamental movement function to move legs to positions





#################################################
############### WALKING FUNCTIONS ###############
#################################################


########## LEG PHASE CONFIG ##########

##### gait states #####

fl_gait_state = {'phase': 'stance', 'last_time': time.time(), 'returned_to_neutral': False}
br_gait_state = {'phase': 'stance', 'last_time': time.time(), 'returned_to_neutral': False}
fr_gait_state = {'phase': 'swing', 'last_time': time.time(), 'returned_to_neutral': False}
bl_gait_state = {'phase': 'swing', 'last_time': time.time(), 'returned_to_neutral': False}


########## GAIT FUNCTIONS ##########

def trotForward(intensity): # function to trot forward

    ##### get intensity #####

    try: # try to calculate intensity of the trot

        speed, acceleration, stride_scalar = interpret_intensity(intensity) # TODO experiment with timing difference

    except Exception as e: # if interpretation fails...

        logging.error(f"(manual_walking.py): Failed to interpret intensity: {e}\n")
        return

    ##### move legs forward #####

    try: # try to update leg gait

        updateLegGait('FL', {'FORWARD': True}, speed, acceleration, stride_scalar)
        time.sleep(.15)
        updateLegGait('BR', {'FORWARD': True}, speed, acceleration, stride_scalar)
        time.sleep(.15)
        updateLegGait('FR', {'FORWARD': True}, speed, acceleration, stride_scalar)
        time.sleep(.15)
        updateLegGait('BL', {'FORWARD': True}, speed, acceleration, stride_scalar)
        time.sleep(.15)

    except Exception as e: # if gait update fails...

        logging.error(f"(manual_walking.py): Failed to update leg gait: {e}\n")


########## UPDATE LEG GAITS ##########

def updateLegGait(leg_id, state, speed, acceleration, stride_scalar):

    ##### gait pre-check #####

    if not state.get('FORWARD', False): # if leg is moving forward...
        return  # skip if not moving forward

    ##### set variables #####

    gait_states = {'FL': fl_gait_state, 'FR': fr_gait_state, 'BL': bl_gait_state, 'BR': br_gait_state}
    swing_positions = {'FL': FL_SWING, 'FR': FR_SWING, 'BL': BL_SWING, 'BR': BR_SWING}
    stance_positions = {'FL': FL_STANCE, 'FR': FR_STANCE, 'BL': BL_STANCE, 'BR': BR_STANCE}
    gait_state = gait_states[leg_id]

    ##### update leg gait #####

    if gait_state['phase'] == 'stance': # if leg is in stance phase...

        try: # try to move leg to swing position
            fundamental_movement.move_leg_to_pos(leg_id, swing_positions[leg_id], speed, acceleration, stride_scalar, use_bezier=False)
            gait_state['phase'] = 'swing'

        except Exception as e: # if movement fails...
            logging.error(f"(manual_walking.py): Failed to move leg {leg_id} to swing position: {e}\n")
            return

    else: # if leg is in swing phase...

        try: # try to move leg to stance position
            fundamental_movement.move_leg_to_pos(leg_id, stance_positions[leg_id], speed, acceleration, stride_scalar, use_bezier=False) #TODO enable bezier once fixed
            gait_state['phase'] = 'stance'

        except Exception as e: # if movement fails...
            logging.error(f"(manual_walking.py): Failed to move leg {leg_id} to stance position: {e}\n")
            return

    gait_state['returned_to_neutral'] = False # reset neutral position flag


########## RESET LEG FORWARD GAIT ##########

def resetLegForwardGait(leg_id):

    ##### set variables #####

    gait_states = {'FL': fl_gait_state, 'FR': fr_gait_state, 'BL': bl_gait_state, 'BR': br_gait_state}
    neutral_positions = {'FL': FL_NEUTRAL, 'FR': FR_NEUTRAL, 'BL': BL_NEUTRAL, 'BR': BR_NEUTRAL}
    gait_state = gait_states[leg_id]
    neutral_position = neutral_positions[leg_id]

    ##### return legs to neutral #####

    if not gait_state['returned_to_neutral']: # if leg has not returned to neutral position...

        try: # try to move leg to neutral position

            fundamental_movement.move_leg_to_pos(
                leg_id,
                pos=neutral_position,
                speed=16383,
                acceleration=255,
                stride_scalar=1,
                use_bezier=False
            )

            gait_state['returned_to_neutral'] = True

        except Exception as e: # if movement fails...

            logging.error(f"(manual_walking.py): Failed to reset {leg_id} leg forward gait: {e}\n")
            return