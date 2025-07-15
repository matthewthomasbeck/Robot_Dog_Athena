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
import threading # import threading for concurrent movement

from movement.standing.standing import tippytoes_position
##### import necessary functions #####

from utilities.mathematics import interpret_intensity # import intensity interpretation function
import utilities.config as config # import leg positions config
from movement.fundamental_movement import move_foot_to_pos # import to move foot to pos





#################################################
############### WALKING FUNCTIONS ###############
#################################################


########## GAIT FUNCTIONS ##########

def trot_forward(intensity): # function to trot forward manually
    logging.debug(f"(forward.py): Running trot_forward() with intensity: {intensity}...\n")

    gait_states = {
        'FL': config.FL_GAIT_STATE, 'FR': config.FR_GAIT_STATE,
        'BL': config.BL_GAIT_STATE, 'BR': config.BR_GAIT_STATE
    }

    try:
        speed, acceleration = interpret_intensity(intensity)
    except Exception as e:
        logging.error(f"(forward.py): Failed to interpret intensity in trot_forward(): {e}\n")
        return

    try:
        # Threaded movement for all legs
        leg_threads = []
        for leg_id in ['FL', 'FR', 'BL', 'BR']:
            t = threading.Thread(
                target=manual_swing_leg,
                args=(leg_id, gait_states[leg_id]['phase'], speed, acceleration, gait_states[leg_id])
            )
            leg_threads.append(t)
            t.start()
        for t in leg_threads:
            t.join()
        time.sleep(0.1)
    except Exception as e:
        logging.error(f"(forward.py): Failed to gait-cycle legs in trot_forward(): {e}\n")


def manual_swing_leg(leg_id, phase, speed, acceleration, gait_state):
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

    try:
        if phase == 'stance':
            move_foot_to_pos(leg_id, stance_positions[leg_id], swing_positions[leg_id], speed, acceleration, use_bezier=False)
            move_foot_to_pos(leg_id, swing_positions[leg_id], touchdown_positions[leg_id], speed, acceleration, use_bezier=False)
            gait_state['phase'] = 'swing'
        elif phase == 'swing':
            move_foot_to_pos(leg_id, touchdown_positions[leg_id], tippytoes_positions[leg_id], speed, acceleration, use_bezier=False)
            move_foot_to_pos(leg_id, tippytoes_positions[leg_id], stance_positions[leg_id], speed, acceleration, use_bezier=False)
            gait_state['phase'] = 'stance'
        gait_state['returned_to_neutral'] = False
    except Exception as e:
        logging.error(f"(forward.py): Failed gait move for {leg_id} in phase {phase}: {e}\n")
