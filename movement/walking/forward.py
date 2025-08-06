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
        if config.USE_SIMULATION and config.USE_ISAAC_SIM:
            # Use queue-based movement for Isaac Sim to avoid PhysX threading violations
            from movement.fundamental_movement import ISAAC_MOVEMENT_QUEUE
            
            # Create movement data for the trot gait
            movement_data = {
                'current_coordinates': config.CURRENT_FEET_COORDINATES.copy(),
                'mid_coordinates': {},
                'target_coordinates': {},
                'movement_rates': {}
            }
            
            # Calculate target positions for each leg based on gait state
            for leg_id in ['FL', 'FR', 'BL', 'BR']:
                gait_state = gait_states[leg_id]
                phase = gait_state['phase']
                
                # Get current position
                current_pos = config.CURRENT_FEET_COORDINATES[leg_id]
                
                # Determine target position based on phase
                if phase == 'stance':
                    # Move from stance to swing positions
                    target_pos = config.FL_SWING if leg_id == 'FL' else \
                                config.FR_SWING if leg_id == 'FR' else \
                                config.BL_SWING if leg_id == 'BL' else \
                                config.BR_SWING
                    mid_pos = config.FL_TOUCHDOWN if leg_id == 'FL' else \
                             config.FR_TOUCHDOWN if leg_id == 'FR' else \
                             config.BL_TOUCHDOWN if leg_id == 'BL' else \
                             config.BR_TOUCHDOWN
                else:  # phase == 'swing'
                    # Move from swing to stance positions
                    target_pos = config.FL_STANCE if leg_id == 'FL' else \
                                config.FR_STANCE if leg_id == 'FR' else \
                                config.BL_STANCE if leg_id == 'BL' else \
                                config.BR_STANCE
                    mid_pos = config.FL_TIPPYTOES if leg_id == 'FL' else \
                             config.FR_TIPPYTOES if leg_id == 'FR' else \
                             config.BL_TIPPYTOES if leg_id == 'BL' else \
                             config.BR_TIPPYTOES
                
                movement_data['mid_coordinates'][leg_id] = mid_pos
                movement_data['target_coordinates'][leg_id] = target_pos
                movement_data['movement_rates'][leg_id] = {
                    'speed': speed,
                    'acceleration': acceleration
                }
                
                # Update gait state for next cycle
                gait_state['phase'] = 'swing' if phase == 'stance' else 'stance'
                gait_state['returned_to_neutral'] = False
            
            # Queue the movement data
            ISAAC_MOVEMENT_QUEUE.put(movement_data)
            logging.debug(f"(forward.py): Queued trot_forward movement data for Isaac Sim\n")
            
        else:
            # Threaded movement for real robot and PyBullet simulation
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
