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
from movement.fundamental_movement import move_foot_to_pos, _queue_single_joint_position_isaac # import fundamental movement functions





##########################################################
############### STANDING INPLACE MOVEMENTS ###############
##########################################################


########## NEUTRAL POSITION ##########

def neutral_position(intensity): # function to set all legs to neutral position
    """
    Set all legs to neutral position with two modes:
    - Physical robot: Uses move_foot_to_pos with 3D coordinates
    - Isaac Sim: Uses direct joint control with NEUTRAL_ANGLE
    """
    
    ##### get intensity #####
    try:
        speed, acceleration = interpret_intensity(intensity)
    except Exception as e:
        logging.error(f"(standing.py): Failed to interpret intensity in neutral_position(): {e}\n")
        return

    ##### move legs to neutral based on simulation mode #####
    try:
        if config.USE_SIMULATION and config.USE_ISAAC_SIM:
            # Isaac Sim mode: Direct joint control using NEUTRAL_ANGLE
            _neutral_position_isaac()
        else:
            # Physical robot mode: Use 3D coordinates
            _neutral_position_physical(speed, acceleration)
            
    except Exception as e:
        logging.error(f"(standing.py): Failed to move legs to neutral position: {e}\n")


def _neutral_position_isaac():
    """
    Set all joints to neutral position for Isaac Sim using direct joint control.
    """
    try:
        from movement.fundamental_movement import ARTICULATION_CONTROLLER
        from isaacsim.core.utils.types import ArticulationAction
        import numpy
        
        joint_count = len(config.ISAAC_ROBOT.dof_names)
        joint_positions = numpy.zeros(joint_count)
        joint_velocities = numpy.zeros(joint_count)
        
        # Define joint order (same as calibration)
        joint_order = [
            ('FL', 'hip'), ('FL', 'upper'), ('FL', 'lower'),
            ('FR', 'hip'), ('FR', 'upper'), ('FR', 'lower'),
            ('BL', 'hip'), ('BL', 'upper'), ('BL', 'lower'),
            ('BR', 'hip'), ('BR', 'upper'), ('BR', 'lower')
        ]
        
        logging.info("(standing.py): Moving all joints to neutral position in Isaac Sim...\n")
        
        # Set all joint positions at once
        for leg_id, joint_name in joint_order:
            joint_full_name = f"{leg_id}_{joint_name}"
            joint_index = config.JOINT_INDEX_MAP.get(joint_full_name)
            
            if joint_index is not None:
                servo_data = config.SERVO_CONFIG[leg_id][joint_name]
                neutral_angle = servo_data['NEUTRAL_ANGLE']  # Always 0.0 radians
                
                joint_positions[joint_index] = neutral_angle
                joint_velocities[joint_index] = 1.0  # Moderate velocity
                
                # Update the current angle in config
                config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE'] = neutral_angle
            else:
                logging.error(f"(standing.py): Joint {joint_full_name} not found in JOINT_INDEX_MAP\n")
        
        # Apply all joint positions in a single action
        action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities
        )
        ARTICULATION_CONTROLLER.apply_action(action)
        
        logging.info("(standing.py): Applied all joints to neutral positions\n")
        
    except Exception as e:
        logging.error(f"(standing.py): Failed to move all joints to neutral: {e}\n")


def _neutral_position_physical(speed, acceleration):
    """
    Set all legs to neutral position for physical robot using 3D coordinates.
    """
    # Define neutral positions for each leg
    neutral_positions = {
        'FL': config.FL_NEUTRAL,
        'FR': config.FR_NEUTRAL,
        'BL': config.BL_NEUTRAL,
        'BR': config.BR_NEUTRAL
    }
    
    logging.info("(standing.py): Moving all legs to neutral position on physical robot...\n")
    
    # Move each leg to its neutral position
    for leg_id, neutral_pos in neutral_positions.items():
        try:
            move_foot_to_pos(
                leg_id,
                neutral_pos,
                neutral_pos,  # No mid-point for neutral position
                speed,
                acceleration,
                use_bezier=False
            )
        except Exception as e:
            logging.error(f"(standing.py): Failed to move {leg_id} leg to neutral: {e}\n")


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
