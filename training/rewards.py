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

##### import config #####

import utilities.config as config

##### import necessary libraries #####

import logging

##### import necessary functions #####

from training.orientation import track_orientation


########## CREATE DEPENDENCIES ##########

##### shared global variables #####

EPISODE_STEP = 0





################################################
############### REWARD FUNCTIONS ###############
################################################


########## CALCULATE REWARD ##########

def calculate_step_reward(robot_idx, current_angles, commands, intensity):
    """
    Calculate reward for a specific robot based on its current state and actions.
    
    Args:
        robot_idx: Index of the robot to calculate reward for
        current_angles: Current joint angles for the robot
        commands: Movement commands
        intensity: Movement intensity
    
    Returns:
        float: Calculated reward for this robot
    """
    global EPISODE_STEP
    
    try:
        # Get robot-specific orientation data
        if not hasattr(track_orientation, 'robot_data') or robot_idx not in track_orientation.robot_data:
            # If no orientation data available, return neutral reward
            return 0.0
            
        robot_data = track_orientation.robot_data[robot_idx]
        
        # Get movement data for this robot
        movement_data = robot_data.get('last_movement_data', {})
        last_rotation = robot_data.get('last_rotation', 0.0)
        last_off_balance = robot_data.get('last_off_balance', 0.0)
        last_facing_deg = robot_data.get('last_facing_deg', 0.0)
        
        # Your existing reward calculation logic here, but using robot-specific data
        # For example:
        reward = 0.0
        
        # Movement reward based on command execution
        if 'w' in commands and movement_data.get('w', 0) > 0.001:
            reward += 0.1
        if 's' in commands and movement_data.get('s', 0) > 0.001:
            reward += 0.1
        if 'a' in commands and movement_data.get('a', 0) > 0.001:
            reward += 0.1
        if 'd' in commands and movement_data.get('d', 0) > 0.001:
            reward += 0.1
            
        # Rotation reward
        if 'arrowleft' in commands and last_rotation > 0:
            reward += 0.1
        if 'arrowright' in commands and last_rotation < 0:
            reward += 0.1
            
        # Balance reward (penalty for being off-balance)
        if last_off_balance > 30:  # More than 30 degrees off-balance
            reward -= 0.5
            
        # Add your other reward components here...
        
        return reward
        
    except Exception as e:
        logging.error(f"❌ Failed to calculate reward for robot {robot_idx}: {e}")
        return 0.0


########## BALANCE REWARD FUNCTION ##########

def _reward_balance(current_balance): # function to reward balance

    ##### reward and range weights #####

    perfect_balance = 0.0
    terrible_balance = 90.0
    total_range = perfect_balance - terrible_balance
    balance_reward_magnitude = 1.0
    balance_penalty_magnitude = 1.0
    perfect_percentile = 10
    good_percentile = 20
    bad_percentile = 50
    terrible_percentile = 90

    ##### calculate ranges #####

    perfect_range = calculate_desired_value('balance', total_range, perfect_percentile, terrible_balance)
    good_range = calculate_desired_value('balance', total_range, good_percentile, terrible_balance)
    bad_range = calculate_desired_value('balance', total_range, bad_percentile, terrible_balance)
    terrible_range = calculate_desired_value('balance', total_range, terrible_percentile, terrible_balance)

    #logging.debug(f"Perfect range: {perfect_range:.1f}, Good range: {good_range:.1f}, Bad range: {bad_range:.1f}, Terrible range: {terrible_range:.1f}")

    ##### calculate balance reward #####

    if current_balance < good_range: # if in good range...
        if current_balance < perfect_range: # if perfect...
            balance_reward = balance_reward_magnitude
            logging.debug(f"🔴 PERFECT BALANCE: +{balance_reward:.1f}/{balance_reward_magnitude:.1f} reward - Balance: {current_balance:.1f}°")
        else: # if good...
            balance_reward = (((100 / (perfect_range - good_range)) * (current_balance - good_range)) / 100) * balance_reward_magnitude
            logging.debug(f"🟠 GOOD BALANCE: +{balance_reward:.2f}/{balance_reward_magnitude:.1f} reward - Balance: {current_balance:.1f}°")

    elif current_balance > bad_range: # if in bad range...
        if current_balance > terrible_range: # if terrible...
            balance_reward = -balance_penalty_magnitude
            logging.debug(f"🔵 TERRIBLE BALANCE: {balance_reward:.1f}/{balance_penalty_magnitude:.1f} penalty - Balance: {current_balance:.1f}°")
        else: # if bad but not terrible...
            balance_reward_progress = (((100 / (bad_range - terrible_range)) * (current_balance - terrible_range)) / 100) * balance_penalty_magnitude
            balance_reward = -balance_penalty_magnitude + balance_reward_progress
            logging.debug(f"🟢 POOR BALANCE: {balance_reward:.2f}/{balance_penalty_magnitude:.1f} penalty - Balance: {current_balance:.1f}°")

    else: # if in middle ground...
        balance_reward = 0.0
        logging.debug(f"🟡 NEUTRAL BALANCE: No reward/penalty - Balance: {current_balance:.1f}°")
    
    return balance_reward


########## HEIGHT REWARD FUNCTION ##########

def _reward_height(current_height): # function to reward height

    ##### reward and range weights #####

    perfect_height = 0.129
    terrible_height = 0.043
    total_range = perfect_height - terrible_height
    height_reward_magnitude = 1.0
    height_penalty_magnitude = 1.0
    perfect_percentile = 10
    good_percentile = 20
    bad_percentile = 50
    terrible_percentile = 90

    ##### calculate ranges #####

    perfect_range = calculate_desired_value('height', total_range, perfect_percentile, terrible_height)
    good_range = calculate_desired_value('height', total_range, good_percentile, terrible_height)
    bad_range = calculate_desired_value('height', total_range, bad_percentile, terrible_height)
    terrible_range = calculate_desired_value('height', total_range, terrible_percentile, terrible_height)

    #logging.debug(f"Perfect range: {perfect_range:.3f}, Good range: {good_range:.3f}, Bad range: {bad_range:.3f}, Terrible range: {terrible_range:.3f}")

    ##### calculate height reward #####

    if current_height > good_range: # if in good range...
        if current_height > perfect_range: # if perfect...
            height_reward = height_reward_magnitude
            logging.debug(f"🔴 PERFECT HEIGHT: +{height_reward:.1f}/{height_reward_magnitude:.1f} reward - Height: {current_height:.3f}m")
        else: # if good...
            height_reward = (((100 / (perfect_range - good_range)) * (current_height - good_range)) / 100) * height_reward_magnitude
            logging.debug(f"🟠 GOOD HEIGHT: +{height_reward:.2f}/{height_reward_magnitude:.1f} reward - Height: {current_height:.3f}m")

    elif current_height < bad_range: # if in bad range...
        if current_height < terrible_range: # if terrible...
            height_reward = -height_penalty_magnitude
            logging.debug(f"🔵 TERRIBLE HEIGHT: {height_reward:.1f}/{height_penalty_magnitude:.1f} penalty - Height: {current_height:.3f}m")
        else: # if bad but not terrible...
            height_reward_progress = (((100 / (bad_range - terrible_range)) * (current_height - terrible_range)) / 100) * height_penalty_magnitude
            height_reward = -height_penalty_magnitude + height_reward_progress
            logging.debug(f"🟢 POOR HEIGHT: {height_reward:.2f}/{height_penalty_magnitude:.1f} penalty - Height: {current_height:.3f}m")

    else: # if in middle ground...
        height_reward = 0.0
        logging.debug(f"🟡 NEUTRAL HEIGHT: No reward/penalty - Height: {current_height:.3f}m")
    
    return height_reward


########## ROTATION REWARD FUNCTIONS ##########

##### master reward function for rotation #####

def _reward_rotation(track_orientation, commands, intensity): # function to reward rotation
    
    if hasattr(track_orientation, 'last_rotation'): # if rotation data available...
        
        ##### check if rotation is commanded #####

        actual_rotation = track_orientation.last_rotation
        rotation_magnitude = abs(actual_rotation)
        command_list = commands.split('+') if isinstance(commands, str) else commands
        rotation_commanded = any(command in ['arrowleft', 'arrowright'] for command in command_list)
        specific_rotation_command = None
        if rotation_commanded:
            for command in command_list:
                if command in ['arrowleft', 'arrowright']:
                    specific_rotation_command = command
                    break

        ##### reward rotation or stability #####

        if rotation_commanded: # if robot supposed to be rotating...
            rotation_reward = _reward_rotation_direction(specific_rotation_command, actual_rotation, rotation_magnitude)
        
        else: # if robot supposed to be standing still...
            rotation_reward = _reward_rotation_stability(rotation_magnitude)

        return rotation_reward

    else: # if no rotation data available...
        return 0.0

##### child reward function for valid rotation #####

def _reward_rotation_direction(specific_rotation_command, actual_rotation, rotation_magnitude): # function to reward valid rotation

    ##### reward and range weights #####

    perfect_rotation = 30.0
    terrible_rotation = 0.0
    total_range = perfect_rotation - terrible_rotation
    rotation_direction_reward_magnitude = 1.0
    rotation_direction_penalty_magnitude = 1.0
    perfect_percentile = 25
    quick_percentile = 50
    acceptable_percentile = 75
    slow_percentile = 100

    ##### calculate ranges #####

    perfect_range = calculate_desired_value('rotation', total_range, perfect_percentile, perfect_rotation)
    quick_range = calculate_desired_value('rotation', total_range, quick_percentile, perfect_rotation)
    acceptable_range = calculate_desired_value('rotation', total_range, acceptable_percentile, perfect_rotation)
    slow_range = calculate_desired_value('rotation', total_range, slow_percentile, perfect_rotation)

    #logging.warning(f"SUPPPOSED TO BE ROTATING {direction}!!!")
    #logging.debug(f"Perfect range: {perfect_range:.1f}, Quick range: {quick_range:.1f}, Acceptable range: {acceptable_range:.1f}, Slow range: {slow_range:.1f}")
    
    ##### determine rotation direction #####

    if specific_rotation_command == 'arrowleft': # if left rotation...
        direction = "LEFT"
        if actual_rotation > 0:
            correct_rotation_direction = True
        else:
            correct_rotation_direction = False
    elif specific_rotation_command == 'arrowright': # if right rotation...
        direction = "RIGHT"
        if actual_rotation < 0:
            correct_rotation_direction = True
        else:
            correct_rotation_direction = False
    else: # if rotation is not commanded or in the wrong direction...
        direction = "N/A"
        correct_rotation_direction = False

    ##### calculate rotation reward #####

    if correct_rotation_direction: # if rotation in correct direction...

        if rotation_magnitude > perfect_range:
            rotation_direction_reward = rotation_direction_reward_magnitude
            logging.debug(f"🔴 PERFECT ROTATION MOVEMENT: +{rotation_direction_reward:.1f}/{rotation_direction_reward_magnitude:.1f} reward - Rotation: {rotation_magnitude:.1f}°")
        elif rotation_magnitude > quick_range:
            rotation_direction_reward = (((100 / perfect_range) * rotation_magnitude) / 100) * rotation_direction_reward_magnitude
            logging.debug(f"🟠 QUICK ROTATION MOVEMENT: +{rotation_direction_reward:.2f}/{rotation_direction_reward_magnitude:.1f} reward - Rotation: {rotation_magnitude:.1f}°")
        elif rotation_magnitude > acceptable_range:
            rotation_direction_reward = (((100 / perfect_range) * rotation_magnitude) / 100) * rotation_direction_reward_magnitude
            logging.debug(f"🟡 ACCEPTABLE ROTATION MOVEMENT: +{rotation_direction_reward:.2f}/{rotation_direction_reward_magnitude:.1f} reward - Rotation: {rotation_magnitude:.1f}°")
        else:
            rotation_direction_reward = (((100 / perfect_range) * rotation_magnitude) / 100) * rotation_direction_reward_magnitude
            logging.debug(f"🟢 SLOW ROTATION MOVEMENT: +{rotation_direction_reward:.1f}/{rotation_direction_reward_magnitude:.1f} reward - Rotation: {rotation_magnitude:.1f}°")

    else: # if rotation in wrong direction...
        rotation_direction_reward = -rotation_direction_penalty_magnitude
        logging.debug(f"🔵 WRONG ROTATION DIRECTION: {rotation_direction_reward:.1f}/{rotation_direction_penalty_magnitude:.1f} penalty - Expected: {actual_rotation}, Got: {actual_rotation:.1f}°, Rotation: {rotation_magnitude:.1f}°")

    return rotation_direction_reward

##### child reward function for no rotation #####

def _reward_rotation_stability(rotation_magnitude): # function to reward no rotation

    ##### reward and range weights #####

    perfect_rotation = 0.0
    terrible_rotation = 30.0
    total_range = perfect_rotation - terrible_rotation
    rotation_stability_reward_magnitude = 1.0
    rotation_stability_penalty_magnitude = 1.0
    perfect_percentile = 10
    good_percentile = 20
    bad_percentile = 50
    terrible_percentile = 90
    
    ##### calculate ranges #####

    perfect_range = calculate_desired_value('no_rotation', total_range, perfect_percentile, terrible_rotation)
    good_range = calculate_desired_value('no_rotation', total_range, good_percentile, terrible_rotation)
    bad_range = calculate_desired_value('no_rotation', total_range, bad_percentile, terrible_rotation)
    terrible_range = calculate_desired_value('no_rotation', total_range, terrible_percentile, terrible_rotation)

    #logging.warning("SUPPPOSED TO BE STILL!!!")
    #logging.debug(f"Perfect range: {perfect_range:.1f}, Good range: {good_range:.1f}, Bad range: {bad_range:.1f}, Terrible range: {terrible_range:.1f}")

    ##### calculate no rotation reward #####

    if rotation_magnitude < good_range:
        if rotation_magnitude < perfect_range:
            rotation_stability_reward = rotation_stability_reward_magnitude
            logging.debug(f"🔴 PERFECT ROTATION STABILITY: +{rotation_stability_reward:.1f}/{rotation_stability_reward_magnitude:.1f} reward - Rotation: {rotation_magnitude:.1f}°")
        else:
            rotation_stability_reward = (((100 / (good_range - perfect_range)) * (rotation_magnitude - perfect_range)) / 100) * rotation_stability_reward_magnitude
            logging.debug(f"🟠 GOOD ROTATION STABILITY: +{rotation_stability_reward:.2f}/{rotation_stability_reward_magnitude:.1f} reward - Rotation: {rotation_magnitude:.1f}°")

    elif rotation_magnitude > bad_range:
        if rotation_magnitude > terrible_range:
            rotation_stability_reward = -rotation_stability_penalty_magnitude
            logging.debug(f"🔵 TERRIBLE ROTATION STABILITY: {rotation_stability_reward:.1f}/{rotation_stability_penalty_magnitude:.1f} penalty - Rotation: {rotation_magnitude:.1f}°")
        else:
            rotation_stability_reward_progress = (((100 / (bad_range - terrible_range)) * (rotation_magnitude - terrible_range)) / 100) * rotation_stability_penalty_magnitude
            rotation_stability_reward = -rotation_stability_penalty_magnitude + rotation_stability_reward_progress
            logging.debug(f"🟢 POOR ROTATION STABILITY: {rotation_stability_reward:.2f}/{rotation_stability_penalty_magnitude:.1f} penalty - Rotation: {rotation_magnitude:.1f}°")

    else: # if in middle ground...
        rotation_stability_reward = 0.0
        logging.debug(f"🟡 NEUTRAL ROTATION STABILITY: No reward/penalty - Rotation: {rotation_magnitude:.1f}°")

    return rotation_stability_reward


########## MOVEMENT REWARD FUNCTIONS ##########

##### master reward fucntion for movement #####

def _reward_movement(track_orientation, commands, intensity): # function to reward movement

    movement_reward = 0.0 # initialize movement reward

    if hasattr(track_orientation, 'last_movement_data'): # if movement data available...
        
        try: # attempt to get movement data...

            ##### set orientation and movement variables #####

            movement_data = track_orientation.last_movement_data
            total_displacement = {
                'w': movement_data.get('w'),
                's': movement_data.get('s'),
                'a': movement_data.get('a'),
                'd': movement_data.get('d')
            }
            command_list = commands.split('+') if isinstance(commands, str) else commands

            # if there are commands and any valid 'wasd' in command list...
            if commands and any(command in ['w', 'a', 's', 'd'] for command in command_list):
                
                # remove 'arrowleft' and 'arrowright' from command list
                filtered_commands = [command for command in command_list if command not in ['arrowleft', 'arrowright']]

                if len(filtered_commands) > 1: # if multidirectional command (i.e. 'w+a')...

                    for command in filtered_commands: # loop through each command
                        movement_reward += _reward_movement_direction(command, total_displacement, intensity, True)

                    return movement_reward

                else: # if unidirectional command (i.e. 'w')...
                    command = filtered_commands[0]  # Get the single command
                    movement_reward += _reward_movement_direction(command, total_displacement, intensity, False)
                    return movement_reward

            else: # if robot is supposed to be still...
                movement_reward += _reward_movement_stillness(total_displacement)
                return movement_reward

        except Exception as e: # if movement analysis failed...
            return movement_reward

    else: # if no movement data available...
        return movement_reward

##### child reward function for valid movement #####

def _reward_movement_direction(command, total_displacement, intensity, multidirectional): # function to reward valid movement

    ##### set variables #####

    movement_direction_reward = 0.0 # initialize direction reward
    commanded_displacement = total_displacement.get(command, 0) # displacement in commanded direction
    
    ##### get perpendicular displacements for drift calculation #####

    if command == 'w' or command == 's': # if forward or backward movement...
        perpendicular_displacement = abs(total_displacement.get('a', 0)) + abs(total_displacement.get('d', 0))
    else: # if left or right movement...
        perpendicular_displacement = abs(total_displacement.get('w', 0)) + abs(total_displacement.get('s', 0))

    ##### reward and range weights #####

    perfect_movement = 0.1 # 10cm of movement
    terrible_movement = 0.0
    acceptable_drift = 0.05 # 5cm of drift allowed
    total_range = perfect_movement - terrible_movement
    movement_reward_magnitude = 1.0
    movement_penalty_magnitude = 1.0
    perfect_percentile = 25
    good_percentile = 50
    acceptable_percentile = 75
    slow_percentile = 100
    
    ##### calculate ranges #####

    perfect_range = calculate_desired_value('rotation', total_range, perfect_percentile, perfect_movement)
    good_range = calculate_desired_value('rotation', total_range, good_percentile, perfect_movement)
    acceptable_range = calculate_desired_value('rotation', total_range, acceptable_percentile, perfect_movement)
    slow_range = calculate_desired_value('rotation', total_range, slow_percentile, perfect_movement)

    #logging.warning(f"SUPPPOSED TO BE MOVING IN {command.upper()} DIRECTION!!!")
    #logging.debug(f"Perfect range: {perfect_range:.3f}, Good range: {good_range:.3f}, Acceptable range: {acceptable_range:.3f}, Slow range: {slow_range:.3f}")

    ##### calculate movement reward #####

    if commanded_displacement > 0: # if moving in correct direction...
        if commanded_displacement > perfect_range: # if perfect...
            movement_direction_reward = movement_reward_magnitude
            logging.debug(f"🔴 PERFECT '{command.upper()}' MOVEMENT: +{movement_direction_reward:.1f}/{movement_reward_magnitude:.1f} reward - Displacement: {commanded_displacement:.3f}m")
        elif commanded_displacement > good_range: # if good...
            movement_direction_reward = (((100 / perfect_range) * commanded_displacement) / 100) * movement_reward_magnitude
            logging.debug(f"🟠 QUICK '{command.upper()}' MOVEMENT: +{movement_direction_reward:.2f}/{movement_reward_magnitude:.1f} reward - Displacement: {commanded_displacement:.3f}m")
        elif commanded_displacement > acceptable_range: # if acceptable...
            movement_direction_reward = (((100 / perfect_range) * commanded_displacement) / 100) * movement_reward_magnitude
            logging.debug(f"🟡 ACCEPTABLE '{command.upper()}' MOVEMENT: +{movement_direction_reward:.2f}/{movement_reward_magnitude:.1f} reward - Displacement: {commanded_displacement:.3f}m")
        else: # if slow...
            movement_direction_reward = (((100 / perfect_range) * commanded_displacement) / 100) * movement_reward_magnitude
            logging.debug(f"🟢 SLOW '{command.upper()}' MOVEMENT: +{movement_direction_reward:.1f}/{movement_reward_magnitude:.1f} reward - Displacement: {commanded_displacement:.3f}m")
        
        # if unidirectional and significant perpendicular movement...
        if not multidirectional and perpendicular_displacement > acceptable_drift:
            drift_penalty = min(perpendicular_displacement * 10, movement_penalty_magnitude)
            movement_direction_reward -= drift_penalty
            logging.debug(f"⚠️ DRIFT PENALTY: -{drift_penalty:.2f} for {perpendicular_displacement:.3f}m perpendicular movement")
            
    else: # if not moving in correct direction...
        movement_direction_reward = -movement_penalty_magnitude
        logging.debug(f"🔵 NO '{command.upper()}' MOVEMENT: {movement_direction_reward:.1f}/{movement_penalty_magnitude:.1f} penalty - Displacement: {commanded_displacement:.3f}m")

    return movement_direction_reward

##### child reward function for no movement #####

def _reward_movement_stillness(total_displacement): # function to reward stillness

    ##### reward and range weights #####

    perfect_movement = 0.0
    terrible_movement = 0.1 # 10cm of movement
    total_range = terrible_movement - perfect_movement
    movement_stillness_reward_magnitude = 1.0
    movement_stillness_penalty_magnitude = 1.0
    perfect_percentile = 10
    good_percentile = 20
    bad_percentile = 50
    terrible_percentile = 90
    
    ##### calculate total movement magnitude #####
    
    total_movement = abs(total_displacement.get('w', 0)) + abs(total_displacement.get('s', 0)) + \
                     abs(total_displacement.get('a', 0)) + abs(total_displacement.get('d', 0))
    
    ##### calculate ranges #####

    perfect_range = calculate_desired_value('no_movement', total_range, perfect_percentile, perfect_movement)
    good_range = calculate_desired_value('no_movement', total_range, good_percentile, perfect_movement)
    bad_range = calculate_desired_value('no_movement', total_range, bad_percentile, perfect_movement)
    terrible_range = calculate_desired_value('no_movement', total_range, terrible_percentile, perfect_movement)

    #logging.warning("SUPPPOSED TO BE STILL!!!")
    #logging.debug(f"Perfect range: {perfect_range:.3f}, Good range: {good_range:.3f}, Bad range: {bad_range:.3f}, Terrible range: {terrible_range:.3f}")

    ##### calculate no movement reward #####

    if total_movement < good_range:
        if total_movement < perfect_range:
            movement_stillness_reward = movement_stillness_reward_magnitude
            logging.debug(f"🔴 PERFECT MOVEMENT STILLNESS: +{movement_stillness_reward:.1f}/{movement_stillness_reward_magnitude:.1f} reward - Movement: {total_movement:.3f}m")
        else:
            movement_stillness_reward = (((100 / (good_range - perfect_range)) * (total_movement - perfect_range)) / 100) * movement_stillness_reward_magnitude
            logging.debug(f"🟠 GOOD MOVEMENT STILLNESS: +{movement_stillness_reward:.2f}/{movement_stillness_reward_magnitude:.1f} reward - Movement: {total_movement:.3f}m")

    elif total_movement > bad_range:
        if total_movement > terrible_range:
            movement_stillness_reward = -movement_stillness_penalty_magnitude
            logging.debug(f"🔵 TERRIBLE MOVEMENT STILLNESS: {movement_stillness_reward:.1f}/{movement_stillness_penalty_magnitude:.1f} penalty - Movement: {total_movement:.3f}m")
        else:
            movement_progress = (((100 / (bad_range - terrible_range)) * (total_movement - terrible_range)) / 100) * movement_stillness_penalty_magnitude
            movement_stillness_reward = -movement_stillness_penalty_magnitude + movement_progress
            logging.debug(f"🟢 POOR MOVEMENT STILLNESS: {movement_stillness_reward:.2f}/{movement_stillness_penalty_magnitude:.1f} penalty - Movement: {total_movement:.3f}m")

    else: # if in middle ground...
        movement_stillness_reward = 0.0
        logging.debug(f"🟡 NEUTRAL MOVEMENT STILLNESS: No reward/penalty - Movement: {total_movement:.3f}m")

    return movement_stillness_reward


########## REWARD RANGE CALCULATION FUNCTION ##########

def calculate_desired_value(reward_type, total_range, selected_range, special_value): # function to calculate desired figure for reward calculation

    if reward_type == 'balance' or reward_type == 'no_rotation': # perfect value is 0, terrible value greater than 0

        desired_value = (((-1 * total_range) / 100) * selected_range)

    elif reward_type == 'height': # the numbers are small so floating points are being an issue

        scale_factor = 1000000
        total_range = total_range * scale_factor
        special_value = special_value * scale_factor

        desired_value = ((total_range / 100) * (100 - selected_range) + special_value) / scale_factor

    elif reward_type == 'rotation': # degrees from any value greater than 0 to 0
        desired_value = special_value - (((total_range) / 100) * selected_range)

    elif reward_type == 'no_movement':
        desired_value = ((total_range / 100) * selected_range)

    return desired_value