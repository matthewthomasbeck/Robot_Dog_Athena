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

def calculate_step_reward(current_angles, commands, intensity): # function to calculate reward for step

    ##### set variables #####

    global EPISODE_STEP

    center_pos, facing_deg = track_orientation()
    if center_pos is None:
        return 0.0
    reward = 0.0
    was_perfect = True

    ##### fall detection #####

    current_height = center_pos[2]
    if hasattr(track_orientation, 'last_off_balance'):
        current_balance = track_orientation.last_off_balance
    else:
        current_balance = 0.0
    has_fallen = current_balance > 90.0

    ##### reward balance #####
    
    balance_reward = reward_balance(current_balance)
    if balance_reward < 1.0:
        was_perfect = False

    reward += balance_reward

    ##### reward height #####

    height_reward = reward_height(current_height)
    if height_reward < 1.0:
        was_perfect = False

    reward += height_reward

    ##### reward movement #####

    movement_reward = reward_movement(track_orientation, commands)
    if movement_reward < 1.0:
        was_perfect = False
    
    reward += movement_reward

    ##### reward rotation #####
    
    rotation_reward = reward_rotation(track_orientation, commands)
    if rotation_reward < 2.0:
        was_perfect = False
    
    reward += rotation_reward

    ##### reward perfect execution #####

    if was_perfect and commands:
        perfect_bonus = 10.0
        reward += perfect_bonus
        logging.debug(f"🏆 PERFECT EXECUTION! +{perfect_bonus:.1f} MASSIVE BONUS - All commands executed flawlessly!")
    elif commands:
        logging.debug(f"📊 Good execution, but not perfect - no bonus this time")
    
    ##### punish fall #####

    if has_fallen:
        fall_penalty = -100
        logging.debug(f"EPISODE FAILURE -100 points (robot fell over)")
        EPISODE_STEP = config.TRAINING_CONFIG['max_steps_per_episode']  # Force episode end

        return fall_penalty

    ##### clamp reward #####

    elif not has_fallen:
        reward = max(-1.0, min(1.0, reward))

    return reward


########## BALANCE REWARD FUNCTION ##########

def reward_balance(current_balance): # function to reward balance

    ##### reward and range weights #####

    perfect_balance = 0.0
    terrible_balance = 90.0
    total_range = perfect_balance - terrible_balance
    balance_reward_magnitude = 0.5
    balance_penalty_magnitude = 2.0
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
            balance_progress = (((100 / (bad_range - terrible_range)) * (current_balance - terrible_range)) / 100) * balance_penalty_magnitude
            balance_reward = -balance_penalty_magnitude + balance_progress
            logging.debug(f"🟢 POOR BALANCE: {balance_reward:.2f}/{balance_penalty_magnitude:.1f} penalty - Balance: {current_balance:.1f}°")

    else: # if in middle ground...
        balance_reward = 0.0
        logging.debug(f"🟡 NEUTRAL BALANCE: No reward/penalty - Balance: {current_balance:.1f}°")
    
    return balance_reward


########## HEIGHT REWARD FUNCTION ##########

def reward_height(current_height): # function to reward height

    ##### reward and range weights #####

    perfect_height = 0.129
    terrible_height = 0.043
    total_range = perfect_height - terrible_height
    height_reward_magnitude = 0.5
    height_penalty_magnitude = 2.0
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
            height_progress = (((100 / (bad_range - terrible_range)) * (current_height - terrible_range)) / 100) * height_penalty_magnitude
            height_reward = -height_penalty_magnitude + height_progress
            logging.debug(f"🟢 POOR HEIGHT: {height_reward:.2f}/{height_penalty_magnitude:.1f} penalty - Height: {current_height:.3f}m")

    else: # if in middle ground...
        height_reward = 0.0
        logging.debug(f"🟡 NEUTRAL HEIGHT: No reward/penalty - Height: {current_height:.3f}m")
    
    return height_reward


########## ROTATION REWARD FUNCTION ##########

def reward_rotation(track_orientation, commands): # function to reward rotation
    
    if hasattr(track_orientation, 'last_rotation'):
        
        ##### check if rotation is commanded #####

        actual_rotation = track_orientation.last_rotation
        rotation_magnitude = abs(actual_rotation)
        command_list = commands.split('+') if isinstance(commands, str) else commands
        rotation_commanded = any(cmd in ['arrowleft', 'arrowright'] for cmd in command_list)
        specific_rotation_command = None
        if rotation_commanded:
            for cmd in command_list:
                if cmd in ['arrowleft', 'arrowright']:
                    specific_rotation_command = cmd
                    break

        ##### reward no rotation #####

        if not rotation_commanded: # if robot supposed to be standing still...

            ##### reward and range weights #####

            perfect_rotation = 0.0
            terrible_rotation = 30.0
            total_range = perfect_rotation - terrible_rotation
            rotation_reward_magnitude = 4.0
            rotation_penalty_magnitude = 2.0
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
                    rotation_reward = rotation_reward_magnitude
                    logging.debug(f"🔴 PERFECT STABILITY: +{rotation_reward:.1f}/{rotation_reward_magnitude:.1f} reward - Rotation: {rotation_magnitude:.1f}°")
                else:
                    rotation_reward = (((100 / (good_range - perfect_range)) * (rotation_magnitude - perfect_range)) / 100) * rotation_reward_magnitude
                    logging.debug(f"🟠 GOOD STABILITY: +{rotation_reward:.2f}/{rotation_reward_magnitude:.1f} reward - Rotation: {rotation_magnitude:.1f}°")

            elif rotation_magnitude > bad_range:
                if rotation_magnitude > terrible_range:
                    rotation_reward = -rotation_penalty_magnitude
                    logging.debug(f"🔵 TERRIBLE STABILITY: {rotation_reward:.1f}/{rotation_penalty_magnitude:.1f} penalty - Rotation: {rotation_magnitude:.1f}°")
                else:
                    rotation_progress = (((100 / (bad_range - terrible_range)) * (rotation_magnitude - terrible_range)) / 100) * rotation_penalty_magnitude
                    rotation_reward = -rotation_penalty_magnitude + rotation_progress
                    logging.debug(f"🟢 POOR STABILITY: {rotation_reward:.2f}/{rotation_penalty_magnitude:.1f} penalty - Rotation: {rotation_magnitude:.1f}°")

            else: # if in middle ground...
                rotation_reward = 0.0
                logging.debug(f"🟡 MIDDLE STABILITY: No reward/penalty - Rotation: {rotation_magnitude:.1f}°")

        ##### reward rotation #####

        else: # if robot supposed to be rotating...

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

            ##### reward and range weights #####

            perfect_rotation = 30.0
            terrible_rotation = 0.0
            total_range = perfect_rotation - terrible_rotation
            rotation_reward_magnitude = 2.0
            rotation_penalty_magnitude = 2.0
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

            ##### calculate rotation reward #####

            if correct_rotation_direction: # if rotation in correct direction...

                if rotation_magnitude > perfect_range:
                    rotation_reward = rotation_reward_magnitude
                    logging.debug(f"🔴 PERFECT ROTATION: +{rotation_reward:.1f}/{rotation_reward_magnitude:.1f} reward - Rotation: {rotation_magnitude:.1f}°")
                elif rotation_magnitude > quick_range:
                    rotation_reward = (((100 / perfect_range) * rotation_magnitude) / 100) * rotation_reward_magnitude
                    logging.debug(f"🟠 QUICK ROTATION: +{rotation_reward:.2f}/{rotation_reward_magnitude:.1f} reward - Rotation: {rotation_magnitude:.1f}°")
                elif rotation_magnitude > acceptable_range:
                    rotation_reward = (((100 / perfect_range) * rotation_magnitude) / 100) * rotation_reward_magnitude
                    logging.debug(f"🟡 ACCEPTABLE ROTATION: +{rotation_reward:.2f}/{rotation_reward_magnitude:.1f} reward - Rotation: {rotation_magnitude:.1f}°")
                else:
                    rotation_reward = (((100 / perfect_range) * rotation_magnitude) / 100) * rotation_reward_magnitude
                    logging.debug(f"🟢 SLOW ROTATION: +{rotation_reward:.1f}/{rotation_reward_magnitude:.1f} reward - Rotation: {rotation_magnitude:.1f}°")

            else: # if rotation in wrong direction...
                rotation_reward = -rotation_penalty_magnitude
                logging.debug(f"🔵 WRONG ROTATION: {rotation_reward:.1f}/{rotation_penalty_magnitude:.1f} penalty - Expected: {actual_rotation}, Got: {actual_rotation:.1f}°, Rotation: {rotation_magnitude:.1f}°")

        return rotation_reward
    else:
        return 0.0


########## CALCULATE COMPARISON VALUE ##########

def calculate_desired_value(reward_type, total_range, selected_range, special_value): # function to calculate desired figure for reward calculation

    if reward_type == 'balance' or reward_type == 'no_rotation': # degrees from 0 to any value greater than 0
        desired_value = (((-1 * total_range) / 100) * selected_range)

    elif reward_type == 'height': # the numbers are small so floating points are being an issue

        scale_factor = 1000000
        total_range = total_range * scale_factor
        special_value = special_value * scale_factor

        desired_value = ((total_range / 100) * (100 - selected_range) + special_value) / scale_factor

    elif reward_type == 'rotation': # degrees from any value greater than 0 to 0
        desired_value = special_value - (((total_range) / 100) * selected_range)

    return desired_value


########## MOVEMENT REWARD FUNCTION ##########

def reward_movement(track_orientation, commands): # function to reward movement
    
    if not commands:
        return 0.0
    
    # Parse complex commands like 'w+a+arrowleft'
    command_list = commands.split('+') if isinstance(commands, str) else commands
    
    # Get movement data from the last call to track_orientation
    try:
        if hasattr(track_orientation, 'last_movement_data'):
            movement_data = track_orientation.last_movement_data
            
            # Check rotation first - must be minimal to get any movement reward
            rotation_during_movement = abs(track_orientation.last_rotation) if hasattr(track_orientation, 'last_rotation') else 0
            
            if rotation_during_movement < 2.0:  # Acceptable rotation for all directions
                total_movement_reward = 0.0
                
                # Check each movement command and reward correct execution, punish wrong directions
                for cmd in command_list:
                    if cmd == 'w':  # Forward movement commanded
                        forward_movement = movement_data.get('w', 0)
                        left_movement = movement_data.get('a', 0)
                        right_movement = movement_data.get('d', 0)
                        backward_movement = movement_data.get('s', 0)

                        # Reward forward movement (commanded direction)
                        if forward_movement > 3.33:  # Within 33% of target (3.33cm to 5cm)
                            if forward_movement > 4.5:  # Within 10% of target (4.5cm to 5cm)
                                movement_reward = 3.0
                                logging.debug(f"🔴 PERFECT FORWARD: +{movement_reward:.1f} reward - Forward: {forward_movement:.3f}m")
                            else:  # Within 33% of target (3.33cm to 4.5cm)
                                movement_progress = 1.0 - ((5.0 - forward_movement) / 1.67) ** 2
                                movement_reward = 0.3 + 2.7 * movement_progress
                                logging.debug(f"🟠 GOOD FORWARD: +{movement_reward:.2f} reward - Forward: {forward_movement:.3f}m")
                        elif forward_movement < 0.0:  # Moving backward (wrong direction)
                            if forward_movement < -4.5:  # Within 10% of bad (-4.5cm to -5cm)
                                movement_reward = -3.0
                                logging.debug(f"🔵 TERRIBLE FORWARD: {movement_reward:.1f} penalty - Forward: {forward_movement:.3f}m")
                            else:  # Within 33% of bad (0cm to -4.5cm)
                                movement_progress = max(0.0, min(1.0, 1.0 - ((forward_movement + 5.0) / 4.5) ** 2))
                                movement_reward = -0.3 - 2.7 * movement_progress
                                logging.debug(f"🟢 POOR FORWARD: {movement_reward:.2f} penalty - Forward: {forward_movement:.3f}m")
                        else:  # Middle ground (0cm to 3.33cm) - no reward, no penalty
                            movement_reward = 0.0
                            logging.debug(f"🟡 MIDDLE FORWARD: No reward/penalty - Forward: {forward_movement:.3f}m")

                        total_movement_reward += movement_reward

                        # PUNISH movement in wrong directions (even if not commanded)
                        if abs(left_movement) > 0.5:  # Moving left when should go forward
                            wrong_direction_penalty = -1.5
                            total_movement_reward += wrong_direction_penalty
                            logging.debug(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving LEFT: {left_movement:.3f}m when FORWARD commanded")

                        if abs(right_movement) > 0.5:  # Moving right when should go forward
                            wrong_direction_penalty = -1.5
                            total_movement_reward += wrong_direction_penalty
                            logging.debug(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving RIGHT: {right_movement:.3f}m when FORWARD commanded")

                        if abs(backward_movement) > 0.5:  # Moving backward when should go forward
                            wrong_direction_penalty = -1.5
                            total_movement_reward += wrong_direction_penalty
                            logging.debug(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving BACKWARD: {backward_movement:.3f}m when FORWARD commanded")

                    elif cmd == 's':  # Backward movement commanded
                        # Similar logic for backward - reward backward, punish forward/left/right
                        backward_movement = movement_data.get('s', 0)
                        forward_movement = movement_data.get('w', 0)
                        left_movement = movement_data.get('a', 0)
                        right_movement = movement_data.get('d', 0)

                        # Reward backward movement (commanded direction)
                        if backward_movement > 3.33:  # Within 33% of target (3.33cm to 5cm)
                            if backward_movement > 4.5:  # Within 10% of target (4.5cm to 5cm)
                                movement_reward = 3.0
                                logging.debug(f"🔴 PERFECT BACKWARD: +{movement_reward:.1f} reward - Backward: {backward_movement:.3f}m")
                            else:  # Within 33% of target (3.33cm to 4.5cm)
                                movement_progress = 1.0 - ((5.0 - backward_movement) / 1.67) ** 2
                                movement_reward = 0.3 + 2.7 * movement_progress
                                logging.debug(f"🟠 GOOD BACKWARD: +{movement_reward:.2f} reward - Backward: {backward_movement:.3f}m")
                        elif backward_movement < 0.0:  # Moving forward (wrong direction)
                            if backward_movement < -4.5:  # Within 10% of bad (-4.5cm to -5cm)
                                movement_reward = -3.0
                                logging.debug(f"🔵 TERRIBLE BACKWARD: {movement_reward:.1f} penalty - Backward: {backward_movement:.3f}m")
                            else:  # Within 33% of bad (0cm to -4.5cm)
                                movement_progress = max(0.0, min(1.0, 1.0 - ((backward_movement + 5.0) / 4.5) ** 2))
                                movement_reward = -0.3 - 2.7 * movement_progress
                                logging.debug(f"🟢 POOR BACKWARD: {movement_reward:.2f} penalty - Backward: {backward_movement:.3f}m")
                        else:  # Middle ground (0cm to 3.33cm) - no reward, no penalty
                            movement_reward = 0.0
                            logging.debug(f"🟡 MIDDLE BACKWARD: No reward/penalty - Backward: {backward_movement:.3f}m")

                        total_movement_reward += movement_reward

                        # PUNISH movement in wrong directions
                        if abs(forward_movement) > 0.5:  # Moving forward when should go backward
                            wrong_direction_penalty = -1.5
                            total_movement_reward += wrong_direction_penalty
                            logging.debug(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving FORWARD: {forward_movement:.3f}m when BACKWARD commanded")

                        if abs(left_movement) > 0.5:  # Moving left when should go backward
                            wrong_direction_penalty = -1.5
                            total_movement_reward += wrong_direction_penalty
                            logging.debug(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving LEFT: {left_movement:.3f}m when BACKWARD commanded")

                        if abs(right_movement) > 0.5:  # Moving right when should go backward
                            wrong_direction_penalty = -1.5
                            total_movement_reward += wrong_direction_penalty
                            logging.debug(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving RIGHT: {right_movement:.3f}m when BACKWARD commanded")

                    elif cmd == 'a':  # Left movement commanded
                        # Similar logic for left - reward left, punish forward/backward/right
                        left_movement = movement_data.get('a', 0)
                        forward_movement = movement_data.get('w', 0)
                        backward_movement = movement_data.get('s', 0)
                        right_movement = movement_data.get('d', 0)

                        # Reward left movement (commanded direction)
                        if left_movement > 3.33:  # Within 33% of target (3.33cm to 5cm)
                            if left_movement > 4.5:  # Within 10% of target (4.5cm to 5cm)
                                movement_reward = 3.0
                                logging.debug(f"🔴 PERFECT LEFT: +{movement_reward:.1f} reward - Left: {left_movement:.3f}m")
                            else:  # Within 33% of target (3.33cm to 4.5cm)
                                movement_progress = 1.0 - ((5.0 - left_movement) / 1.67) ** 2
                                movement_reward = 0.3 + 2.7 * movement_progress
                                logging.debug(f"🟠 GOOD LEFT: +{movement_reward:.2f} reward - Left: {left_movement:.3f}m")
                        elif left_movement < 0.0:  # Moving right (wrong direction)
                            if left_movement < -4.5:  # Within 10% of bad (-4.5cm to -5cm)
                                movement_reward = -3.0
                                logging.debug(f"🔵 TERRIBLE LEFT: {movement_reward:.1f} penalty - Left: {left_movement:.3f}m")
                            else:  # Within 33% of bad (0cm to -4.5cm)
                                movement_progress = max(0.0, min(1.0, 1.0 - ((left_movement + 5.0) / 4.5) ** 2))
                                movement_reward = -0.3 - 2.7 * movement_progress
                                logging.debug(f"🟢 POOR LEFT: {movement_reward:.2f} penalty - Left: {left_movement:.3f}m")
                        else:  # Middle ground (0cm to 3.33cm) - no reward, no penalty
                            movement_reward = 0.0
                            logging.debug(f"🟡 MIDDLE LEFT: No reward/penalty - Left: {left_movement:.3f}m")

                        total_movement_reward += movement_reward

                        # PUNISH movement in wrong directions
                        if abs(forward_movement) > 0.5:  # Moving forward when should go left
                            wrong_direction_penalty = -1.5
                            total_movement_reward += wrong_direction_penalty
                            logging.debug(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving FORWARD: {forward_movement:.3f}m when LEFT commanded")

                        if abs(backward_movement) > 0.5:  # Moving backward when should go left
                            wrong_direction_penalty = -1.5
                            total_movement_reward += wrong_direction_penalty
                            logging.debug(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving BACKWARD: {backward_movement:.3f}m when LEFT commanded")

                        if abs(right_movement) > 0.5:  # Moving right when should go left
                            wrong_direction_penalty = -1.5
                            total_movement_reward += wrong_direction_penalty
                            logging.debug(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving RIGHT: {right_movement:.3f}m when LEFT commanded")

                    elif cmd == 'd':  # Right movement commanded
                        # Similar logic for right - reward right, punish forward/backward/left
                        right_movement = movement_data.get('d', 0)
                        forward_movement = movement_data.get('w', 0)
                        backward_movement = movement_data.get('s', 0)
                        left_movement = movement_data.get('a', 0)

                        # Reward right movement (commanded direction)
                        if right_movement > 3.33:  # Within 33% of target (3.33cm to 5cm)
                            if right_movement > 4.5:  # Within 10% of target (4.5cm to 5cm)
                                movement_reward = 3.0
                                logging.debug(f"🔴 PERFECT RIGHT: +{movement_reward:.1f} reward - Right: {right_movement:.3f}m")
                            else:  # Within 33% of target (3.33cm to 4.5cm)
                                movement_progress = 1.0 - ((5.0 - right_movement) / 1.67) ** 2
                                movement_reward = 0.3 + 2.7 * movement_progress
                                logging.debug(f"🟠 GOOD RIGHT: +{movement_reward:.2f} reward - Right: {right_movement:.3f}m")
                        elif right_movement < 0.0:  # Moving left (wrong direction)
                            if right_movement < -4.5:  # Within 10% of bad (-4.5cm to -5cm)
                                movement_reward = -3.0
                                logging.debug(f"🔵 TERRIBLE RIGHT: {movement_reward:.1f} penalty - Right: {right_movement:.3f}m")
                            else:  # Within 33% of bad (0cm to -4.5cm)
                                movement_progress = max(0.0, min(1.0, 1.0 - ((right_movement + 5.0) / 4.5) ** 2))
                                movement_reward = -0.3 - 2.7 * movement_progress
                                logging.debug(f"🟢 BAD RIGHT: {movement_reward:.2f} penalty - Right: {right_movement:.3f}m")
                        else:  # Middle ground (0cm to 3.33cm) - no reward, no penalty
                            movement_reward = 0.0
                            logging.debug(f"🟡 MIDDLE RIGHT: No reward/penalty - Right: {right_movement:.3f}m")

                        total_movement_reward += movement_reward

                        # PUNISH movement in wrong directions
                        if abs(forward_movement) > 0.5:  # Moving forward when should go right
                            wrong_direction_penalty = -1.5
                            total_movement_reward += wrong_direction_penalty
                            logging.debug(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving FORWARD: {forward_movement:.3f}m when RIGHT commanded")

                        if abs(backward_movement) > 0.5:  # Moving backward when should go right
                            wrong_direction_penalty = -1.5
                            total_movement_reward += wrong_direction_penalty
                            logging.debug(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving BACKWARD: {backward_movement:.3f}m when RIGHT commanded")

                        if abs(left_movement) > 0.5:  # Moving left when should go right
                            wrong_direction_penalty = -1.5
                            total_movement_reward += wrong_direction_penalty
                            logging.debug(f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving LEFT: {left_movement:.3f}m when RIGHT commanded")

                return total_movement_reward
            else:
                # Too much rotation during movement - no movement reward
                return 0.0
        else:
            # No movement data available
            return 0.0
    except Exception as e:
        # If movement analysis fails, give neutral reward
        return 0.0
