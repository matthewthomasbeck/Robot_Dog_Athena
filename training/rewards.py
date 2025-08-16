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

##### import necessary functions #####

from training.orientation import track_orientation





################################################
############### REWARD FUNCTIONS ###############
################################################


########## CALCULATE REWARD ##########

def calculate_step_reward(current_angles, commands, intensity):
    """
    Reward system that rewards balance, proper height, and correct movement execution.
    Falls result in massive penalty and episode termination.

    This function is called after get_rl_action_blind() in the while True loop in control_logic.

    Args:
        current_angles: Current joint angles for all legs
        commands: Movement commands (e.g., 'w', 's', 'a', 'd', 'w+a+arrowleft')
        intensity: Movement intensity (1-10) (ignored for now)

    Returns:
        float: Reward value for this step
    """
    # Get fresh position and orientation data
    center_pos, facing_deg = track_orientation()

    if center_pos is None:
        return 0.0  # Can't calculate reward without position data

    # Initialize reward and perfect execution tracker
    reward = 0.0
    was_perfect = True  # Start assuming perfect execution, set to False if any issues found

    # 0. FALL DETECTION: Massive penalty and episode termination if robot falls
    # Check if robot has fallen based on height and balance
    current_height = center_pos[2]

    # Get actual off_balance from track_orientation data
    if hasattr(track_orientation, 'last_off_balance'):
        current_balance = track_orientation.last_off_balance
    else:
        current_balance = 0.0

    # Robot has fallen if:
    # - Height is too low (lying on ground)
    # - Balance is too extreme (tipped over)
    has_fallen = current_balance > 90.0

    # 1. BALANCE REWARD: Reward near 0°, punish near 90°, ignore middle ground
    # Target: 0° (perfect balance), Bad: 90° (tipped over)
    # 33% tolerance: ±30° from 0° and ±30° from 90°

    if current_balance < 30.0:  # Within 33% of perfect balance (0° to 30°)
        # Reward being close to perfect balance
        if current_balance < 3.0:  # Within 10% of perfect (0° ± 3°)
            balance_reward = 1.0
            print(f"🎯 PERFECT BALANCE! +1.0 reward - Balance: {current_balance:.1f}°")
        else:  # Within 33% of perfect (3° to 30°)
            # Logarithmic scaling: 0.1 at 33% error, 1.0 at 10% error
            balance_progress = 1.0 - (current_balance / 30.0) ** 2
            balance_reward = 0.1 + 0.9 * balance_progress
            print(f"🎯 GOOD BALANCE: +{balance_reward:.2f} reward - Balance: {current_balance:.1f}°")
            was_perfect = False

    elif current_balance > 60.0:  # Within 33% of bad balance (60° to 90°)
        # Punish being close to tipped over
        if current_balance > 87.0:  # Within 10% of bad (87° to 90°)
            balance_reward = -1.0  # Maximum penalty for being tipped over
            print(f"❌ TIPPED OVER: {balance_reward:.1f} penalty - Balance: {current_balance:.1f}°")
            was_perfect = False
        else:  # Within 33% of bad (60° to 87°)
            # Logarithmic scaling: -0.1 at 33% from bad, -1.0 at 10% from bad
            balance_progress = 1.0 - ((90.0 - current_balance) / 30.0) ** 2
            balance_reward = -0.1 - 0.9 * balance_progress
            print(f"❌ POOR BALANCE: {balance_reward:.2f} penalty - Balance: {current_balance:.1f}°")
            was_perfect = False

    else:  # Middle ground (30° to 60°) - no reward, no penalty
        balance_reward = 0.0
        print(f"📊 MIDDLE BALANCE: No reward/penalty - Balance: {current_balance:.1f}°")
        was_perfect = False

    reward += balance_reward

    # 2. HEIGHT REWARD: Reward being at the correct height (0.129m), punish being too low
    # Target: 0.129m (perfect height), Bad: 0.0m (lying on ground)
    # 33% zones: Good (0.129m to 0.086m), Neutral (0.086m to 0.043m), Bad (0.043m to 0.0m)

    if current_height > 0.086:  # Within 33% of perfect height (0.086m to 0.129m)
        # Reward being close to perfect height
        if current_height > 0.126:  # Within 10% of perfect (0.126m to 0.129m)
            height_reward = 1.0
            print(f"�� PERFECT HEIGHT! +1.0 reward - Height: {current_height:.3f}m")
        else:  # Within 33% of perfect (0.086m to 0.126m)
            # Logarithmic scaling: 0.1 at 33% error, 1.0 at 10% error
            height_progress = 1.0 - ((0.129 - current_height) / 0.043) ** 2
            height_reward = 0.1 + 0.9 * height_progress
            print(f"🎯 GOOD HEIGHT: +{height_reward:.2f} reward - Height: {current_height:.3f}m")
            was_perfect = False

    elif current_height < 0.043:  # Within 33% of bad height (0.0m to 0.043m)
        # Punish being close to lying on ground
        if current_height < 0.003:  # Within 10% of bad (0.0m to 0.003m)
            height_reward = -1.0  # Maximum penalty for lying on ground
            print(f"❌ LYING ON GROUND: {height_reward:.1f} penalty - Height: {current_height:.3f}m")
            was_perfect = False
        else:  # Within 33% of bad (0.003m to 0.043m)
            # Logarithmic scaling: -0.1 at 33% from bad, -1.0 at 10% from bad
            height_progress = 1.0 - (current_height / 0.043) ** 2
            height_reward = -0.1 - 0.9 * height_progress
            print(f"❌ POOR HEIGHT: {height_reward:.2f} penalty - Height: {current_height:.3f}m")
            was_perfect = False

    else:  # Middle ground (0.043m to 0.086m) - no reward, no penalty
        height_reward = 0.0
        print(f"📊 MIDDLE HEIGHT: No reward/penalty - Height: {current_height:.3f}m")
        was_perfect = False

    reward += height_reward

    # 3. MOVEMENT REWARD: Reward moving in the correct direction, punish ALL wrong directions
    if commands:
        # Parse complex commands like 'w+a+arrowleft'
        command_list = commands.split('+') if isinstance(commands, str) else commands

        # Get movement data from the last call to track_orientation
        try:
            if hasattr(track_orientation, 'last_movement_data'):
                movement_data = track_orientation.last_movement_data

                # Check rotation first - must be minimal to get any movement reward
                rotation_during_movement = abs(track_orientation.last_rotation) if hasattr(track_orientation,
                                                                                           'last_rotation') else 0

                if rotation_during_movement < 2.0:  # Acceptable rotation for all directions
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
                                    print(
                                        f"�� PERFECT FORWARD: +{movement_reward:.1f} reward - Forward: {forward_movement:.3f}m")
                                else:  # Within 33% of target (3.33cm to 4.5cm)
                                    movement_progress = 1.0 - ((5.0 - forward_movement) / 1.67) ** 2
                                    movement_reward = 0.3 + 2.7 * movement_progress
                                    print(
                                        f"🎯 GOOD FORWARD: +{movement_reward:.2f} reward - Forward: {forward_movement:.3f}m")
                                    was_perfect = False
                            elif forward_movement < 0.0:  # Moving backward (wrong direction)
                                if forward_movement < -4.5:  # Within 10% of bad (-4.5cm to -5cm)
                                    movement_reward = -3.0
                                    print(
                                        f"❌ PERFECT BACKWARD: {movement_reward:.1f} penalty - Forward: {forward_movement:.3f}m")
                                    was_perfect = False
                                else:  # Within 33% of bad (0cm to -4.5cm)
                                    movement_progress = max(0.0, min(1.0, 1.0 - ((forward_movement + 5.0) / 4.5) ** 2))
                                    movement_reward = -0.3 - 2.7 * movement_progress
                                    print(
                                        f"❌ POOR FORWARD: {movement_reward:.2f} penalty - Forward: {forward_movement:.3f}m")
                                    was_perfect = False
                            else:  # Middle ground (0cm to 3.33cm) - no reward, no penalty
                                movement_reward = 0.0
                                print(f"�� MIDDLE FORWARD: No reward/penalty - Forward: {forward_movement:.3f}m")
                                was_perfect = False

                            reward += movement_reward

                            # PUNISH movement in wrong directions (even if not commanded)
                            if abs(left_movement) > 0.5:  # Moving left when should go forward
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(
                                    f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving LEFT: {left_movement:.3f}m when FORWARD commanded")
                                was_perfect = False

                            if abs(right_movement) > 0.5:  # Moving right when should go forward
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(
                                    f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving RIGHT: {right_movement:.3f}m when FORWARD commanded")
                                was_perfect = False

                            if abs(backward_movement) > 0.5:  # Moving backward when should go forward
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(
                                    f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving BACKWARD: {backward_movement:.3f}m when FORWARD commanded")
                                was_perfect = False

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
                                    print(
                                        f"🎯 PERFECT BACKWARD: +{movement_reward:.1f} reward - Backward: {backward_movement:.3f}m")
                                else:  # Within 33% of target (3.33cm to 4.5cm)
                                    movement_progress = 1.0 - ((5.0 - backward_movement) / 1.67) ** 2
                                    movement_reward = 0.3 + 2.7 * movement_progress
                                    print(
                                        f"🎯 GOOD BACKWARD: +{movement_reward:.2f} reward - Backward: {backward_movement:.3f}m")
                                    was_perfect = False
                            elif backward_movement < 0.0:  # Moving forward (wrong direction)
                                if backward_movement < -4.5:  # Within 10% of bad (-4.5cm to -5cm)
                                    movement_reward = -3.0
                                    print(
                                        f"❌ PERFECT FORWARD: {movement_reward:.1f} penalty - Backward: {backward_movement:.3f}m")
                                    was_perfect = False
                                else:  # Within 33% of bad (0cm to -4.5cm)
                                    movement_progress = max(0.0, min(1.0, 1.0 - ((backward_movement + 5.0) / 4.5) ** 2))
                                    movement_reward = -0.3 - 2.7 * movement_progress
                                    print(
                                        f"❌ POOR BACKWARD: {movement_reward:.2f} penalty - Backward: {backward_movement:.3f}m")
                                    was_perfect = False
                            else:  # Middle ground (0cm to 3.33cm) - no reward, no penalty
                                movement_reward = 0.0
                                print(f"📊 MIDDLE BACKWARD: No reward/penalty - Backward: {backward_movement:.3f}m")
                                was_perfect = False

                            reward += movement_reward

                            # PUNISH movement in wrong directions
                            if abs(forward_movement) > 0.5:  # Moving forward when should go backward
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(
                                    f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving FORWARD: {forward_movement:.3f}m when BACKWARD commanded")
                                was_perfect = False

                            if abs(left_movement) > 0.5:  # Moving left when should go backward
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(
                                    f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving LEFT: {left_movement:.3f}m when BACKWARD commanded")
                                was_perfect = False

                            if abs(right_movement) > 0.5:  # Moving right when should go backward
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(
                                    f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving RIGHT: {right_movement:.3f}m when BACKWARD commanded")
                                was_perfect = False

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
                                    print(f"🎯 PERFECT LEFT: +{movement_reward:.1f} reward - Left: {left_movement:.3f}m")
                                else:  # Within 33% of target (3.33cm to 4.5cm)
                                    movement_progress = 1.0 - ((5.0 - left_movement) / 1.67) ** 2
                                    movement_reward = 0.3 + 2.7 * movement_progress
                                    print(f"🎯 GOOD LEFT: +{movement_reward:.2f} reward - Left: {left_movement:.3f}m")
                                    was_perfect = False
                            elif left_movement < 0.0:  # Moving right (wrong direction)
                                if left_movement < -4.5:  # Within 10% of bad (-4.5cm to -5cm)
                                    movement_reward = -3.0
                                    print(
                                        f"❌ PERFECT RIGHT: {movement_reward:.1f} penalty - Left: {left_movement:.3f}m")
                                    was_perfect = False
                                else:  # Within 33% of bad (0cm to -4.5cm)
                                    movement_progress = max(0.0, min(1.0, 1.0 - ((left_movement + 5.0) / 4.5) ** 2))
                                    movement_reward = -0.3 - 2.7 * movement_progress
                                    print(f"❌ POOR LEFT: {movement_reward:.2f} penalty - Left: {left_movement:.3f}m")
                                    was_perfect = False
                            else:  # Middle ground (0cm to 3.33cm) - no reward, no penalty
                                movement_reward = 0.0
                                print(f"📊 MIDDLE LEFT: No reward/penalty - Left: {left_movement:.3f}m")
                                was_perfect = False

                            reward += movement_reward

                            # PUNISH movement in wrong directions
                            if abs(forward_movement) > 0.5:  # Moving forward when should go left
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(
                                    f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving FORWARD: {forward_movement:.3f}m when LEFT commanded")
                                was_perfect = False

                            if abs(backward_movement) > 0.5:  # Moving backward when should go left
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(
                                    f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving BACKWARD: {backward_movement:.3f}m when LEFT commanded")
                                was_perfect = False

                            if abs(right_movement) > 0.5:  # Moving right when should go left
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(
                                    f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving RIGHT: {right_movement:.3f}m when LEFT commanded")
                                was_perfect = False

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
                                    print(
                                        f"🎯 PERFECT RIGHT: +{movement_reward:.1f} reward - Right: {right_movement:.3f}m")
                                else:  # Within 33% of target (3.33cm to 4.5cm)
                                    movement_progress = 1.0 - ((5.0 - right_movement) / 1.67) ** 2
                                    movement_reward = 0.3 + 2.7 * movement_progress
                                    print(f"🎯 GOOD RIGHT: +{movement_reward:.2f} reward - Right: {right_movement:.3f}m")
                                    was_perfect = False
                            elif right_movement < 0.0:  # Moving left (wrong direction)
                                if right_movement < -4.5:  # Within 10% of bad (-4.5cm to -5cm)
                                    movement_reward = -3.0
                                    print(
                                        f"❌ PERFECT LEFT: {movement_reward:.1f} penalty - Right: {right_movement:.3f}m")
                                    was_perfect = False
                                else:  # Within 33% of bad (0cm to -4.5cm)
                                    movement_progress = max(0.0, min(1.0, 1.0 - ((right_movement + 5.0) / 4.5) ** 2))
                                    movement_reward = -0.3 - 2.7 * movement_progress
                                    print(f"❌ POOR RIGHT: {movement_reward:.2f} penalty - Right: {right_movement:.3f}m")
                                    was_perfect = False
                            else:  # Middle ground (0cm to 3.33cm) - no reward, no penalty
                                movement_reward = 0.0
                                print(f"📊 MIDDLE RIGHT: No reward/penalty - Right: {right_movement:.3f}m")
                                was_perfect = False

                            reward += movement_reward

                            # PUNISH movement in wrong directions
                            if abs(forward_movement) > 0.5:  # Moving forward when should go right
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(
                                    f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving FORWARD: {forward_movement:.3f}m when RIGHT commanded")
                                was_perfect = False

                            if abs(backward_movement) > 0.5:  # Moving backward when should go right
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(
                                    f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving BACKWARD: {backward_movement:.3f}m when RIGHT commanded")
                                was_perfect = False

                            if abs(left_movement) > 0.5:  # Moving left when should go right
                                wrong_direction_penalty = -1.5
                                reward += wrong_direction_penalty
                                print(
                                    f"❌ WRONG DIRECTION: {wrong_direction_penalty:.1f} penalty - Moving LEFT: {left_movement:.3f}m when RIGHT commanded")
                                was_perfect = False

                        elif cmd in ['arrowleft', 'arrowright']:  # Rotation commands - handled in rotation section
                            pass  # Skip here, handled in rotation reward section

                else:  # Too much rotation during movement
                    reward -= 0.1
                    print(f"❌ EXCESSIVE ROTATION: -0.1 penalty - Rotation: {rotation_during_movement:.1f}°")
                    was_perfect = False

        except Exception as e:
            # If movement analysis fails, give neutral reward
            reward += 0.0

    # 4. ROTATION CONTROL REWARD: Reward maintaining stable orientation or executing rotation commands
    if hasattr(track_orientation, 'last_rotation'):
        rotation_magnitude = abs(track_orientation.last_rotation)
        rotation_commanded = any(cmd in ['arrowleft', 'arrowright'] for cmd in command_list)

        if not rotation_commanded:
            # NO ROTATION COMMANDED: Reward staying stable (minimal rotation)
            # Target: 0° rotation (perfect stability), Bad: 30° rotation (excessive rotation)
            # 33% zones: Good (0° to 10°), Neutral (10° to 20°), Bad (20° to 30°)

            if rotation_magnitude < 10.0:  # Within 33% of perfect stability (0° to 10°)
                # Reward being close to perfect stability
                if rotation_magnitude < 1.0:  # Within 10% of perfect (0° to 1°)
                    rotation_reward = 2.0  # Perfect stability
                    print(f"�� PERFECT STABILITY: +1.0 reward - Rotation: {rotation_magnitude:.1f}°")
                else:  # Within 33% of perfect (1° to 10°)
                    # Logarithmic scaling: 0.2 at 33% error, 2.0 at 10% error
                    rotation_progress = 1.0 - (rotation_magnitude / 10.0) ** 2
                    rotation_reward = 0.2 + 1.8 * rotation_progress
                    print(f"�� GOOD STABILITY: +{rotation_reward:.2f} reward - Rotation: {rotation_magnitude:.1f}°")
                    was_perfect = False

            elif rotation_magnitude > 20.0:  # Within 33% of bad rotation (20° to 30°)
                # Punish being close to excessive rotation
                if rotation_magnitude > 29.0:  # Within 10% of bad (29° to 30°)
                    rotation_reward = -2.0  # Maximum penalty for excessive rotation
                    print(f"❌ EXCESSIVE ROTATION: {rotation_reward:.1f} penalty - Rotation: {rotation_magnitude:.1f}°")
                    was_perfect = False
                else:  # Within 33% of bad (20° to 29°)
                    # Logarithmic scaling: -0.2 at 33% from bad, -2.0 at 10% from bad
                    rotation_progress = 1.0 - ((30.0 - rotation_magnitude) / 10.0) ** 2
                    rotation_reward = -0.2 - 1.8 * rotation_progress
                    print(f"❌ POOR STABILITY: {rotation_reward:.2f} penalty - Rotation: {rotation_magnitude:.1f}°")
                    was_perfect = False

            else:  # Middle ground (10° to 20°) - no reward, no penalty
                rotation_reward = 0.0
                print(f"📊 MIDDLE STABILITY: No reward/penalty - Rotation: {rotation_magnitude:.1f}°")
                was_perfect = False

            reward += rotation_reward

        else:
            # ROTATION COMMANDED: Reward executing rotation commands properly
            # Target: 30° rotation (rapid rotation), Bad: 0° rotation (no rotation)
            # 33% zones: Good (20° to 30°), Neutral (10° to 20°), Bad (0° to 10°)

            if rotation_magnitude > 20.0:  # Within 33% of target rotation (20° to 30°)
                # Reward being close to target rotation
                if rotation_magnitude > 29.0:  # Within 10% of target (29° to 30°)
                    rotation_reward = 2.0  # Perfect rotation execution
                    print(f"🎯 PERFECT ROTATION: +1.0 reward - Rotation: {rotation_magnitude:.1f}°")
                else:  # Within 33% of target (20° to 29°)
                    # Logarithmic scaling: 0.2 at 33% error, 2.0 at 10% error
                    rotation_progress = 1.0 - ((30.0 - rotation_magnitude) / 10.0) ** 2
                    rotation_reward = 0.2 + 1.8 * rotation_progress
                    print(f"🎯 GOOD ROTATION: +{rotation_reward:.2f} reward - Rotation: {rotation_magnitude:.1f}°")
                    was_perfect = False

            elif rotation_magnitude < 10.0:  # Within 33% of bad rotation (0° to 10°)
                # Punish being close to no rotation
                if rotation_magnitude < 1.0:  # Within 10% of bad (0° to 1°)
                    rotation_reward = -2.0  # Maximum penalty for no rotation
                    print(f"❌ NO ROTATION: {rotation_reward:.1f} penalty - Rotation: {rotation_magnitude:.1f}°")
                    was_perfect = False
                else:  # Within 33% of bad (1° to 10°)
                    # Logarithmic scaling: -0.2 at 33% from bad, -2.0 at 10% from bad
                    rotation_progress = 1.0 - (rotation_magnitude / 10.0) ** 2
                    rotation_reward = -0.2 - 1.8 * rotation_progress
                    print(f"❌ POOR ROTATION: {rotation_reward:.2f} penalty - Rotation: {rotation_magnitude:.1f}°")
                    was_perfect = False

            else:  # Middle ground (10° to 20°) - no reward, no penalty
                rotation_reward = 0.0
                print(f"📊 MIDDLE ROTATION: No reward/penalty - Rotation: {rotation_magnitude:.1f}°")
                was_perfect = False

            reward += rotation_reward

    # 5. PERFECT PERFORMANCE BONUS: Massive reward for doing exactly what we want
    # Simple boolean approach: was_perfect starts True, gets set False if any issues found

    if was_perfect and commands:  # Only give bonus if we had commands and were perfect
        perfect_bonus = 10.0  # Much bigger than individual movement rewards
        reward += perfect_bonus
        print(f"🏆 PERFECT EXECUTION! +{perfect_bonus:.1f} MASSIVE BONUS - All commands executed flawlessly!")
    elif commands:
        print(f"📊 Good execution, but not perfect - no bonus this time")

    if has_fallen:
        # MASSIVE PENALTY: Standard -100 penalty for falling, episode ends
        fall_penalty = -100
        print(f"EPISODE FAILURE -100 points (robot fell over)")

        # Signal that episode should end due to falling
        global episode_step
        episode_step = TRAINING_CONFIG['max_steps_per_episode']  # Force episode end

        return fall_penalty

    # 5. CLAMP REWARD: Prevent extreme values (but allow fall penalty)
    elif not has_fallen:
        reward = max(-1.0, min(1.0, reward))

    return reward
