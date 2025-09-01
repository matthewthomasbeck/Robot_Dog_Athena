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

import numpy as np
import logging

##### import necessary functions #####

from scipy.spatial.transform import Rotation





#####################################################
############### ORIENTATION FUNCTIONS ###############
#####################################################


########## ORIENTATION FUNCTION ##########

def track_orientation(robot_idx=0):
    try:
        # Get the specific robot's data
        if robot_idx >= len(config.ISAAC_ROBOTS):
            logging.error(f"(orientation.py): Robot {robot_idx} not found, using robot 0")
            robot_idx = 0
            
        positions, rotations = config.ISAAC_ROBOTS[robot_idx].get_world_poses()
        print(f"(orientation.py): Positions: {positions}")
        print(f"(orientation.py): Rotations: {rotations}")
        center_pos = positions[0]
        rotation = rotations[0]
        quat_wxyz = [rotation[0], rotation[1], rotation[2], rotation[3]]
        r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        euler_angles = r.as_euler('xyz', degrees=True)
        yaw_deg = euler_angles[2]  # Yaw is rotation around Z-axis

        # Robot spawns facing -Y (forward), so we need to calculate relative rotation
        # -Y direction = 0 degrees
        # Left rotation = positive degrees (10° = facing slightly left of -Y)
        # Right rotation = negative degrees, but we convert to 360° system

        # Convert to 0-360 degree system where 0° = facing -Y (forward)
        if yaw_deg < 0:
            facing_deg = 360 + yaw_deg
        else:
            facing_deg = yaw_deg

        # Calculate off-balance (combined pitch and roll deviation from vertical)
        roll_deg = abs(euler_angles[0])  # Roll around X-axis
        pitch_deg = abs(euler_angles[1])  # Pitch around Y-axis

        # Combined off-balance is the sum of roll and pitch deviations from vertical
        # Perfectly upright = 0°, maximum tilt = 180° (though robot would fall before then)
        off_balance = roll_deg + pitch_deg

        # Height off the ground (Z coordinate)
        height = center_pos[2]

        # Store current facing direction for reward function (even on first call)
        # Use robot-specific storage
        if not hasattr(track_orientation, 'robot_data'):
            track_orientation.robot_data = {}
        
        if robot_idx not in track_orientation.robot_data:
            track_orientation.robot_data[robot_idx] = {}
            
        track_orientation.robot_data[robot_idx]['last_facing_deg'] = facing_deg

        # Calculate current directions relative to robot's current facing (WASD format)
        # These change as the robot rotates
        curr_w = facing_deg  # Current forward direction (W key)
        curr_s = (facing_deg + 180) % 360  # Opposite of forward (S key)
        curr_a = (facing_deg + 90) % 360  # 90° left of forward (A key)
        curr_d = (facing_deg + 270) % 360  # 90° right of forward (D key)

        # Check if position is changing (robot is actually moving)
        if 'last_position' not in track_orientation.robot_data[robot_idx]:
            track_orientation.robot_data[robot_idx]['last_position'] = center_pos
            track_orientation.robot_data[robot_idx]['static_count'] = 0
        else:
            # Calculate horizontal distance moved (ignore Z-axis bouncing)
            dx = center_pos[0] - track_orientation.robot_data[robot_idx]['last_position'][0]  # X movement
            dy = center_pos[1] - track_orientation.robot_data[robot_idx]['last_position'][1]  # Y movement

            horizontal_distance = np.sqrt(dx ** 2 + dy ** 2)

            if horizontal_distance < 0.001:  # Less than 1mm horizontal movement
                track_orientation.robot_data[robot_idx]['static_count'] += 1
                if track_orientation.robot_data[robot_idx]['static_count'] > 10:
                    logging.warning(f"   ⚠️  Robot {robot_idx} hasn't moved horizontally in {track_orientation.robot_data[robot_idx]['static_count']} steps!")

                # Store current facing direction even when not moving (for reward function)
                track_orientation.robot_data[robot_idx]['last_facing_deg'] = facing_deg
            else:
                track_orientation.robot_data[robot_idx]['static_count'] = 0

                # Calculate directional movement components relative to robot's current facing (WASD format)
                facing_rad = np.radians(facing_deg)

                # Project movement onto robot's current WASD directions
                # W = forward, S = backward, A = left, D = right
                w_movement = -dy * np.cos(facing_rad) - dx * np.sin(facing_rad)  # W direction (forward)
                s_movement = -w_movement  # S direction (backward) - opposite of W
                a_movement = -dx * np.cos(facing_rad) + dy * np.sin(facing_rad)  # A direction (left)
                d_movement = -a_movement  # D direction (right) - opposite of A

                # Determine movement direction string (WASD format)
                movement_direction = ""
                if abs(w_movement) > 0.001:
                    if w_movement > 0:
                        movement_direction += "w"
                    else:
                        movement_direction += "s"

                if abs(a_movement) > 0.001:
                    if a_movement > 0:
                        movement_direction += "a"
                    else:
                        movement_direction += "d"

                if not movement_direction:
                    movement_direction = "n"  # No significant movement

                # Calculate rotation (change in facing direction)
                if 'last_facing' not in track_orientation.robot_data[robot_idx]:
                    track_orientation.robot_data[robot_idx]['last_facing'] = facing_deg
                    rotation_deg = 0.0
                else:
                    # Calculate rotation as change in facing direction
                    rotation_change = facing_deg - track_orientation.robot_data[robot_idx]['last_facing']

                    # Handle wrapping around 360° boundary
                    if rotation_change > 180:
                        rotation_change -= 360
                    elif rotation_change < -180:
                        rotation_change += 360

                    rotation_deg = rotation_change
                    track_orientation.robot_data[robot_idx]['last_facing'] = facing_deg

                # Store movement data for reward function to access
                track_orientation.robot_data[robot_idx]['last_movement_data'] = {
                    'w': w_movement,
                    's': s_movement,
                    'a': a_movement,
                    'd': d_movement,
                    'movement_direction': movement_direction,
                    'horizontal_distance': horizontal_distance
                }
                track_orientation.robot_data[robot_idx]['last_rotation'] = rotation_deg
                track_orientation.robot_data[robot_idx]['last_off_balance'] = off_balance
                track_orientation.robot_data[robot_idx]['last_facing_deg'] = facing_deg  # Store current facing direction for strict movement detection

            track_orientation.robot_data[robot_idx]['last_position'] = center_pos

        return center_pos, facing_deg

    except Exception as e:
        logging.error(f"❌ Failed to track orientation for robot {robot_idx}: {e}")
        return None, None
