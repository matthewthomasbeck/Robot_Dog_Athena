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

import math
import random


########## CREATE DEPENDENCIES ##########

##### global variables #####

PREVIOUS_COMMAND = None
PREVIOUS_INTENSITY = None





################################################
############### ISAAC SIM SET UP ###############
################################################


########## ISAAC SIM FUNCTIONS ##########

##### generate grid positions for each robot #####

def generate_grid_positions(num_robots, spacing=2.0, start_z=0.14):
    """
    Generate robot positions in a grid pattern centered at origin.
    
    Args:
        num_robots (int): Number of robots to position
        spacing (float): Distance between robots in meters (default: 2.0)
        start_z (float): Z-coordinate for all robots (default: 0.14)
    
    Returns:
        list: List of (x, y, z) tuples for robot positions
        
    Example:
        For 5 robots with spacing 2.0:
        Robot 0: (-2.0, -2.0, 0.14)  Robot 1: (0.0, -2.0, 0.14)  Robot 2: (2.0, -2.0, 0.14)
        Robot 3: (-2.0,  0.0, 0.14)  Robot 4: (0.0,  0.0, 0.14)
    """
    positions = []
    
    # Calculate grid dimensions (square-ish grid)
    grid_size = int(math.ceil(math.sqrt(num_robots)))
    
    # Calculate starting offset to center the grid
    start_offset = -(grid_size - 1) * spacing / 2
    
    for i in range(num_robots):
        row = i // grid_size
        col = i % grid_size
        x = start_offset + col * spacing
        y = start_offset + row * spacing
        positions.append((x, y, start_z))
    
    return positions

##### set up universal joint index map #####

def build_isaac_joint_index_map(dof_names):

    alias_to_actual = {}

    for name in dof_names:
        parts = name.split("_")
        if len(parts) >= 2:
            leg_id = parts[0]
            if parts[1] in {"hip", "upper", "lower"}:
                joint_type = parts[1]
            elif "femur" in name:
                joint_type = "upper"
            elif "shin" in name:
                joint_type = "lower"
            else:
                continue
            alias = f"{leg_id}_{joint_type}"
            if alias not in alias_to_actual:
                alias_to_actual[alias] = name  # first valid mapping

    joint_index_map = {}
    for alias, actual_name in alias_to_actual.items():
        if actual_name in dof_names:
            joint_index_map[alias] = dof_names.index(actual_name)

    return joint_index_map

##### random commands #####

def get_random_command(phase=1): # returns semirandom, realistic command combinations based on phase
    global PREVIOUS_COMMAND

    # Phase 1: Basic forward movement and turning only
    if phase == 1:
        command_combinations = [
            # Single movements only
            'w', 'arrowleft', 'arrowright'
        ]
        
        # Simple transition weights for phase 1
        command_weights = {
            'w': {
                'w': 0.6, 'arrowleft': 0.2, 'arrowright': 0.2
            },
            'arrowleft': {
                'w': 0.4, 'arrowleft': 0.4, 'arrowright': 0.2
            },
            'arrowright': {
                'w': 0.4, 'arrowleft': 0.2, 'arrowright': 0.4
            }
        }
    
    # Phase 2: Full movement + rotation combinations
    elif phase == 2:
        command_combinations = [
            # Single movements
            'w', 's', 'a', 'd',
            # Movement + rotation
            'w+arrowleft', 'w+arrowright', 's+arrowleft', 's+arrowright',
            'a+arrowleft', 'a+arrowright', 'd+arrowleft', 'd+arrowright'
        ]
        
        # Transition weights for phase 2
        command_weights = {
            # Single forward movement
            'w': {
                'w': 0.3, 'w+arrowleft': 0.2, 'w+arrowright': 0.2, 's': 0.1, 'a': 0.1, 'd': 0.1
            },
            # Single backward movement  
            's': {
                's': 0.3, 's+arrowleft': 0.2, 's+arrowright': 0.2, 'w': 0.1, 'a': 0.1, 'd': 0.1
            },
            # Single left movement
            'a': {
                'a': 0.3, 'a+arrowleft': 0.2, 'a+arrowright': 0.2, 'w': 0.1, 's': 0.1, 'd': 0.1
            },
            # Single right movement
            'd': {
                'd': 0.3, 'd+arrowleft': 0.2, 'd+arrowright': 0.2, 'w': 0.1, 's': 0.1, 'a': 0.1
            },
            # Movement + rotation combinations
            'w+arrowleft': {'w': 0.3, 'w+arrowleft': 0.3, 'arrowleft': 0.2, 'a': 0.1, 's': 0.1},
            'w+arrowright': {'w': 0.3, 'w+arrowright': 0.3, 'arrowright': 0.2, 'd': 0.1, 's': 0.1},
            's+arrowleft': {'s': 0.3, 's+arrowleft': 0.3, 'arrowleft': 0.2, 'a': 0.1, 'w': 0.1},
            's+arrowright': {'s': 0.3, 's+arrowright': 0.3, 'arrowright': 0.2, 'd': 0.1, 'w': 0.1},
            'a+arrowleft': {'a': 0.3, 'a+arrowleft': 0.3, 'arrowleft': 0.2, 'w': 0.1, 's': 0.1},
            'a+arrowright': {'a': 0.3, 'a+arrowright': 0.3, 'arrowright': 0.2, 'w': 0.1, 's': 0.1},
            'd+arrowleft': {'d': 0.3, 'd+arrowleft': 0.3, 'arrowleft': 0.2, 'w': 0.1, 's': 0.1},
            'd+arrowright': {'d': 0.3, 'd+arrowright': 0.3, 'arrowright': 0.2, 'w': 0.1, 's': 0.1}
        }
    
    # Phase 3: Full complexity with diagonals and complex combinations
    elif phase == 3:
        command_combinations = [
            # Single movements
            'w', 's', 'a', 'd',
            # Diagonals
            'w+a', 'w+d', 's+a', 's+d',
            # Movement + rotation
            'w+arrowleft', 'w+arrowright', 's+arrowleft', 's+arrowright',
            'a+arrowleft', 'a+arrowright', 'd+arrowleft', 'd+arrowright',
            # Complex combinations (diagonal + rotation)
            'w+a+arrowleft', 'w+a+arrowright', 'w+d+arrowleft', 'w+d+arrowright',
            's+a+arrowleft', 's+a+arrowright', 's+d+arrowleft', 's+d+arrowright'
        ]
        
        # Full transition weights for phase 3
        command_weights = {
            # Single forward movement
            'w': {
                'w': 0.3, 'w+a': 0.15, 'w+d': 0.15, 'w+arrowleft': 0.1, 'w+arrowright': 0.1,
                'a': 0.05, 'd': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Single backward movement  
            's': {
                's': 0.3, 's+a': 0.15, 's+d': 0.15, 's+arrowleft': 0.1, 's+arrowright': 0.1,
                'a': 0.05, 'd': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Single left movement
            'a': {
                'a': 0.3, 'w+a': 0.15, 's+a': 0.15, 'a+arrowleft': 0.1, 'a+arrowright': 0.1,
                'w': 0.05, 's': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Single right movement
            'd': {
                'd': 0.3, 'w+d': 0.15, 's+d': 0.15, 'd+arrowleft': 0.1, 'd+arrowright': 0.1,
                'w': 0.05, 's': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Single left rotation
            'arrowleft': {
                'arrowleft': 0.25, 'w+arrowleft': 0.15, 's+arrowleft': 0.15, 'a+arrowleft': 0.1, 'd+arrowleft': 0.1,
                'w': 0.05, 's': 0.05, 'a': 0.05, 'd': 0.05, 'arrowright': 0.05
            },
            # Single right rotation
            'arrowright': {
                'arrowright': 0.25, 'w+arrowright': 0.15, 's+arrowright': 0.15, 'a+arrowright': 0.1, 'd+arrowright': 0.1,
                'w': 0.05, 's': 0.05, 'a': 0.05, 'd': 0.05, 'arrowleft': 0.05
            },
            # Diagonal forward-left
            'w+a': {
                'w+a': 0.25, 'w': 0.15, 'a': 0.15, 'w+a+arrowleft': 0.1, 'w+a+arrowright': 0.1,
                'w+arrowleft': 0.05, 'w+arrowright': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Diagonal forward-right
            'w+d': {
                'w+d': 0.25, 'w': 0.15, 'd': 0.15, 'w+d+arrowleft': 0.1, 'w+d+arrowright': 0.1,
                'w+arrowleft': 0.05, 'w+arrowright': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Diagonal backward-left
            's+a': {
                's+a': 0.25, 's': 0.15, 'a': 0.15, 's+a+arrowleft': 0.1, 's+a+arrowright': 0.1,
                's+arrowleft': 0.05, 's+arrowright': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Diagonal backward-right
            's+d': {
                's+d': 0.25, 's': 0.15, 'd': 0.15, 's+d+arrowleft': 0.1, 's+d+arrowright': 0.1,
                's+arrowleft': 0.05, 's+arrowright': 0.05, 'arrowleft': 0.05, 'arrowright': 0.05
            },
            # Movement + rotation combinations
            'w+arrowleft': {'w': 0.2, 'w+a': 0.15, 'w+arrowleft': 0.2, 'w+a+arrowleft': 0.1, 'a': 0.1, 'arrowleft': 0.1, 'w+d': 0.05, 'w+arrowright': 0.05, 's': 0.05},
            'w+arrowright': {'w': 0.2, 'w+d': 0.15, 'w+arrowright': 0.2, 'w+d+arrowright': 0.1, 'd': 0.1, 'arrowright': 0.1, 'w+a': 0.05, 'w+arrowleft': 0.05, 's': 0.05},
            's+arrowleft': {'s': 0.2, 's+a': 0.15, 's+arrowleft': 0.2, 's+a+arrowleft': 0.1, 'a': 0.1, 'arrowleft': 0.1, 's+d': 0.05, 's+arrowright': 0.05, 'w': 0.05},
            's+arrowright': {'s': 0.2, 's+d': 0.15, 's+arrowright': 0.2, 's+d+arrowright': 0.1, 'd': 0.1, 'arrowright': 0.1, 's+a': 0.05, 's+arrowleft': 0.05, 'w': 0.05},
            'a+arrowleft': {'a': 0.2, 'w+a': 0.15, 'a+arrowleft': 0.2, 'w+a+arrowleft': 0.1, 'w': 0.1, 'arrowleft': 0.1, 's+a': 0.05, 'a+arrowright': 0.05, 's': 0.05},
            'a+arrowright': {'a': 0.2, 's+a': 0.15, 'a+arrowright': 0.2, 's+a+arrowright': 0.1, 's': 0.1, 'arrowright': 0.1, 'w+a': 0.05, 'a+arrowleft': 0.05, 'w': 0.05},
            'd+arrowleft': {'d': 0.2, 's+d': 0.15, 'd+arrowleft': 0.2, 's+d+arrowleft': 0.1, 's': 0.1, 'arrowleft': 0.1, 'w+d': 0.05, 'd+arrowright': 0.05, 'w': 0.05},
            'd+arrowright': {'d': 0.2, 'w+d': 0.15, 'd+arrowright': 0.2, 'w+d+arrowright': 0.1, 'w': 0.1, 'arrowright': 0.1, 's+d': 0.05, 'd+arrowleft': 0.05, 's': 0.05},
            # Complex combinations
            'w+a+arrowleft': {'w+a': 0.2, 'w+a+arrowleft': 0.25, 'w': 0.15, 'a': 0.1, 'arrowleft': 0.1, 'w+arrowleft': 0.1, 'w+arrowright': 0.05, 's': 0.05},
            'w+a+arrowright': {'w+a': 0.2, 'w+a+arrowright': 0.25, 'w': 0.15, 'a': 0.1, 'arrowright': 0.1, 'w+arrowright': 0.1, 'w+arrowleft': 0.05, 's': 0.05},
            'w+d+arrowleft': {'w+d': 0.2, 'w+d+arrowleft': 0.25, 'w': 0.15, 'd': 0.1, 'arrowleft': 0.1, 'w+arrowleft': 0.1, 'w+arrowright': 0.05, 's': 0.05},
            'w+d+arrowright': {'w+d': 0.2, 'w+d+arrowright': 0.25, 'w': 0.15, 'd': 0.1, 'arrowright': 0.1, 'w+arrowright': 0.1, 'w+arrowleft': 0.05, 's': 0.05},
            's+a+arrowleft': {'s+a': 0.2, 's+a+arrowleft': 0.25, 's': 0.15, 'a': 0.1, 'arrowleft': 0.1, 's+arrowleft': 0.1, 's+arrowright': 0.05, 'w': 0.05},
            's+a+arrowright': {'s+a': 0.2, 's+a+arrowright': 0.25, 's': 0.15, 'a': 0.1, 'arrowright': 0.1, 's+arrowright': 0.1, 's+arrowleft': 0.05, 'w': 0.05},
            's+d+arrowleft': {'s+d': 0.2, 's+d+arrowleft': 0.25, 's': 0.15, 'd': 0.1, 'arrowleft': 0.1, 's+arrowleft': 0.1, 's+arrowright': 0.05, 'w': 0.05},
            's+d+arrowright': {'s+d': 0.2, 's+d+arrowright': 0.25, 's': 0.15, 'd': 0.1, 'arrowright': 0.1, 's+arrowright': 0.1, 's+arrowleft': 0.05, 'w': 0.05}
        }
    
    # Default fallback to phase 1
    else:
        command_combinations = ['w', 'arrowleft', 'arrowright']
        command_weights = {
            'w': {'w': 0.6, 'arrowleft': 0.2, 'arrowright': 0.2},
            'arrowleft': {'w': 0.4, 'arrowleft': 0.4, 'arrowright': 0.2},
            'arrowright': {'w': 0.4, 'arrowleft': 0.2, 'arrowright': 0.4}
        }
    
    # If this is the first command, choose randomly from current phase options
    if PREVIOUS_COMMAND is None:
        command = random.choice(command_combinations)
        PREVIOUS_COMMAND = command
        return command
    
    # Get weights for current previous command
    if PREVIOUS_COMMAND in command_weights:
        weights = command_weights[PREVIOUS_COMMAND]
    else:
        # Fallback: if previous command not in weights, choose randomly
        command = random.choice(command_combinations)
        PREVIOUS_COMMAND = command
        return command
    
    # Convert weights to list for random.choices
    commands = list(weights.keys())
    weight_values = list(weights.values())
    
    # Choose command based on weighted probabilities
    command = random.choices(commands, weights=weight_values, k=1)[0]
    
    # Update previous command
    PREVIOUS_COMMAND = command
    
    return command

##### random intensity #####

def get_random_intensity(phase=1): # returns intensity based on phase

    global PREVIOUS_INTENSITY

    # Phase 1 & 2: Intensity locked at 10 for stability
    if phase in [1, 2]:
        return 10
    
    # Phase 3: Full intensity range 1-10 with realistic transitions
    elif phase == 3:
        # If this is the first intensity, start with moderate
        if PREVIOUS_INTENSITY is None:
            intensity = random.choice([4, 5, 6, 7])
            PREVIOUS_INTENSITY = intensity
            return intensity
        
        # Define intensity change probabilities for realistic movement
        # Higher chance of staying close to previous intensity, with gradual changes
        intensity_change = random.choices(
            [-3, -2, -1, 0, 1, 2, 3],  # Possible changes
            weights=[0.05, 0.1, 0.25, 0.2, 0.25, 0.1, 0.05],  # Weights favoring small changes
            k=1
        )[0]
        
        # Calculate new intensity
        new_intensity = PREVIOUS_INTENSITY + intensity_change
        
        # Clamp to valid range 1-10
        new_intensity = max(1, min(10, new_intensity))
        
        # Update previous intensity
        PREVIOUS_INTENSITY = new_intensity
        
        return new_intensity
    
    # Default fallback to phase 1 (intensity 10)
    else:
        return 10
  