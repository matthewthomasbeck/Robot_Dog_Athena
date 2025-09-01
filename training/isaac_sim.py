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





###################################################
############### ISAAC SIM FUNCTIONS ###############
###################################################


########## GENERATE GRID POSITIONS ##########

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


########## ISAAC SIM JOINT MAPPING ##########

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
