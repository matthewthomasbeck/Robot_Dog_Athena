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

import numpy

##### import config #####

import utilities.config as config

##### import isaac sim libraries #####

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.pose import get_world_pose
from pxr import Gf





###################################################
############### ISAAC SIM FUNCTIONS ###############
###################################################

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


########## REWARD FUNCTION ##########

def compute_reward(robot_prim_path, previous_pose, current_pose, command, intensity):

    # position delta
    prev_pos = numpy.array(previous_pose[0])  # (x, y, z)
    curr_pos = numpy.array(current_pose[0])
    delta_pos = curr_pos - prev_pos

    # orientation delta
    prev_rot = Gf.Quatf(*previous_pose[1])  # (w, x, y, z)
    curr_rot = Gf.Quatf(*current_pose[1])
    delta_rot = curr_rot * prev_rot.GetInverse()
    delta_yaw = delta_rot.GetAxisAngle()[0] * delta_rot.GetAxisAngle()[1][2]  # yaw only

    reward = 0.0

    # cardinal direction rewards
    if 'w' in command:
        reward += delta_pos[0] * intensity  # +X is forward
    if 's' in command:
        reward -= delta_pos[0] * intensity
    if 'a' in command:
        reward += delta_pos[1] * intensity  # +Y is left
    if 'd' in command:
        reward -= delta_pos[1] * intensity

    # rotation-based reward
    if 'arrowleft' in command:
        reward += delta_yaw * intensity
    if 'arrowright' in command:
        reward -= delta_yaw * intensity

    # penalty for undesired movement or instability
    noise_penalty = numpy.linalg.norm(delta_pos[:2]) * 0.05 if not command else 0
    reward -= noise_penalty

    return reward
