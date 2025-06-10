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

import math # import math library for calculations





##########################################
############### KINEMATICS ###############
##########################################


########## INVERSE KINEMATICS ##########

class Kinematics:
    def __init__(self, link_config):
        self.hip_offset = link_config['HIP_OFFSET']
        self.hip_to_leg_plane = link_config['HIP_TO_LEG_PLANE']
        self.femur = link_config['FEMUR_LENGTH']
        self.tibia = link_config['TIBIA_LENGTH']

    def inverse_kinematics(self, x, y, z):
        """
        Compute joint angles needed to place the foot at (x, y, z)
        relative to the leg base frame (hip joint at origin).
        Returns angles in degrees: (hip_abduction_angle, upper_angle, lower_angle)
        """

        # Hip abduction (rotation around Z) from Y offset
        hip_angle = math.degrees(math.atan2(y, math.sqrt(x**2 + z**2)))

        # Project leg into sagittal plane (XZ), accounting for hip linkage offset
        hip_plane_offset = self.hip_to_leg_plane
        x_leg = x - hip_plane_offset  # adjust X
        leg_plane_dist = math.sqrt(x_leg**2 + z**2)

        # Clamp leg distance to reachable range
        min_reach = abs(self.femur - self.tibia)
        max_reach = self.femur + self.tibia
        leg_plane_dist = max(min_reach, min(max_reach, leg_plane_dist))

        # Law of Cosines for knee angle (theta3)
        cos_theta3 = (self.femur**2 + self.tibia**2 - leg_plane_dist**2) / (2 * self.femur * self.tibia)
        theta3 = math.acos(cos_theta3)  # radians

        # Law of Cosines for shoulder angle (theta2)
        cos_theta2 = (self.femur**2 + leg_plane_dist**2 - self.tibia**2) / (2 * self.femur * leg_plane_dist)
        theta2_offset = math.acos(cos_theta2)  # radians
        shoulder_to_foot_angle = math.atan2(-z, x_leg)  # radians

        theta2 = shoulder_to_foot_angle - theta2_offset

        return (
            hip_angle,                  # around Z (abduction/adduction)
            math.degrees(theta2),      # upper leg
            math.degrees(theta3)       # knee
        )