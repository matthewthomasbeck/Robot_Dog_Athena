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
import logging # import logging for debugging





#########################################
############### UTILITIES ###############
#########################################


########## INVERSE KINEMATICS ##########

class Kinematics:
    def __init__(self, link_config):
        self.hip_offset = link_config['HIP_OFFSET']
        self.hip_to_leg_plane = link_config['HIP_TO_LEG_PLANE']
        self.femur = link_config['FEMUR_LENGTH']
        self.tibia = link_config['TIBIA_LENGTH']

    def inverse_kinematics(self, x, y, z):

        ##### calculate inverse kinematics #####

        try: # attempt to calculate inverse kinematics

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

        except Exception as e: # if there is an error calculating inverse kinematics...

            logging.error(f"(mathematics.py) Failed to calculate inverse kinematics for x={x}, y={y}, z={z}: {e}\n")


########## CALCULATE INTENSITY ##########

def interpret_intensity(intensity): # function to interpret intensity

    ##### find speed, acceleration, stride_scalar #####

    try: # attempt to interpret intensity

        if intensity == 1 or intensity == 2:
            speed = int(((16383 / 5) / 10) * intensity)
            acceleration = int(((255 / 5) / 10) * intensity)
        elif intensity == 3 or intensity == 4:
            speed = int(((16383 / 4) / 10) * intensity)
            acceleration = int(((255 / 4) / 10) * intensity)
        elif intensity == 5 or intensity == 6:
            speed = int(((16383 / 3) / 10) * intensity)
            acceleration = int(((255 / 3) / 10) * intensity)
        elif intensity == 7 or intensity == 8:
            speed = int(((16383 / 2) / 10) * intensity)
            acceleration = int(((255 / 2) / 10) * intensity)
        else:
            speed = int((16383 / 10) * intensity)
            acceleration = int((255 / 10) * intensity)

        return speed, acceleration # return movement parameters

    except Exception as e: # if there is an error interpreting intensity...

        logging.error(f"(mathematics.py) Failed to interpret intensity {intensity}: {e}\n")


########## BEZIER CURVE ##########

def bezier_curve(p0, p1, p2, steps):

    ##### set variables #####

    curve = []

    ##### create bezier curve #####

    try: # attempt to create bezier curve

        for t in [i / steps for i in range(steps + 1)]:
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
            z = (1 - t) ** 2 * p0[2] + 2 * (1 - t) * t * p1[2] + t ** 2 * p2[2]
            curve.append((x, y, z))

        return curve

    except Exception as e: # if there is an error creating bezier curve...

        logging.error(f"(mathematics.py) Failed to create bezier curve: {e}\n")