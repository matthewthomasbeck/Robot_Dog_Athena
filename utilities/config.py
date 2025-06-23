##################################################################################
# Copyright (c) 2024 Matthew Thomas Beck                                         #
#                                                                                #
# All rights reserved. This code and its associated files may not be reproduced, #
# modified, distributed, or otherwise used, in part or in whole, by any person   #
# or entity without the express written permission of the copyright holder,      #
# Matthew Thomas Beck.                                                           #
##################################################################################





##########################################################
############### IMPORT/CREATE DEPENDENCIES ###############
##########################################################


########## IMPORT DEPENDENCIES ##########

import time # import time library for gait timing
import logging





#############################################################
############### CREATE UTILITY CONFIGURATIONS ###############
#############################################################


########## LOGGING CONFIG ##########

##### declare camera parameters for video #####

LOG_CONFIG = {
    'LOG_PATH': "/home/matthewthomasbeck/Projects/Robot_Dog/robot_dog.log", # path to log file DO NOT CHANGE
    'LOG_LEVEL': logging.DEBUG # set log level to logging.<DEBUG, INFO, WARNING, ERROR, or CRITICAL>
}


########## CAMERA CONFIG ##########

##### declare camera parameters for video #####

CAMERA_CONFIG = {
    'WIDTH': 640, # width of the camera image
    'HEIGHT': 480, # height of the camera image
    'FRAMERATE': 30, # frames per second of the camera image
}


########## INFERENCE CONFIG ##########

##### declare camera parameters for running AI #####

INFERENCE_CONFIG = {

    # path to model XML file DO NOT CHANGE
    'MODEL_PATH': "/home/matthewthomasbeck/Projects/Robot_Dog/model/person-detection-0200.xml",
    'TPU_NAME': "MYRIAD"  # literal device name in code
}


########## LEG LINKAGE CONFIG ##########

##### set dictionary of linkages and their lengths #####

LINK_CONFIG = { # dictionary of leg linkages # TODO rename anything 'hip' related to 'pelvis' for 3d object consistency

    'HIP_OFFSET': 0.0485394, # centerline to hip servo
    'HIP_TO_LEG_PLANE': 0.0290068, # axis to leg plane
    'FEMUR_LENGTH': 0.11, # femur length
    'TIBIA_LENGTH': 0.125, # tibia length
}


########## SERVO CONFIG ##########

##### set dictionary of servos and their ranges #####

SERVO_CONFIG = { # dictionary of leg configurations

    'FL': {'hip': {'servo': 3, 'FULL_BACK': 1236.50, 'FULL_FRONT': 1892.25, 'NEUTRAL': 1564.375, 'CUR_POS': 1564.375, 'DIR': 0, 'MOVED': False},
           'upper': {'servo': 5, 'FULL_BACK': 1921.50, 'FULL_FRONT': 1266.00, 'NEUTRAL': 1593.75, 'CUR_POS': 1593.75, 'DIR': 0, 'MOVED': False},
           'lower': {'servo': 4, 'FULL_BACK': 1872.75, 'FULL_FRONT': 1148.50, 'NEUTRAL': 1510.625, 'CUR_POS': 1510.625, 'DIR': 0, 'MOVED': False}},

    'FR': {'hip': {'servo': 2, 'FULL_BACK': 1613.25, 'FULL_FRONT': 992.00, 'NEUTRAL': 1302.625, 'CUR_POS': 1302.625, 'DIR': 0, 'MOVED': False},
           'upper': {'servo': 1, 'FULL_BACK': 1310.00, 'FULL_FRONT': 1921.50, 'NEUTRAL': 1615.75, 'CUR_POS': 1615.75, 'DIR': 0, 'MOVED': False},
           'lower': {'servo': 0, 'FULL_BACK': 1231.75, 'FULL_FRONT': 2000.00, 'NEUTRAL': 1615.875, 'CUR_POS': 1615.875, 'DIR': 0, 'MOVED': False}},

    'BL': {'hip': {'servo': 8, 'FULL_BACK': 1623.00, 'FULL_FRONT': 1036.00, 'NEUTRAL': 1329.5, 'CUR_POS': 1329.5, 'DIR': 0, 'MOVED': False},
           'upper': {'servo': 7, 'FULL_BACK': 2000.00, 'FULL_FRONT': 1354.00, 'NEUTRAL': 1777.0, 'CUR_POS': 1777.0, 'DIR': 0, 'MOVED': False},
           'lower': {'servo': 6, 'FULL_BACK': 2000.00, 'FULL_FRONT': 1138.75, 'NEUTRAL': 1569.375, 'CUR_POS': 1569.375, 'DIR': 0, 'MOVED': False}},

    'BR': {'hip': {'servo': 11, 'FULL_BACK': 1261.00, 'FULL_FRONT': 1848.25, 'NEUTRAL': 1554.625, 'CUR_POS': 1554.625, 'DIR': 0, 'MOVED': False},
           'upper': {'servo': 10, 'FULL_BACK': 1065.25, 'FULL_FRONT': 1701.50, 'NEUTRAL': 1283.375, 'CUR_POS': 1283.375, 'DIR': 0, 'MOVED': False},
           'lower': {'servo': 9, 'FULL_BACK': 1221.75, 'FULL_FRONT': 2000.00, 'NEUTRAL': 1610.875, 'CUR_POS': 1610.875, 'DIR': 0, 'MOVED': False}},
}





########################################################################
############### CREATE EUCLIDEAN-BASED LEG CONFIGURATION ###############
########################################################################


########## LEG PHASE CONFIG ##########

FL_GAIT_STATE = {'phase': 'stance', 'last_time': time.time(), 'returned_to_neutral': False}
BR_GAIT_STATE = {'phase': 'stance', 'last_time': time.time(), 'returned_to_neutral': False}
FR_GAIT_STATE = {'phase': 'swing', 'last_time': time.time(), 'returned_to_neutral': False}
BL_GAIT_STATE = {'phase': 'swing', 'last_time': time.time(), 'returned_to_neutral': False}


########## FRONT LEFT ##########

##### standing positions #####

FL_SQUATTING = {'x': 0.0400, 'y': -0.0065, 'z': 0.0250} # x is _____, y is hip up/down, z is _____
FL_NEUTRAL = {'x': -0.0450, 'y': -0.0165, 'z': -0.0750}
FL_TIPPYTOES = {'x': 0.0800, 'y': 0.0035, 'z': -0.1500}

##### walking positions #####

FL_SWING = {'x': 0.0300, 'y': 0.0165, 'z': -0.0550}
FL_TOUCHDOWN = {'x': 0.0300, 'y': 0.0015, 'z': -0.0550}
FL_MIDSTANCE = {'x': 0.0850, 'y': 0.0025, 'z': -0.1200} # TODO fix height 'x': 0.0000, 'y': 0.0075, 'z': -0.0800
FL_STANCE = {'x': -0.0300, 'y': 0.0135, 'z': -0.1050}


########## FRONT RIGHT ##########

##### standing positions #####

FR_SQUATTING = {'x': 0.1600, 'y': 0.0035, 'z': -0.1000}
FR_NEUTRAL = {'x': 0.0100, 'y': -0.0015, 'z': -0.1050}
FR_TIPPYTOES = {'x': -0.0150, 'y': -0.0015, 'z': -0.0050}

##### walking positions #####

FR_SWING = {'x': -0.1100, 'y': -0.0565, 'z': 0.0100}
FR_TOUCHDOWN = {'x': -0.1100, 'y': -0.0115, 'z': 0.0100}
FR_MIDSTANCE = {'x': -0.0600, 'y': -0.0115, 'z': -0.0350} # TODO fix height
FR_STANCE = {'x': -0.0100, 'y': -0.0115, 'z': -0.0800}


########## BACK LEFT ##########

##### standing positions #####

BL_SQUATTING = {'x': 0.0100, 'y': 0.0015, 'z': 0.0000}
BL_NEUTRAL = {'x': -0.0250, 'y': 0.0065, 'z': -0.0600}
BL_TIPPYTOES = {'x': 0.0800, 'y': -0.0085, 'z': -0.1450}

##### walking positions #####

BL_SWING = {'x': -0.0200, 'y': -0.0385, 'z': -0.0900}
BL_TOUCHDOWN = {'x': -0.0200, 'y': -0.0135, 'z': -0.0900}
BL_MIDSTANCE = {'x': -0.0200, 'y': -0.0135, 'z': -0.1250} # TODO fix height 'x': -0.0500, 'y': -0.0135, 'z': -0.1000
BL_STANCE = {'x': -0.0800, 'y': -0.0135, 'z': -0.1100}


########## BACK RIGHT ##########

##### standing positions #####

BR_SQUATTING = {'x': 0.1950, 'y': -0.0185, 'z': -0.1150}
BR_NEUTRAL = {'x': 0.0000, 'y': -0.0085, 'z': -0.0850}
BR_TIPPYTOES = {'x': -0.0100, 'y': 0.0015, 'z': -0.0050}

##### walking positions #####

BR_SWING = {'x': -0.0050, 'y': 0.0315, 'z': -0.0600}
BR_TOUCHDOWN = {'x': -0.0050, 'y': 0.0115, 'z': -0.0600}
BR_MIDSTANCE = {'x': 0.0025, 'y': 0.0025, 'z': -0.0150} # TODO fix height 'x': 0.0025, 'y': 0.0075, 'z': -0.0400
BR_STANCE = {'x': 0.0100, 'y': 0.0035, 'z': -0.0200}


########## LEG TUNE CONFIG ##########

FL_TUNE = FL_TIPPYTOES
FR_TUNE = FR_TIPPYTOES
BL_TUNE = BL_TIPPYTOES
BR_TUNE = BR_TIPPYTOES