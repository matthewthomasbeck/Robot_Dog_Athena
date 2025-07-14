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

##### import necessary libraries #####

import time # import time library for gait timing
import logging # import logging library for debugging





#####################################################
############### CREATE CONFIGURATIONS ###############
#####################################################


########## UTILITY CONFIGURATIONS ##########

##### set global fps to be used by all modules #####

LOOP_RATE_HZ = 30 # global loop rate in Hz for all modules TODO DEPRECATED/LEGACY
CONTROL_MODE = 'web' # current control mode of the robot (web or radio)
RL_NOT_CNN: False  # boolean to switch between testing and RL models (true is RL, false is testing)
DEFAULT_INTENSITY = 7 # default intensity for keyboard commands (1 to 10)

##### set logging configuration #####

LOG_CONFIG = {
    'LOG_PATH': "/home/matthewthomasbeck/Projects/Robot_Dog/robot_dog.log", # path to log file DO NOT CHANGE
    'LOG_LEVEL': logging.INFO # set log level to logging.<DEBUG, INFO, WARNING, ERROR, or CRITICAL>
}

##### set camera configuration #####

CAMERA_CONFIG = {
    'WIDTH': 640, # width of the camera image
    'HEIGHT': 480, # height of the camera image
    'FRAME_RATE': 30, # frame rate of the camera in frames per second
    'CROP_FRACTION': 0.5, # fraction of the image to crop from each side (0.0 to 1.0)
    'OUTPUT_WIDTH': 640, # width of the ML image
    'OUTPUT_HEIGHT': 480, # height of the image for ML inference
}

##### set inference configuration #####

INFERENCE_CONFIG = {
    'TPU_NAME': "MYRIAD",  # literal device name in code
    'RL_PATH': "/home/matthewthomasbeck/Projects/Robot_Dog/model/", # in-house RL model(s)
    'CNN_PATH': "/home/matthewthomasbeck/Projects/Robot_Dog/model/person-detection-0200.xml",  # person detection
}

##### declare movement channel GPIO pins #####

SIGNAL_TUNING_CONFIG = { # dictionary of signal tuning configuration for sensitivity
    'JOYSTICK_THRESHOLD': 40, # number of times condition must be met to trigger a request on a joystick channel
    'TOGGLE_THRESHOLD': 40, # number of times condition must be met to trigger a request on a button channel
    'TIME_FRAME': 0.10017, # time frame for condition to be met, default: 0.100158
    'DEADBAND_HIGH': 1600, # deadband high for PWM signal
    'DEADBAND_LOW': 1400 # deadband low for PWM signal
}

RECEIVER_CHANNELS = { # dictionary of receiver channels' names, GPIO pins, and states
    'channel_0': {'name': 'tilt_up_down', 'gpio_pin': 17, 'counter': 0, 'timestamp': time.time()},
    'channel_1': {'name': 'trigger_shoot', 'gpio_pin': 27, 'counter': 0, 'timestamp': time.time()},
    'channel_2': {'name': 'squat_up_down', 'gpio_pin': 22, 'counter': 0, 'timestamp': time.time()},
    'channel_3': {'name': 'rotate_left_right', 'gpio_pin': 5, 'counter': 0, 'timestamp': time.time()},
    'channel_4': {'name': 'look_up_down', 'gpio_pin': 6, 'counter': 0, 'timestamp': time.time()},
    'channel_5': {'name': 'move_forward_backward', 'gpio_pin': 13, 'counter': 0, 'timestamp': time.time()},
    'channel_6': {'name': 'shift_left_right', 'gpio_pin': 19, 'counter': 0, 'timestamp': time.time()},
    'channel_7': {'name': 'extra_channel', 'gpio_pin': 26, 'counter': 0, 'timestamp': time.time()},
}

##### set receiver configuration #####

MAESTRO_CONFIG = {
    'SERIAL_PATH': "/dev/serial0", # set serial port name to first available
    'SERIAL_BAUD_RATE': 9600, # set baud rate for serial connection
    'SERIAL_TIMEOUT': 1 # set timeout for serial connection
}

##### set internet connectivity configuration #####

INTERNET_CONFIG = {
    'BACKEND_API_URL': "https://api.matthewthomasbeck.com", # URL of the backend API endpoint
    'BACKEND_PUBLIC_IP': "72.177.232.19", # public IP address of backend
    'BACKEND_PORT': 3000, # port number for backend (fixed typo from 'BACKED_PORT')
    'SSH_SOCKET_PATH': "/tmp/robot.sock" # path to unix socket for SSH communication
}


########## PHYSICAL CONFIGURATION ##########

##### set link configuration #####

LINK_CONFIG = { # dictionary of leg linkages # TODO rename anything 'hip' related to 'pelvis' for 3d object consistency
    'HIP_OFFSET': 0.0485394, # centerline to hip servo
    'HIP_TO_LEG_PLANE': 0.0290068, # axis to leg plane
    'FEMUR_LENGTH': 0.11, # femur length
    'TIBIA_LENGTH': 0.125, # tibia length
}

##### set dictionary of servos and their ranges #####

SERVO_CONFIG = { # dictionary of leg configurations

    'FL': {'hip': {'servo': 3, 'FULL_FRONT': 1892.25, 'FULL_BACK': 1236.50, 'NEUTRAL': 1564.375, 'FULL_FRONT_ANGLE': 0.349066, 'FULL_BACK_ANGLE': -0.349066},
           'upper': {'servo': 5, 'FULL_FRONT': 1266.00, 'FULL_BACK': 1921.50, 'NEUTRAL': 1593.75, 'FULL_FRONT_ANGLE': -0.654, 'FULL_BACK_ANGLE': 0.654},
           'lower': {'servo': 4, 'FULL_FRONT': 1148.50, 'FULL_BACK': 1872.75, 'NEUTRAL': 1510.625, 'FULL_FRONT_ANGLE': -0.698, 'FULL_BACK_ANGLE': 0.698}},

    'FR': {'hip': {'servo': 2, 'FULL_FRONT': 992.00, 'FULL_BACK': 1613.25, 'NEUTRAL': 1302.625, 'FULL_FRONT_ANGLE': -0.349066, 'FULL_BACK_ANGLE': 0.349066},
           'upper': {'servo': 1, 'FULL_FRONT': 1921.50, 'FULL_BACK': 1310.00, 'NEUTRAL': 1615.75, 'FULL_FRONT_ANGLE': 0.654, 'FULL_BACK_ANGLE': -0.654},
           'lower': {'servo': 0, 'FULL_FRONT': 2000.00, 'FULL_BACK': 1231.75, 'NEUTRAL': 1615.875, 'FULL_FRONT_ANGLE': 0.698, 'FULL_BACK_ANGLE': -0.698}},

    'BL': {'hip': {'servo': 8, 'FULL_FRONT': 1036.00, 'FULL_BACK': 1623.00, 'NEUTRAL': 1329.5, 'FULL_FRONT_ANGLE': -0.349066, 'FULL_BACK_ANGLE': 0.349066},
           'upper': {'servo': 7, 'FULL_FRONT': 1354.00, 'FULL_BACK': 2000.00, 'NEUTRAL': 1777.0, 'FULL_FRONT_ANGLE': -0.654, 'FULL_BACK_ANGLE': 0.654},
           'lower': {'servo': 6, 'FULL_FRONT': 1138.75, 'FULL_BACK': 2000.00, 'NEUTRAL': 1569.375, 'FULL_FRONT_ANGLE': -0.698, 'FULL_BACK_ANGLE': 0.698}},

    'BR': {'hip': {'servo': 11, 'FULL_FRONT': 1848.25, 'FULL_BACK': 1261.00, 'NEUTRAL': 1554.625, 'FULL_FRONT_ANGLE': 0.349066, 'FULL_BACK_ANGLE': -0.349066},
           'upper': {'servo': 10, 'FULL_FRONT': 1701.50, 'FULL_BACK': 1065.25, 'NEUTRAL': 1283.375, 'FULL_FRONT_ANGLE': 0.654, 'FULL_BACK_ANGLE': -0.654},
           'lower': {'servo': 9, 'FULL_FRONT': 2000.00, 'FULL_BACK': 1221.75, 'NEUTRAL': 1610.875, 'FULL_FRONT_ANGLE': 0.698, 'FULL_BACK_ANGLE': -0.698}},
}

##### set dictionary of feet current positions for the AI model #####

CURRENT_FEET_POSITIONS = {
    'FL': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    'FR': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    'BL': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    'RL': {'x': 0.0, 'y': 0.0, 'z': 0.0}
}





#####################################################################################
############### CREATE EUCLIDEAN-BASED LEG CONFIGURATION (HAND-TUNED) ###############
#####################################################################################

# TODO NOTE: THESE ARE ALL DEPRECATED AND ONLY USED IN HAND-TUNED MOVEMENTS


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