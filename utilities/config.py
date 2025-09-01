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
RL_NOT_CNN = True # boolean to switch between testing and RL models (true is RL, false is testing)
USE_SIMULATION = True # boolean to switch between using simulator or not
USE_ISAAC_SIM = True # boolean to switch between using pybullet and isaac sim
DEFAULT_INTENSITY = 10 # default intensity for keyboard commands (1 to 10)

##### set logging configuration #####

LOG_CONFIG = {
    'LOG_PATH': "/home/matthewthomasbeck/Projects/Robot_Dog/robot_dog.log", # path to log file DO NOT CHANGE
    'LOG_LEVEL': logging.INFO # set log level to logging.<DEBUG, INFO, WARNING, ERROR, or CRITICAL>
}

########## CAMERA CONFIGURATION ##########

##### set camera configuration #####

CAMERA_CONFIG = { # TODO BE VERY CAREFUL WITH OUTPUT WIDTHxHEIGHT! Remember, the height gets cut in half via 0.5 crop!
    'FOV': 75, # degrees
    'CAMERA_WIDTH': 4608,
    'CAMERA_HEIGHT': 2592,
    'FOV_HORIZONTAL': 66,  # degrees
    'FOV_VERTICAL': 41,  # degrees
    'PIXEL_SIZE_UM': 1.4,  # pixel size in micrometers
    'DEPTH_OF_FIELD': 0.1,  # depth of field distance in meters
    'APERTURE_RATIO': 1.8,
    'WIDTH': 640, # width of the camera image
    'HEIGHT': 480, # height of the camera image
    'FRAME_RATE': 30, # frame rate of the camera in frames per second
    'CROP_FRACTION': 0.5, # fraction of the image to crop from each side (0.0 to 1.0)
    'OUTPUT_WIDTH': 128, # width of the ML image
    'OUTPUT_HEIGHT': 48, # height of the image for ML inference
}


########## INFERENCE CONFIGURATIONS ##########

##### set ML configurations #####

INFERENCE_CONFIG = {
    'TPU_NAME': "MYRIAD",  # literal device name in code
    'STANDARD_RL_PATH': "/home/matthewthomasbeck/Projects/Robot_Dog/model/standard", # standard all terrain RL model
    'BLIND_RL_PATH': "/home/matthewthomasbeck/Projects/Robot_Dog/model/blind_rl_model.xml", # speedy imageless RL model
    'CNN_PATH': "/home/matthewthomasbeck/Projects/Robot_Dog/model/person-detection-0200.xml",  # person detection
}


########## ROBOT CONTROL CONFIGURATIONS (internet and radio) ##########

##### declare movement channel GPIO pins #####

SIGNAL_TUNING_CONFIG = { # dictionary of signal tuning configuration for sensitivity
    'JOYSTICK_THRESHOLD': 40, # number of times condition must be met to trigger a request on a joystick channel
    'TOGGLE_THRESHOLD': 40, # number of times condition must be met to trigger a request on a button channel
    'TIME_FRAME': 0.10017, # time frame for condition to be met, default: 0.100158
    'DEADBAND_HIGH': 1600, # deadband high for PWM signal
    'DEADBAND_LOW': 1400 # deadband low for PWM signal
}

##### set receiver channels #####

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

    'FL': {'hip': {'servo': 3, 'FULL_FRONT': 1892.25, 'FULL_BACK': 1236.50, 'NEUTRAL': 1564.375, 'CURRENT': 1564.375, 'FULL_FRONT_ANGLE': 0.958934, 'FULL_BACK_ANGLE': -0.349066, 'CURRENT_ANGLE': 0.0, 'NEUTRAL_ANGLE': 0.0},
           'upper': {'servo': 5, 'FULL_FRONT': 1266.00, 'FULL_BACK': 1921.50, 'NEUTRAL': 1593.75, 'CURRENT': 1593.75, 'FULL_FRONT_ANGLE': -0.654, 'FULL_BACK_ANGLE': 0.654, 'CURRENT_ANGLE': 0.0, 'NEUTRAL_ANGLE': 0.0},
           'lower': {'servo': 4, 'FULL_FRONT': 1148.50, 'FULL_BACK': 1872.75, 'NEUTRAL': 1510.625, 'CURRENT': 1510.625, 'FULL_FRONT_ANGLE': -0.698, 'FULL_BACK_ANGLE': 0.698, 'CURRENT_ANGLE': 0.0, 'NEUTRAL_ANGLE': 0.0}},

    'FR': {'hip': {'servo': 2, 'FULL_FRONT': 992.00, 'FULL_BACK': 1613.25, 'NEUTRAL': 1302.625, 'CURRENT': 1302.625, 'FULL_FRONT_ANGLE': -0.958934, 'FULL_BACK_ANGLE': 0.349066, 'CURRENT_ANGLE': 0.0, 'NEUTRAL_ANGLE': 0.0},
           'upper': {'servo': 1, 'FULL_FRONT': 1921.50, 'FULL_BACK': 1310.00, 'NEUTRAL': 1615.75, 'CURRENT': 1615.75, 'FULL_FRONT_ANGLE': 0.654, 'FULL_BACK_ANGLE': -0.654, 'CURRENT_ANGLE': 0.0, 'NEUTRAL_ANGLE': 0.0},
           'lower': {'servo': 0, 'FULL_FRONT': 2000.00, 'FULL_BACK': 1231.75, 'NEUTRAL': 1615.875, 'CURRENT': 1615.875, 'FULL_FRONT_ANGLE': 0.698, 'FULL_BACK_ANGLE': -0.698, 'CURRENT_ANGLE': 0.0, 'NEUTRAL_ANGLE': 0.0}},

    'BL': {'hip': {'servo': 8, 'FULL_FRONT': 1036.00, 'FULL_BACK': 1623.00, 'NEUTRAL': 1329.5, 'CURRENT': 1329.5, 'FULL_FRONT_ANGLE': -0.958934, 'FULL_BACK_ANGLE': 0.349066, 'CURRENT_ANGLE': 0.0, 'NEUTRAL_ANGLE': 0.0},
           'upper': {'servo': 7, 'FULL_FRONT': 1354.00, 'FULL_BACK': 2000.00, 'NEUTRAL': 1777.0, 'CURRENT': 1777.0, 'FULL_FRONT_ANGLE': -0.654, 'FULL_BACK_ANGLE': 0.654, 'CURRENT_ANGLE': 0.0, 'NEUTRAL_ANGLE': 0.0},
           'lower': {'servo': 6, 'FULL_FRONT': 1138.75, 'FULL_BACK': 2000.00, 'NEUTRAL': 1569.375, 'CURRENT': 1569.375, 'FULL_FRONT_ANGLE': -0.698, 'FULL_BACK_ANGLE': 0.698, 'CURRENT_ANGLE': 0.0, 'NEUTRAL_ANGLE': 0.0}},

    'BR': {'hip': {'servo': 11, 'FULL_FRONT': 1848.25, 'FULL_BACK': 1261.00, 'NEUTRAL': 1554.625, 'CURRENT': 1554.625, 'FULL_FRONT_ANGLE': 0.958934, 'FULL_BACK_ANGLE': -0.349066, 'CURRENT_ANGLE': 0.0, 'NEUTRAL_ANGLE': 0.0},
           'upper': {'servo': 10, 'FULL_FRONT': 1701.50, 'FULL_BACK': 1065.25, 'NEUTRAL': 1283.375, 'CURRENT': 1283.375, 'FULL_FRONT_ANGLE': 0.654, 'FULL_BACK_ANGLE': -0.654, 'CURRENT_ANGLE': 0.0, 'NEUTRAL_ANGLE': 0.0},
           'lower': {'servo': 9, 'FULL_FRONT': 2000.00, 'FULL_BACK': 1221.75, 'NEUTRAL': 1610.875, 'CURRENT': 1610.875, 'FULL_FRONT_ANGLE': 0.698, 'FULL_BACK_ANGLE': -0.698, 'CURRENT_ANGLE': 0.0, 'NEUTRAL_ANGLE': 0.0}},
}


########## ISAAC SIM CONFIGURATION ##########

##### isaac sim paths #####

ISAAC_ROBOT_PATH = "/home/matthewthomasbeck/Projects/Robot_Dog/training/urdf/robot_dog/robot_dog.usd"
MODELS_DIRECTORY = "/home/matthewthomasbeck/Projects/Robot_Dog/model"

##### isaac sim objects #####

ISAAC_SIM_APP = None # isaac sim application instance
ISAAC_WORLD = None # isaac sim world

# Multi-robot arrays for parallel training
ISAAC_ROBOTS = [] # array of isaac sim robot articulations
ISAAC_ROBOT_ARTICULATION_CONTROLLERS = [] # array of isaac sim robot articulation controllers
CAMERA_PROCESSES = [] # array of camera processes for each robot
SERVO_CONFIGS = [] # array of servo configurations for each robot

##### isaac sim joint config #####

JOINT_INDEX_MAP = None # placeholder for joint configuration, to be set by isaac sim

##### multi-robot configuration #####

MULTI_ROBOT_CONFIG = {
    'num_robots': 1,  # number of robots to spawn for parallel training (back to 2 robots)
    'robot_spacing': 2.0,  # spacing between robots in meters
    'robot_start_z': 0.14,  # starting height for robots to avoid clipping
    'robot_positions': [  # predefined positions for robots (x, y, z) - 2 robots
        #(-3.0, -3.0, 0.14),  # robot 0: front-left
        #(3.0, -3.0, 0.14),   # robot 1: front-right  
        (0.0, 0.0, 0.14),   # robot 0: center position (commented out)
        #(-3.0, 3.0, 0.14),   # robot 2: back-left (commented out)
        #(3.0, 3.0, 0.14),    # robot 3: back-right (commented out)
    ]
}

##### training config #####

TRAINING_CONFIG = { # used to track training metrics and save frequencies

    'max_episodes': 10000000,
    'max_steps_per_episode': 750,  # GPT-5 recommendation: 600-1200 steps (~10-20 seconds)
    'save_frequency': 20000,  # Save model every 20000 steps (quick testing)
    'training_frequency': 2,  # Train every 2 steps (GPT-5: more frequent training)
    'batch_size': 64,  # GPT-5 recommendation: standard batch size
    'learning_rate': 3e-4,  # Back to standard learning rate
    'gamma': 0.99,  # Discount factor
    'tau': 0.005,  # Standard target network update rate
    'exploration_noise': 0.1,  # Standard exploration noise
    'max_action': 1.0
}
