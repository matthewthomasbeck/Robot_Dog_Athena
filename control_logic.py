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


########## MANDATORY DEPENDENCIES ##########

##### mandatory libraries #####

import threading
import queue
import time
import os
import socket

##### mandatory dependencies #####

from utilities.log import initialize_logging
import utilities.config as config
from utilities.receiver import interpret_commands
from utilities.camera import decode_frame

##### (pre)initialize all utilities #####

LOGGER = initialize_logging()
CAMERA_PROCESS = None
CHANNEL_DATA = {}
SOCK = None
COMMAND_QUEUE = None
ROBOT_ID = None
JOINT_MAP = {}


########## SET REAL ROBOT DEPENDENCIES ##########

def set_real_robot_dependencies():

    ##### import/create dependencies #####

    global CAMERA_PROCESS, CHANNEL_DATA, SOCK, COMMAND_QUEUE, ROBOT_ID, JOINT_MAP, internet
    from utilities.receiver import initialize_receiver  # import receiver initialization functions
    from utilities.camera import initialize_camera  # import to start camera logic
    import utilities.internet as internet  # dynamically import internet utilities to be constantly updated

    CAMERA_PROCESS = initialize_camera()  # create camera process
    if CAMERA_PROCESS is None:
        logging.error("(control_logic.py): Failed to initialize CAMERA_PROCESS!\n")

    if config.CONTROL_MODE == 'web': # if web control mode and robot needs a socket connection for controls and video...
        SOCK = internet.initialize_backend_socket()  # initialize EC2 socket connection
        COMMAND_QUEUE = internet.initialize_command_queue(SOCK)  # initialize command queue for socket communication
        if SOCK is None:
            logging.error("(control_logic.py): Failed to initialize SOCK!\n")
        if COMMAND_QUEUE is None:
            logging.error("(control_logic.py): Failed to initialize COMMAND_QUEUE!\n")

    elif config.CONTROL_MODE == 'radio':  # if radio control mode...
        CHANNEL_DATA = initialize_receiver()  # get pigpio instance, decoders, and channel data
        if CHANNEL_DATA == None:
            logging.error("(control_logic.py): Failed to initialize CHANNEL_DATA!\n")


########## SET SIMULATED ROBOT DEPENDENCIES ##########

def set_isaac_dependencies():
    global CAMERA_PROCESS, CHANNEL_DATA, SOCK, COMMAND_QUEUE

    from isaacsim.simulation_app import SimulationApp
    config.ISAAC_SIM_APP = SimulationApp({"headless": False})

    import sys
    import carb
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.storage.native import get_assets_root_path

    config.ISAAC_WORLD = World(stage_units_in_meters=1.0)
    usd_path = os.path.expanduser("/home/matthewthomasbeck/Projects/Robot_Dog/training/urdf/robot_dog/robot_dog.usd")
    add_reference_to_stage(usd_path, "/World/robot_dog")
    config.ISAAC_ROBOT = Articulation(prim_paths_expr="/World/robot_dog", name="robot_dog")
    config.ISAAC_WORLD.scene.add(config.ISAAC_ROBOT)
    config.ISAAC_WORLD.reset()

    print("Isaac Sim initialized using SERVO_CONFIG for joint mapping")

def set_pybullet_dependencies():

    ##### import/create dependencies #####

    global CAMERA_PROCESS, CHANNEL_DATA, SOCK, COMMAND_QUEUE, ROBOT_ID, JOINT_MAP
    import numpy
    import pybullet

    def sim_decode_frame(camera_process, mjpeg_buffer):
        dummy_frame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
        return mjpeg_buffer, dummy_frame, dummy_frame

    decode_frame = sim_decode_frame
    def interpret_commands(channel_data):
        return {}

    class MockInternet:
        def initialize_backend_socket(self):
            return None
        def initialize_command_queue(self, sock):
            return None
        def stream_to_backend(self, sock, frame):
            pass

    internet = MockInternet()
    CAMERA_PROCESS = None
    CHANNEL_DATA = {}
    SOCK = None
    COMMAND_QUEUE = None
    pybullet.connect(pybullet.GUI)
    pybullet.setGravity(0, 0, -9.81)
    pybullet.loadURDF("plane.urdf")
    urdf_path = os.path.join(os.path.dirname(__file__), "training", "urdf", "robot_dog.urdf")
    ROBOT_ID = pybullet.loadURDF(urdf_path, [0, 0, 0.1], useFixedBase=True)
    JOINT_MAP = {}
    num_joints = pybullet.getNumJoints(ROBOT_ID)

    for i in range(num_joints):
        info = pybullet.getJointInfo(ROBOT_ID, i)
        joint_name = info[1].decode('utf-8')
        if '_' in joint_name and joint_name.split('_')[0] in ['FL', 'FR', 'BL', 'BR']:
            parts = joint_name.split('_')
            if len(parts) == 2 and parts[1] in ['hip', 'upper', 'lower']:
                JOINT_MAP[(parts[0], parts[1])] = i

    logging.info(f"(control_logic.py): PyBullet initialized with robot ID {ROBOT_ID}.\n")
    logging.info(f"(control_logic.py): Joint map initialized as {JOINT_MAP}.\n")
    set_simulation_variables(ROBOT_ID, JOINT_MAP)

##### movement dependencies #####

from movement.fundamental_movement import *
from movement.standing.standing import *
from movement.walking.forward import trot_forward





#########################################
############### RUN ROBOT ###############
#########################################


########## RUN ROBOTIC PROCESS ##########

##### import/create necessary dependencies #####

if not USE_SIMULATION:
    set_real_robot_dependencies()
elif USE_SIMULATION:
    if USE_ISAAC_SIM:
        set_isaac_dependencies()
    elif not USE_ISAAC_SIM:
        set_pybullet_dependencies()

##### set global variables #####

IMAGELESS_GAIT = True  # set global variable for imageless gait
IS_COMPLETE = True  # boolean that tracks if the robot is done moving, independent of it being neutral or not
IS_NEUTRAL = False  # set global neutral standing boolean
CURRENT_LEG = 'FL'  # set global current leg

##### run robotic process #####

def _perception_loop(CHANNEL_DATA):  # central function that runs robot

    ##### set/initialize variables #####

    global IS_COMPLETE, IS_NEUTRAL, CURRENT_LEG # declare as global as these will be edited by function
    mjpeg_buffer = b''  # initialize buffer for MJPEG frames

    ##### run robotic logic #####

    try:  # try to run robot startup sequence

        squatting_position(1)
        time.sleep(3)
        # tippytoes_position(1)
        # time.sleep(3)
        neutral_position(1)
        time.sleep(3)
        IS_NEUTRAL = True  # set is_neutral to True
        time.sleep(3)  # wait for 3 seconds

    except Exception as e:  # if there is an error, log error

        logging.error(f"(control_logic.py): Failed to move to neutral standing position in runRobot: {e}\n")

    ##### stream video, run inference, and control the robot #####

    try:  # try to run main robotic process

        while True:  # central loop to entire process, commenting out of importance

            # returns visual parameters, can choose to run the RL model or the CNN model for testing in config
            mjpeg_buffer, streamed_frame, inference_frame = decode_frame(
                CAMERA_PROCESS,
                mjpeg_buffer
            )
            command = None # initially no command

            if config.CONTROL_MODE == 'web': # if web control enabled...
                internet.stream_to_backend(SOCK, streamed_frame)  # stream frame data to backend

                # if command queue is not empty, get command from queue
                if COMMAND_QUEUE is not None and not COMMAND_QUEUE.empty():
                    command = COMMAND_QUEUE.get()
                    if command is not None:
                        if IS_COMPLETE: # if movement is complete, run command
                            logging.info(f"(control_logic.py): Received command '{command}' from queue (WILL RUN).\n")
                        else:
                            logging.info(f"(control_logic.py): Received command '{command}' from queue (BLOCKED).\n")

            # NEW MULTI-CHANNEL RADIO COMMAND SYSTEM
            if config.CONTROL_MODE == 'radio':
                commands = interpret_commands(CHANNEL_DATA)
                if IS_COMPLETE:  # if movement is complete, process radio commands
                    logging.debug(f"(control_logic.py): Processing radio commands: {commands}\n")
                    threading.Thread(target=_handle_command, args=(commands, inference_frame), daemon=True).start()

            # WEB COMMAND HANDLING
            if command and IS_COMPLETE: # if command present and movement complete...
                logging.debug(f"(control_logic.py): Running command: {command}...\n")
                threading.Thread(target=_handle_command, args=(command, inference_frame), daemon=True).start()

            # NEUTRAL POSITION HANDLING (for both modes)
            elif not command and IS_COMPLETE and not IS_NEUTRAL: # if no command and movement complete and not neutral...
                logging.debug(f"(control_logic.py): No command received, returning to neutral position...\n")
                threading.Thread(target=_handle_command, args=('n', inference_frame), daemon=True).start()

            # step simulation if enabled
            if USE_SIMULATION:
                if USE_ISAAC_SIM:
                    config.ISAAC_WORLD.step(render=True)
                else:
                    pybullet.stepSimulation()

    except KeyboardInterrupt:  # if user ends program...
        logging.info("(control_logic.py): KeyboardInterrupt received, exiting.\n")

    except Exception as e:  # if something breaks and only God knows what it is...
        logging.error(f"(control_logic.py): Unexpected exception in main loop: {e}\n")
        exit(1)


########## HANDLE COMMANDS ##########

def _handle_command(command, frame):

    logging.debug(f"(control_logic.py): Threading command: {command}...\n")

    global IS_COMPLETE, IS_NEUTRAL, CURRENT_LEG

    IS_COMPLETE = False  # block new commands until movement is complete

    # TODO deprecated CNN-based inference model
    #if command != 'n': # don't waste energy running inference for neutral command
        #try:
            #run_inference(
                #COMPILED_MODEL,
                #INPUT_LAYER,
                #OUTPUT_LAYER,
                #frame,
                #run_inference=False
            #)
            #logging.info(f"(control_logic.py): Ran inference for command: {command}\n")
        #except Exception as e:
            #logging.error(f"(control_logic.py): Failed to run inference for command: {e}\n")

    if isinstance(command, str):
        if '+' in command:
            keys = command.split('+')
        elif command == 'n':
            keys = []
        else:
            keys = [command]
    elif isinstance(command, (list, tuple)):
        keys = list(command)
    elif isinstance(command, dict):
        keys = []  # not used for radio mode
    else:
        keys = []

    if config.CONTROL_MODE == 'radio':
        try:
            logging.debug(f"(control_logic.py): Executing radio commands: {command}...\n")
            IS_NEUTRAL, CURRENT_LEG = _execute_radio_commands(
                command,  # command should be the full commands dict from interpret_commands
                frame,
                IS_NEUTRAL,
                CURRENT_LEG,
                tune_mode=False
            )
            logging.info(f"(control_logic.py): Executed radio commands: {command}\n")
            IS_COMPLETE = True
        except Exception as e:
            logging.error(f"(control_logic.py): Failed to execute radio commands: {e}\n")
            IS_NEUTRAL = False
            IS_COMPLETE = True

    elif config.CONTROL_MODE == 'web':
        try:
            logging.debug(f"(control_logic.py): Executing keyboard command: {keys}\n")
            IS_NEUTRAL, CURRENT_LEG = _execute_keyboard_commands(
                keys,
                frame,
                IS_NEUTRAL,
                CURRENT_LEG,
                config.DEFAULT_INTENSITY,
                tune_mode=False
            )
            logging.info(f"(control_logic.py): Executed keyboard command: {keys}\n")
            IS_COMPLETE = True
        except Exception as e:
            logging.error(f"(control_logic.py): Failed to execute keyboard command: {e}\n")
            IS_NEUTRAL = False
            IS_COMPLETE = True

##### keyboard commands for tuning mode and normal operation #####

def _execute_keyboard_commands(keys, frame, is_neutral, current_leg, intensity, tune_mode):

    global IMAGELESS_GAIT  # set IMAGELESS_GAIT as global to switch between modes via button press

    if not tune_mode:

        if 'i' in keys: # if user wishes to enable/disable imageless gait...
            IMAGELESS_GAIT = not IMAGELESS_GAIT # toggle imageless gait mode
            logging.warning(f"(control_logic.py): Toggled IMAGELESS_GAIT to {IMAGELESS_GAIT}\n")
            keys = [k for k in keys if k != 'i'] # remove 'i' from the keys list

        # cancel out contradictory keys
        if 'w' in keys and 's' in keys:
            keys = [k for k in keys if k not in ['w', 's']]
        if 'a' in keys and 'd' in keys:
            keys = [k for k in keys if k not in ['a', 'd']]
        if 'arrowleft' in keys and 'arrowright' in keys:
            keys = [k for k in keys if k not in ['arrowleft', 'arrowright']]
        if 'arrowup' in keys and 'arrowdown' in keys:
            keys = [k for k in keys if k not in ['arrowup', 'arrowdown']]

        # movement direction logic
        direction = None

        # movement (WASD and diagonals)
        if 'w' in keys and 'd' in keys:
            direction = 'w+d'
        elif 'w' in keys and 'a' in keys:
            direction = 'w+a'
        elif 's' in keys and 'd' in keys:
            direction = 's+d'
        elif 's' in keys and 'a' in keys:
            direction = 's+a'
        elif 'w' in keys:
            direction = 'w'
        elif 's' in keys:
            direction = 's'
        elif 'a' in keys:
            direction = 'a'
        elif 'd' in keys:
            direction = 'd'

        # rotation (arrow keys)
        if 'arrowleft' in keys:
            direction = 'arrowleft'
        elif 'arrowright' in keys:
            direction = 'arrowright'

        # tilt (arrow up/down)
        if 'arrowup' in keys:
            direction = 'arrowup'
        elif 'arrowdown' in keys:
            direction = 'arrowdown'

        # neutral and special actions
        if 'n' in keys or not keys:
            logging.debug(f"(control_logic.py): NEUTRAL\n")
            neutral_position(10)
            is_neutral = True
        elif ' ' in keys:
            logging.debug(f"(control_logic.py): space: LIE DOWN\n")
            squatting_position(1)
            is_neutral = False
        elif direction:
            logging.debug(f"(control_logic.py): {keys}: {direction}\n")
            trot_forward(intensity)
            #move_direction(direction, frame, intensity, IMAGELESS_GAIT)
            is_neutral = False
        else:
            logging.warning(f"(control_logic.py): Invalid command: {keys}.\n")

    else:
        # tuning mode - handle individual key actions
        for key in keys:
            if key in ADJUSTMENT_FUNCS.get(current_leg, {}):
                try:
                    ADJUSTMENT_FUNCS[current_leg][key]()
                    logging.debug(f"(control_logic.py): Tuning {current_leg} with {key}.\n")
                except Exception as e:
                        logging.error(f"(control_logic.py): Failed to adjust {current_leg} with {key}: {e}\n")
            elif key == 'q':
                # switch to next leg
                legs = ['FL', 'FR', 'BL', 'BR']
                current_index = legs.index(current_leg)
                current_leg = legs[(current_index + 1) % len(legs)]
                logging.info(f"(control_logic.py): Switched to leg: {current_leg}\n")
            elif key == 'e':
                # switch to previous leg
                legs = ['FL', 'FR', 'BL', 'BR']
                current_index = legs.index(current_leg)
                current_leg = legs[(current_index - 1) % len(legs)]
                logging.info(f"(control_logic.py): Switched to leg: {current_leg}\n")

    return is_neutral, current_leg


##### radio commands for multi-channel processing #####

def _execute_radio_commands(commands, frame, is_neutral, current_leg, tune_mode):

    global IMAGELESS_GAIT # set IMAGELESS_GAIT as global to switch between modes via button press

    # as radio is currently not reliable enough for individual button pressing, prevent image use and reserve radio as
    # backup control as model-switching is an expensive maneuver that should be avoided unless necessary
    IMAGELESS_GAIT = True

    if not tune_mode:
        active_commands = {
            channel: (action, intensity) for channel, (action, intensity) in commands.items() if action != 'NEUTRAL'
        }

        if not active_commands:
            logging.info(f"(control_logic.py): All channels neutral, returning to neutral position.\n")
            neutral_position(10)
            is_neutral = True
            return is_neutral, current_leg

        # cancel out contradictory commands (similar to keyboard logic)
        # check for contradictory movement commands
        move_actions = []
        for channel in ['channel_5', 'channel_6']:
            if channel in active_commands:
                action = active_commands[channel][0]
                if 'MOVE' in action or 'SHIFT' in action:
                    move_actions.append((channel, action))

        # determine primary direction and intensity
        direction = None
        max_intensity = 0

        # movement combinations (channels 5 and 6)
        move_forward = active_commands.get('channel_5', ('NEUTRAL', 0))[0] == 'MOVE FORWARD'
        move_backward = active_commands.get('channel_5', ('NEUTRAL', 0))[0] == 'MOVE BACKWARD'
        shift_left = active_commands.get('channel_6', ('NEUTRAL', 0))[0] == 'SHIFT LEFT'
        shift_right = active_commands.get('channel_6', ('NEUTRAL', 0))[0] == 'SHIFT RIGHT'

        # rotation (channel 3)
        rotate_left = active_commands.get('channel_3', ('NEUTRAL', 0))[0] == 'ROTATE LEFT'
        rotate_right = active_commands.get('channel_3', ('NEUTRAL', 0))[0] == 'ROTATE RIGHT'

        # tilt (channel 4)
        tilt_up = active_commands.get('channel_4', ('NEUTRAL', 0))[0] == 'TILT UP'
        tilt_down = active_commands.get('channel_4', ('NEUTRAL', 0))[0] == 'TILT DOWN'

        # special actions (channels 0, 1, 2, 7)
        look_up = active_commands.get('channel_0', ('NEUTRAL', 0))[0] == 'LOOK UP'
        look_down = active_commands.get('channel_0', ('NEUTRAL', 0))[0] == 'LOOK DOWN'
        trigger_shoot = active_commands.get('channel_1', ('NEUTRAL', 0))[0] == 'TRIGGER SHOOT'
        squat_down = active_commands.get('channel_2', ('NEUTRAL', 0))[0] == 'SQUAT DOWN'
        squat_up = active_commands.get('channel_2', ('NEUTRAL', 0))[0] == 'SQUAT UP'
        plus_action = active_commands.get('channel_7', ('NEUTRAL', 0))[0] == '+'
        minus_action = active_commands.get('channel_7', ('NEUTRAL', 0))[0] == '-'

        # build complex direction string for computer-friendly parsing
        direction_parts = []

        # handle diagonal movements (combine channels 5 and 6)
        if move_forward and shift_left:
            direction_parts.append('w+a')
            max_intensity = max(
                active_commands.get('channel_5', ('NEUTRAL', 0))[1],
                active_commands.get('channel_6', ('NEUTRAL', 0))[1]
            )
        elif move_forward and shift_right:
            direction_parts.append('w+d')
            max_intensity = max(
                active_commands.get('channel_5', ('NEUTRAL', 0))[1],
                active_commands.get('channel_6', ('NEUTRAL', 0))[1]
            )
        elif move_backward and shift_left:
            direction_parts.append('s+a')
            max_intensity = max(
                active_commands.get('channel_5', ('NEUTRAL', 0))[1],
                active_commands.get('channel_6', ('NEUTRAL', 0))[1]
            )
        elif move_backward and shift_right:
            direction_parts.append('s+d')
            max_intensity = max(
                active_commands.get('channel_5', ('NEUTRAL', 0))[1],
                active_commands.get('channel_6', ('NEUTRAL', 0))[1]
            )
        # handle single movements
        elif move_forward:
            direction_parts.append('w')
            max_intensity = active_commands.get('channel_5', ('NEUTRAL', 0))[1]
        elif move_backward:
            direction_parts.append('s')
            max_intensity = active_commands.get('channel_5', ('NEUTRAL', 0))[1]
        elif shift_left:
            direction_parts.append('a')
            max_intensity = active_commands.get('channel_6', ('NEUTRAL', 0))[1]
        elif shift_right:
            direction_parts.append('d')
            max_intensity = active_commands.get('channel_6', ('NEUTRAL', 0))[1]

        # handle rotation (can combine with movement)
        if rotate_left:
            direction_parts.append('arrowleft')
            max_intensity = max(max_intensity, active_commands.get('channel_3', ('NEUTRAL', 0))[1])
        elif rotate_right:
            direction_parts.append('arrowright')
            max_intensity = max(max_intensity, active_commands.get('channel_3', ('NEUTRAL', 0))[1])

        # handle tilt (can combine with movement and rotation)
        if tilt_up:
            direction_parts.append('arrowup')
            max_intensity = max(max_intensity, active_commands.get('channel_4', ('NEUTRAL', 0))[1])
        elif tilt_down:
            direction_parts.append('arrowdown')
            max_intensity = max(max_intensity, active_commands.get('channel_4', ('NEUTRAL', 0))[1])

        # handle special actions (these don't add to direction but are logged)
        special_actions = []
        if look_up:
            special_actions.append('LOOK_UP')
        elif look_down:
            special_actions.append('LOOK_DOWN')
        if trigger_shoot:
            special_actions.append('TRIGGER_SHOOT')
        if squat_down:
            special_actions.append('SQUAT_DOWN')
        elif squat_up:
            special_actions.append('SQUAT_UP')
        if plus_action:
            special_actions.append('PLUS')
        elif minus_action:
            special_actions.append('MINUS')

        # combine all direction parts
        if direction_parts:
            direction = '+'.join(direction_parts)
            logging.debug(f"(control_logic.py): Radio commands: ({active_commands}:{direction})\n")
            if special_actions:
                logging.debug(f"(control_logic.py): Special actions: ({special_actions})\n")
            #move_direction(direction, frame, max_intensity, IMAGELESS_GAIT)
            trot_forward(max_intensity)
            is_neutral = False
        elif special_actions:
            # only special actions, no movement
            logging.debug(f"(control_logic.py): Special actions only: ({special_actions})\n")
            # handle special actions here
            if 'SQUAT_DOWN' in special_actions:
                squatting_position(1)
                is_neutral = False
            elif 'SQUAT_UP' in special_actions:
                    neutral_position(10)
                    is_neutral = True
            # TODO: Implement other special actions

    return is_neutral, current_leg


########## EXECUTE KEYBOARD COMMANDS ##########

##### temporary dictionary of commands #####

ADJUSTMENT_FUNCS = {
    'FL': {
        'x+': adjustFL_X,
        'x-': lambda: adjustFL_X(forward=False),
        'y+': adjustFL_Y,
        'y-': lambda: adjustFL_Y(left=False),
        'z+': adjustFL_Z,
        'z-': lambda: adjustFL_Z(up=False),
    },
    'FR': {
        'x+': adjustFR_X,
        'x-': lambda: adjustFR_X(forward=False),
        'y+': adjustFR_Y,
        'y-': lambda: adjustFR_Y(left=False),
        'z+': adjustFR_Z,
        'z-': lambda: adjustFR_Z(up=False),
    },
    'BL': {
        'x+': adjustBL_X,
        'x-': lambda: adjustBL_X(forward=False),
        'y+': adjustBL_Y,
        'y-': lambda: adjustBL_Y(left=False),
        'z+': adjustBL_Z,
        'z-': lambda: adjustBL_Z(up=False),
    },
    'BR': {
        'x+': adjustBR_X,
        'x-': lambda: adjustBR_X(forward=False),
        'y+': adjustBR_Y,
        'y-': lambda: adjustBR_Y(left=False),
        'z+': adjustBR_Z,
        'z-': lambda: adjustBR_Z(up=False),
    }
}

##### run robotic process #####

def restart_process(): # start background thread to restart robot_dog.service every 30 minutes by checking elapsed time
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed >= 1800:  # 30 minutes = 1800 seconds
            os.system('sudo systemctl restart robot_dog.service')
            start_time = time.time()  # reset timer after restart
        time.sleep(1)  # check every second

def send_voltage_to_backend(voltage):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', 3000))  # Use backend's IP if remote
        msg = f'VOLTAGE:{voltage:.4f}'
        s.sendall(msg.encode())
        s.close()
    except Exception as e:
        logging.error(f"(control_logic.py) Failed to send voltage: {e}\n")

def voltage_monitor():
    while True:
        voltage_output = os.popen('vcgencmd measure_voltage').read()
        voltage_str = voltage_output.strip().replace('volt=', '').replace('V', '')
        try:
            voltage = float(voltage_str)
            if voltage < 0.8600:
                logging.warning(f"(control_logic.py) Low voltage ({voltage:.4f}V) detected!\n")
            send_voltage_to_backend(voltage)
        except ValueError:
            logging.error(f"(control_logic.py) Failed to parse voltage: {voltage_output}\n")
        time.sleep(10)  # check every 10 seconds

#restart_thread = threading.Thread(target=restart_process, daemon=True) # TODO disabling for endurance testing
#restart_thread.start()
#voltage_thread = threading.Thread(target=voltage_monitor, daemon=True) # TODO clean up your messy code first!
#voltage_thread.start()
_perception_loop(CHANNEL_DATA)  # run robot process
