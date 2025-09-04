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
import logging
from collections import deque
import numpy as np

##### mandatory dependencies #####

from utilities.log import initialize_logging
import utilities.config as config
from utilities.receiver import interpret_commands
import utilities.internet as internet  # dynamically import internet utilities to be constantly updated

##### (pre)initialize all utilities #####

LOGGER = initialize_logging()
CAMERA_PROCESS = None
CHANNEL_DATA = {}
SOCK = None
COMMAND_QUEUE = None
ROBOT_ID = None
JOINT_MAP = {}


########## PREPARE ROBOT ##########

##### prepare real robot #####

def set_real_robot_dependencies():  # function to initialize real robot dependencies

    ##### import necessary functions #####

    from utilities.receiver import initialize_receiver  # import receiver initialization functions
    from utilities.camera import initialize_camera  # import to start camera logic
    import utilities.internet as internet  # dynamically import internet utilities to be constantly updated

    ##### initialize global variables #####

    global CAMERA_PROCESS, CHANNEL_DATA, SOCK, COMMAND_QUEUE, ROBOT_ID, JOINT_MAP, internet

    ##### initialize PREVIOUS_POSITIONS for physical robot (1 robot) #####
    
    # Initialize PREVIOUS_POSITIONS for single physical robot
    config.PREVIOUS_POSITIONS = []
    robot_history = deque(maxlen=5)
    for _ in range(5):
        robot_history.append(np.zeros(12, dtype=np.float32))
    config.PREVIOUS_POSITIONS.append(robot_history)
    logging.debug("PREVIOUS_POSITIONS initialized for physical robot with zeros")

    ##### initialize camera process #####

    CAMERA_PROCESS = initialize_camera()  # create camera process
    if CAMERA_PROCESS is None:
        logging.error("(control_logic.py): Failed to initialize CAMERA_PROCESS for robot!\n")

    ##### initialize socket and command queue #####

    if config.CONTROL_MODE == 'web':  # if web control mode and robot needs a socket connection for controls and video...
        SOCK = internet.initialize_backend_socket()  # initialize EC2 socket connection
        COMMAND_QUEUE = internet.initialize_command_queue(SOCK)  # initialize command queue for socket communication
        if SOCK is None:
            logging.error("(control_logic.py): Failed to initialize SOCK for robot!\n")
        if COMMAND_QUEUE is None:
            logging.error("(control_logic.py): Failed to initialize COMMAND_QUEUE for robot!\n")

    ##### initialize channel data #####

    elif config.CONTROL_MODE == 'radio':  # if radio control mode...
        CHANNEL_DATA = initialize_receiver()  # get pigpio instance, decoders, and channel data
        if CHANNEL_DATA == None:
            logging.error("(control_logic.py): Failed to initialize CHANNEL_DATA for robot!\n")


##### prepare isaac sim #####

def set_isaac_dependencies():  # function to initialize isaac sim dependencies

    global CAMERA_PROCESS, CHANNEL_DATA, SOCK, COMMAND_QUEUE
    import sys
    import carb
    import numpy
    import copy
    from collections import deque

    # IMPORTANT 'SimulationApp' MUST be imported and made before any other isaac utilization of any kind!!!
    from isaacsim.simulation_app import SimulationApp
    config.ISAAC_SIM_APP = SimulationApp({"headless": False})
    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
    from isaacsim.core.api.controllers.articulation_controller import ArticulationController
    from training.isaac_sim import build_isaac_joint_index_map
    from utilities.camera import initialize_camera

    ##### build simulated world #####

    config.ISAAC_WORLD = World(stage_units_in_meters=1.0)  # create a new world
    config.ISAAC_WORLD.scene.add_default_ground_plane()  # add a ground plane

    ##### spawn multiple robots for parallel training #####

    # Import grid positioning function
    from training.isaac_sim import generate_grid_positions

    # Initialize camera processes list for multi-robot setup
    config.CAMERA_PROCESSES = []
    
    # Initialize PREVIOUS_POSITIONS for movement history (replaces CURRENT_ANGLES)
    config.PREVIOUS_POSITIONS = []
    
    # Generate robot positions using grid algorithm
    robot_positions = generate_grid_positions(
        config.MULTI_ROBOT_CONFIG['num_robots'],
        config.MULTI_ROBOT_CONFIG['robot_spacing'],
        config.MULTI_ROBOT_CONFIG['robot_start_z']
    )
    
    # Save spawn positions to config for individual robot resets
    config.SPAWN_POSITIONS = robot_positions
    
    logging.info(f"(control_logic.py): Spawning {config.MULTI_ROBOT_CONFIG['num_robots']} robots for parallel training...\n")
    logging.info(f"(control_logic.py): Generated grid positions: {robot_positions}\n")
    logging.info(f"(control_logic.py): Saved spawn positions to config for individual resets\n")

    for robot_id in range(config.MULTI_ROBOT_CONFIG['num_robots']):
        try:
            # Create unique path for each robot
            robot_path = f"/World/robot_dog_{robot_id}"
            
            # Add robot to world at specified position
            add_reference_to_stage(config.ISAAC_ROBOT_PATH, robot_path)
            
            # Move robot to its designated position
            import omni.usd
            from pxr import UsdGeom, Gf
            
            stage = omni.usd.get_context().get_stage()
            robot_prim = stage.GetPrimAtPath(robot_path)
            
            if robot_prim:
                xform = UsdGeom.Xformable(robot_prim)
                if xform:
                    xform_ops = xform.GetOrderedXformOps()
                    translate_op = None
                    for op in xform_ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            translate_op = op
                            break
                    
                    if translate_op:
                        # Get position from generated grid
                        pos = robot_positions[robot_id]
                        translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
                        logging.info(f"(control_logic.py): Robot {robot_id} positioned at {pos}\n")
                    else:
                        logging.warning(f"(control_logic.py): No translate operation found on robot {robot_id} prim.\n")
                else:
                    logging.warning(f"(control_logic.py): Robot {robot_id} prim found but not Xformable.\n")
            else:
                logging.warning(f"(control_logic.py): Robot {robot_id} prim not found at {robot_path}.\n")
                
        except Exception as e:
            logging.error(f"(control_logic.py): Failed to spawn robot {robot_id}: {e}\n")

    ##### let isaac sim load and create robot objects #####

    for _ in range(3):  # let isaac sim load a few steps for general process
        config.ISAAC_WORLD.step(render=True)

    ##### create robot articulations and controllers #####

    for robot_id in range(config.MULTI_ROBOT_CONFIG['num_robots']):
        try:
            robot_path = f"/World/robot_dog_{robot_id}"
            robot_name = f"robot_dog_{robot_id}"
            
            # Create robot articulation
            robot = Articulation(prim_paths_expr=robot_path, name=robot_name)
            config.ISAAC_WORLD.scene.add(robot)
            config.ISAAC_ROBOTS.append(robot)
            
            # Initialize movement history for this robot (5 zero arrays of 12D each)
            robot_history = deque(maxlen=5)
            for _ in range(5):
                robot_history.append(np.zeros(12, dtype=np.float32))
            config.PREVIOUS_POSITIONS.append(robot_history)
            
            # Initialize camera process for this robot
            #camera_process = initialize_camera(robot_id=robot_id)
            #config.CAMERA_PROCESSES.append(camera_process)
            
            logging.info(f"(control_logic.py): Robot {robot_id} initialized successfully.\n")
            
        except Exception as e:
            logging.error(f"(control_logic.py): Failed to initialize robot {robot_id}: {e}\n")

    ##### reset world and log success #####

    config.ISAAC_WORLD.reset()
    
    # Give Isaac Sim a few more steps to ensure robots are fully initialized
    for _ in range(5):
        config.ISAAC_WORLD.step(render=True)
    
    logging.info(f"(control_logic.py): Successfully spawned {len(config.ISAAC_ROBOTS)} robots for parallel training.\n")

    ##### initialize isaac sim command queue #####

    try:  # try to initialize command queue for isaac to get commands from training
        COMMAND_QUEUE = queue.Queue()
        logging.info("(control_logic.py): RL command queue initialized successfully for Isaac Sim.\n")
    except Exception as e:
        logging.error(f"(control_logic.py): Failed to initialize RL command queue: {e}\n")
        COMMAND_QUEUE = None

    # TODO dont remove the commented out code below, I may need it someday
    # if config.CONTROL_MODE == 'web': # if web control mode and robot needs a socket connection for controls and video...
    # SOCK = internet.initialize_backend_socket()  # initialize EC2 socket connection
    # COMMAND_QUEUE = internet.initialize_command_queue(SOCK)  # initialize command queue for socket communication
    # if SOCK is None:
    # logging.error("(control_logic.py): Failed to initialize SOCK for robot!\n")
    # if COMMAND_QUEUE is None:
    # logging.error("(control_logic.py): Failed to initialize COMMAND_QUEUE for robot!\n")


########## PREPARE ROBOT ##########

##### prepare robot with correct dependencies #####

if not config.USE_SIMULATION:
    set_real_robot_dependencies()
elif config.USE_SIMULATION:
    set_isaac_dependencies()

##### post-initialization dependencies #####

from movement.movement_coordinator import *
from utilities.camera import decode_real_frame, decode_isaac_frame





#########################################
############### RUN ROBOT ###############
#########################################


########## STATE MACHINE LOOPS ##########

##### set global variables #####

IMAGELESS_GAIT = True  # set global variable for imageless gait
IS_COMPLETE = True  # boolean that tracks if the robot is done moving, independent of it being neutral or not
IS_NEUTRAL = False  # set global neutral standing boolean
CURRENT_LEG = 'FL'  # set global current leg


##### physical loop #####

def _physical_loop(CHANNEL_DATA):  # central function that runs robot in real life

    ##### set/initialize variables #####

    global IS_COMPLETE, IS_NEUTRAL, CURRENT_LEG  # declare as global as these will be edited by function
    mjpeg_buffer = b''  # initialize buffer for MJPEG frames

    ##### run robotic logic #####

    try:  # try to run robot startup sequence
        neutral_position(1)
        time.sleep(3)
        IS_NEUTRAL = True  # set is_neutral to True

    except Exception as e:  # if there is an error, log error
        logging.error(f"(control_logic.py): Failed to move to neutral standing position in runRobot: {e}\n")

    ##### stream video, run inference, and control the robot #####

    try:  # try to run main robotic process

        while True:  # central loop to entire process, commenting out of importance

            mjpeg_buffer, streamed_frame, inference_frame = decode_real_frame(  # run camera and decode frame
                CAMERA_PROCESS,
                mjpeg_buffer
            )
            command = None  # initially no command

            if config.CONTROL_MODE == 'web':  # if web control enabled...
                internet.stream_to_backend(SOCK, streamed_frame)

                if COMMAND_QUEUE is not None and not COMMAND_QUEUE.empty():  # if command queue is not empty...
                    command = COMMAND_QUEUE.get()  # get command from queue
                    if command is not None:
                        if IS_COMPLETE:  # if movement is complete, run command
                            logging.info(f"(control_logic.py): Received command '{command}' from queue (WILL RUN).\n")
                        else:
                            logging.info(f"(control_logic.py): Received command '{command}' from queue (BLOCKED).\n")

            # NEW MULTI-CHANNEL RADIO COMMAND SYSTEM
            if config.CONTROL_MODE == 'radio':
                commands = interpret_commands(CHANNEL_DATA)
                if IS_COMPLETE:  # if movement is complete, process radio commands
                    logging.debug(f"(control_logic.py): Processing radio commands: {commands}\n")
                    threading.Thread(target=_handle_command, args=(commands, inference_frame), daemon=True).start()

            # WEB COMMAND HANDLING (includes RL commands for Isaac Sim)
            if command and IS_COMPLETE:  # if command present and movement complete...
                # logging.debug(f"(control_logic.py): Running command: {command}...\n")
                threading.Thread(target=_handle_command, args=(command, inference_frame), daemon=True).start()

            # NEUTRAL POSITION HANDLING (for both modes)
            elif not command and IS_COMPLETE and not IS_NEUTRAL:  # if no command and movement complete and not neutral...
                logging.debug(f"(control_logic.py): No command received, returning to neutral position...\n")
                threading.Thread(target=_handle_command, args=('n', inference_frame), daemon=True).start()

    except KeyboardInterrupt:  # if user ends program...
        logging.info("(control_logic.py): KeyboardInterrupt received, exiting.\n")

    except Exception as e:  # if something breaks and only God knows what it is...
        logging.error(f"(control_logic.py): Unexpected exception in main loop: {e}\n")
        exit(1)


##### isaac sim loop #####

def _isaac_sim_loop():  # central function that runs robot in simulation

    ##### loop pre-check #####

    if config.CONTROL_MODE != 'web':
        logging.error(f"(control_logic.py): Isaac Sim loop only works with web control mode, exiting.\n")
        return

    global IS_COMPLETE, IS_NEUTRAL, CURRENT_LEG  # declare as global as these will be edited by function
    mjpeg_buffer = b''  # initialize buffer for MJPEG frames

    for _ in range(3):  # let isaac sim load a few steps for general process
        config.ISAAC_WORLD.step(render=True)

    try:  # try to run robot startup sequence for all robots
        neutral_position_isaac()
        time.sleep(3)
        IS_NEUTRAL = True  # set is_neutral to True

    except Exception as e:  # if there is an error, log error
        logging.error(f"(control_logic.py): Failed to move to neutral standing position: {e}\n")

    ##### stream video, run inference, and control the robot #####

    try:  # try to run main robotic process

        while True:  # central loop to entire process, commenting out of importance

            ##### decode frame #####

            # Multi-robot camera processing
            camera_frames = []
            
            #if config.CAMERA_PROCESSES:
                #for robot_id, camera_process in enumerate(config.CAMERA_PROCESSES):
                    #try:
                        #mjpeg_buffer, streamed_frame, inference_frame = decode_isaac_frame(camera_process)
                        #camera_frames.append({
                            #'robot_id': robot_id,
                            #'mjpeg_buffer': mjpeg_buffer,
                            #'streamed_frame': streamed_frame,
                            #'inference_frame': inference_frame
                        #})
                    #except Exception as e:
                        #logging.error(f"(control_logic.py): Failed to decode frame for robot {robot_id}: {e}\n")
                
                #if not camera_frames:
                    #logging.warning("(control_logic.py): No camera frames processed")

            ##### get command #####

            command = None  # initially no command

            # Generate RL commands if queue is empty and robot is ready for new commands
            if COMMAND_QUEUE is not None and COMMAND_QUEUE.empty() and IS_COMPLETE:
                
                # Import and use random command generation
                from training.isaac_sim import get_random_command
                # Set training phase here (1, 2, or 3)
                training_phase = 1  # Start with phase 1 for basic movement
                command = get_random_command(phase=training_phase)

            # Check RL command queue for Isaac Sim
            if COMMAND_QUEUE is not None and not COMMAND_QUEUE.empty():
                command_data = COMMAND_QUEUE.get()  # Get command data from RL queue
                if command_data is not None:
                    # Extract command and intensity from RL command data
                    rl_command = command_data.get('command')

                    # Convert RL command format to web command format for consistency
                    if rl_command is None:
                        command = 'n'  # Neutral command
                    else:
                        command = rl_command  # Use RL command directly

            if COMMAND_QUEUE is not None and not COMMAND_QUEUE.empty():  # if command queue is not empty...
                command = COMMAND_QUEUE.get()  # get command from queue
                if command is not None:
                    if IS_COMPLETE:  # if movement is complete, run command
                        logging.info(f"(control_logic.py): Received command '{command}' from queue (WILL RUN).\n")
                    else:
                        logging.info(f"(control_logic.py): Received command '{command}' from queue (BLOCKED).\n")

            ##### command handling #####

            if command and IS_COMPLETE:  # if command present and movement complete...
                threading.Thread(target=_handle_command, args=(command, camera_frames), daemon=True).start()

            # NEUTRAL POSITION HANDLING (for both modes)
            elif not command and IS_COMPLETE and not IS_NEUTRAL:  # if no command and movement complete and not neutral...
                logging.debug(f"(control_logic.py): No command received, returning to neutral position...\n")
                threading.Thread(target=_handle_command, args=('n', camera_frames), daemon=True).start()

            ##### step simulation #####

            config.ISAAC_WORLD.step(render=True)

            if config.USE_SIMULATION: # TODO ensure all robots pause when one robot resets
                from training.training import integrate_with_main_loop
                episode_reset_occurred = integrate_with_main_loop()
                if episode_reset_occurred:
                    for _ in range(3):
                        config.ISAAC_WORLD.step(render=True)

    except Exception as e:  # if something breaks and only God knows what it is...
        logging.error(f"(control_logic.py): Unexpected exception in main loop: {e}\n")
        exit(1)


########## HANDLE COMMANDS ##########

def _handle_command(command, camera_frames=None):
    # logging.debug(f"(control_logic.py): Threading command: {command}...\n")

    global IS_COMPLETE, IS_NEUTRAL, CURRENT_LEG
    IS_COMPLETE = False  # block new commands until movement is complete

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
                camera_frames,
                IS_NEUTRAL,
                CURRENT_LEG
            )
            logging.info(f"(control_logic.py): Executed radio commands: {command}\n")
            IS_COMPLETE = True
        except Exception as e:
            logging.error(f"(control_logic.py): Failed to execute radio commands: {e}\n")
            IS_NEUTRAL = False
            IS_COMPLETE = True

    elif config.CONTROL_MODE == 'web':

        if config.USE_SIMULATION:
            # Import and use random intensity generation
            from training.isaac_sim import get_random_intensity
            # Set training phase here (1, 2, or 3) - same as command phase
            training_phase = 1  # Start with phase 1 for basic movement
            intensity = get_random_intensity(phase=training_phase)

        else:
            intensity = 10

        try:
            IS_NEUTRAL, CURRENT_LEG = _execute_keyboard_commands(
                keys,
                camera_frames,
                IS_NEUTRAL,
                CURRENT_LEG,
                intensity
            )
            # logging.info(f"(control_logic.py): Executed keyboard command: {keys}\n")
            IS_COMPLETE = True
        except Exception as e:
            logging.error(f"(control_logic.py): Failed to execute keyboard command: {e}\n")
            IS_NEUTRAL = False
            IS_COMPLETE = True


########## EXECUTE COMMANDS ##########

def _convert_direction_parts_to_fixed_list(direction_parts):
    """
    Convert direction parts into a fixed-length list of 4 elements:
    [forward/backward, left/right, rotate_left/right, tilt_up/down]
    
    This ensures consistent input size for the model regardless of command complexity.
    """
    # Initialize fixed-length list with None values
    fixed_direction = [None, None, None, None]
    
    for part in direction_parts:
        if part in ['w', 's']:
            # Forward/backward movement (index 0)
            fixed_direction[0] = part
        elif part in ['a', 'd']:
            # Left/right movement (index 1)
            fixed_direction[1] = part
        elif part in ['arrowleft', 'arrowright']:
            # Rotation (index 2)
            fixed_direction[2] = part
        elif part in ['arrowup', 'arrowdown']:
            # Tilt (index 3)
            fixed_direction[3] = part
        elif part in ['w+a', 'w+d', 's+a', 's+d']:
            # Diagonal movements - split into individual components
            if 'w' in part:
                fixed_direction[0] = 'w'
            elif 's' in part:
                fixed_direction[0] = 's'
            if 'a' in part:
                fixed_direction[1] = 'a'
            elif 'd' in part:
                fixed_direction[1] = 'd'
    
    return fixed_direction


##### keyboard commands #####

def _execute_keyboard_commands(keys, camera_frames, is_neutral, current_leg, intensity):
    global IMAGELESS_GAIT  # set IMAGELESS_GAIT as global to switch between modes via button press

    if 'i' in keys:  # if user wishes to enable/disable imageless gait...
        IMAGELESS_GAIT = not IMAGELESS_GAIT  # toggle imageless gait mode
        logging.warning(f"(control_logic.py): Toggled IMAGELESS_GAIT to {IMAGELESS_GAIT}\n")
        keys = [k for k in keys if k != 'i']  # remove 'i' from the keys list

    # cancel out contradictory keys
    if 'w' in keys and 's' in keys:
        keys = [k for k in keys if k not in ['w', 's']]
    if 'a' in keys and 'd' in keys:
        keys = [k for k in keys if k not in ['a', 'd']]
    if 'arrowleft' in keys and 'arrowright' in keys:
        keys = [k for k in keys if k not in ['arrowleft', 'arrowright']]
    if 'arrowup' in keys and 'arrowdown' in keys:
        keys = [k for k in keys if k not in ['arrowup', 'arrowdown']]

    # build direction parts list (similar to radio commands)
    direction_parts = []

    # movement (WASD and diagonals)
    move_forward = 'w' in keys
    move_backward = 's' in keys
    shift_left = 'a' in keys
    shift_right = 'd' in keys

    # rotation (arrow left/right)
    rotate_left = 'arrowleft' in keys
    rotate_right = 'arrowright' in keys

    # tilt (arrow up/down)
    tilt_up = 'arrowup' in keys
    tilt_down = 'arrowdown' in keys

    # handle diagonal movements (combine w/s with a/d)
    if move_forward and shift_left:
        direction_parts.append('w+a')
    elif move_forward and shift_right:
        direction_parts.append('w+d')
    elif move_backward and shift_left:
        direction_parts.append('s+a')
    elif move_backward and shift_right:
        direction_parts.append('s+d')
    # handle single movements
    elif move_forward:
        direction_parts.append('w')
    elif move_backward:
        direction_parts.append('s')
    elif shift_left:
        direction_parts.append('a')
    elif shift_right:
        direction_parts.append('d')

    # handle rotation (can combine with movement)
    if rotate_left:
        direction_parts.append('arrowleft')
    elif rotate_right:
        direction_parts.append('arrowright')

    # handle tilt (can combine with movement and rotation)
    if tilt_up:
        direction_parts.append('arrowup')
    elif tilt_down:
        direction_parts.append('arrowdown')

    # combine all direction parts into fixed-length list
    direction = None
    if direction_parts:
        direction = _convert_direction_parts_to_fixed_list(direction_parts)

    # neutral and special actions
    if 'n' in keys or not keys:
        neutral_position(10)
        is_neutral = True

    elif direction:
        move_direction(direction, camera_frames, intensity, IMAGELESS_GAIT)
        is_neutral = False
    else:
        logging.warning(f"(control_logic.py): Invalid command: {keys}.\n")

    return is_neutral, current_leg


##### radio commands #####

def _execute_radio_commands(commands, camera_frames, is_neutral, current_leg):
    global IMAGELESS_GAIT  # set IMAGELESS_GAIT as global to switch between modes via button press

    # as radio is currently not reliable enough for individual button pressing, prevent image use and reserve radio as
    # backup control as model-switching is an expensive maneuver that should be avoided unless necessary
    IMAGELESS_GAIT = True

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

    # combine all direction parts into fixed-length list
    if direction_parts:
        direction = _convert_direction_parts_to_fixed_list(direction_parts)
        logging.debug(f"(control_logic.py): Radio commands: ({active_commands}:{direction})\n")
        if special_actions:
            logging.debug(f"(control_logic.py): Special actions: ({special_actions})\n")
        move_direction(direction, camera_frames, 10, IMAGELESS_GAIT) # TODO locking intensity at 10 for now
        is_neutral = False

    elif special_actions:
        # only special actions, no movement
        logging.debug(f"(control_logic.py): Special actions only: ({special_actions})\n")
        # handle special actions here
        if 'SQUAT_DOWN' in special_actions:
            pass
            is_neutral = False

        elif 'SQUAT_UP' in special_actions:
            neutral_position(10)
            is_neutral = True

    return is_neutral, current_leg


########## MISCELLANEOUS CONTROL FUNCTIONS ##########

def restart_process():  # start background thread to restart robot_dog.service every 30 minutes by checking elapsed time
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


########## RUN ROBOTIC PROCESS ##########

if not config.USE_SIMULATION:
    restart_thread = threading.Thread(target=restart_process, daemon=True)
    restart_thread.start()
    # voltage_thread = threading.Thread(target=voltage_monitor, daemon=True)
    # voltage_thread.start()

if not config.USE_SIMULATION and not config.USE_ISAAC_SIM:
    _physical_loop(CHANNEL_DATA)  # run robot process
elif config.USE_SIMULATION and config.USE_ISAAC_SIM:
    _isaac_sim_loop()  # run robot process
else:
    logging.error(f"(control_logic.py): Invalid configuration: {config.USE_SIMULATION} and {config.USE_ISAAC_SIM}\n")
    exit(1)