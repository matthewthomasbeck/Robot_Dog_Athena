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


########## SET REAL ROBOT DEPENDENCIES ##########

def set_real_robot_dependencies():

    ##### import/create dependencies #####

    global CAMERA_PROCESS, CHANNEL_DATA, SOCK, COMMAND_QUEUE, ROBOT_ID, JOINT_MAP, internet
    from utilities.receiver import initialize_receiver  # import receiver initialization functions
    from utilities.camera import initialize_camera  # import to start camera logic
    import utilities.internet as internet  # dynamically import internet utilities to be constantly updated

    CAMERA_PROCESS = initialize_camera()  # create camera process
    if CAMERA_PROCESS is None:
        logging.error("(control_logic.py): Failed to initialize CAMERA_PROCESS for robot!\n")

    if config.CONTROL_MODE == 'web': # if web control mode and robot needs a socket connection for controls and video...
        SOCK = internet.initialize_backend_socket()  # initialize EC2 socket connection
        COMMAND_QUEUE = internet.initialize_command_queue(SOCK)  # initialize command queue for socket communication
        if SOCK is None:
            logging.error("(control_logic.py): Failed to initialize SOCK for robot!\n")
        if COMMAND_QUEUE is None:
            logging.error("(control_logic.py): Failed to initialize COMMAND_QUEUE for robot!\n")

    elif config.CONTROL_MODE == 'radio':  # if radio control mode...
        CHANNEL_DATA = initialize_receiver()  # get pigpio instance, decoders, and channel data
        if CHANNEL_DATA == None:
            logging.error("(control_logic.py): Failed to initialize CHANNEL_DATA for robot!\n")


########## SET SIMULATED ROBOT DEPENDENCIES ##########

def set_isaac_dependencies():

    global CAMERA_PROCESS, CHANNEL_DATA, SOCK, COMMAND_QUEUE
    import sys
    import carb
    import numpy

    # IMPORTANT 'SimulationApp' MUST be imported and made before any other isaac utilization of any kind!!!
    from isaacsim.simulation_app import SimulationApp
    config.ISAAC_SIM_APP = SimulationApp({"headless": False})
    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
    from training.isaac_sim import build_isaac_joint_index_map

    config.ISAAC_WORLD = World(stage_units_in_meters=1.0)
    
    # Add ground plane
    config.ISAAC_WORLD.scene.add_default_ground_plane()
    
    # Add robot
    usd_path = os.path.expanduser("/home/matthewthomasbeck/Projects/Robot_Dog/training/urdf/robot_dog/robot_dog.usd")
    add_reference_to_stage(usd_path, "/World/robot_dog")
    
    # Move robot up 30cm using the EXACT same method as the compass rose
    try:
        import omni.usd
        from pxr import UsdGeom, Gf
        
        # Get the stage the same way we do in create_coordinate_frames
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath("/World/robot_dog")
        
        if robot_prim:
            # Robot already has transforms, so we need to modify the existing translate
            xform = UsdGeom.Xformable(robot_prim)
            if xform:
                # Get existing transform operations
                xform_ops = xform.GetOrderedXformOps()
                
                # Find the translate operation
                translate_op = None
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                
                if translate_op:
                    # Get current translate value and add 30cm to Z
                    current_translate = translate_op.Get()
                    new_translate = Gf.Vec3d(current_translate[0], current_translate[1], current_translate[2] + 0.14)
                    translate_op.Set(new_translate)
                    logging.info(f"(control_logic.py): Robot moved up 30cm. Old pos: {current_translate}, New pos: {new_translate}\n")
                else:
                    logging.warning("(control_logic.py): No translate operation found on robot prim.\n")
            else:
                logging.warning("(control_logic.py): Robot prim found but not Xformable.\n")
        else:
            logging.warning("(control_logic.py): Robot prim not found at /World/robot_dog.\n")
            
    except Exception as e:
        logging.error(f"(control_logic.py): Failed to move robot up: {e}\n")
        # Continue anyway - robot will spawn at default height
    
    for _ in range(3): # let isaac sim load a few steps for general process
        config.ISAAC_WORLD.step(render=True)
    config.ISAAC_ROBOT = Articulation(prim_paths_expr="/World/robot_dog", name="robot_dog")
    config.ISAAC_WORLD.scene.add(config.ISAAC_ROBOT)
    
    # For now, let's just use the default spawn position and see if the robot clipping issue persists
    # We can revisit the height adjustment once we understand the Isaac Sim 5 API better
    logging.info("(control_logic.py): Robot added to Isaac Sim scene.\n")
    
    config.ISAAC_WORLD.reset()
    config.JOINT_INDEX_MAP = build_isaac_joint_index_map(config.ISAAC_ROBOT.dof_names)
    logging.info(f"(control_logic.py) Isaac Sim initialized using SERVO_CONFIG for joint mapping. Joint map: {config.JOINT_INDEX_MAP}\n")

    # Add coordinate frames for orientation tracking
    from training.isaac_sim import create_coordinate_frames
    create_coordinate_frames()
    logging.info("(control_logic.py) Coordinate frames created for robot orientation tracking.\n")

    from utilities.camera import initialize_camera  # import to start camera logic
    CAMERA_PROCESS = initialize_camera()  # create camera process
    if CAMERA_PROCESS is None:
        logging.error("(control_logic.py): Failed to initialize CAMERA_PROCESS for isaac sim!\n")

    # Create RL command queue for Isaac Sim (similar to web command queue)
    try:
        COMMAND_QUEUE = queue.Queue()  # Create RL command queue
        logging.info("(control_logic.py): RL command queue initialized successfully for Isaac Sim.\n")
    except Exception as e:
        logging.error(f"(control_logic.py): Failed to initialize RL command queue: {e}\n")
        COMMAND_QUEUE = None

    # TODO dont remove the commented out code below, I may need it someday
    #if config.CONTROL_MODE == 'web': # if web control mode and robot needs a socket connection for controls and video...
        #SOCK = internet.initialize_backend_socket()  # initialize EC2 socket connection
        #COMMAND_QUEUE = internet.initialize_command_queue(SOCK)  # initialize command queue for socket communication
        #if SOCK is None:
            #logging.error("(control_logic.py): Failed to initialize SOCK for robot!\n")
        #if COMMAND_QUEUE is None:
            #logging.error("(control_logic.py): Failed to initialize COMMAND_QUEUE for robot!\n")

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
    from training.training import set_simulation_variables
    set_simulation_variables(ROBOT_ID, JOINT_MAP)

##### import/create necessary dependencies based on detected environment #####

if not config.USE_SIMULATION:
    set_real_robot_dependencies()
elif config.USE_SIMULATION:
    if config.USE_ISAAC_SIM:
        set_isaac_dependencies()
    elif not config.USE_ISAAC_SIM:
        set_pybullet_dependencies()

##### post-initialization dependencies #####

from movement.fundamental_movement import *
from utilities.camera import decode_real_frame, decode_isaac_frame

# Import Isaac Sim specific functions
if config.USE_SIMULATION and config.USE_ISAAC_SIM:
    from training.isaac_sim import process_isaac_movement_queue





#########################################
############### RUN ROBOT ###############
#########################################


########## RUN ROBOTIC PROCESS ##########

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

    if config.USE_SIMULATION and config.USE_ISAAC_SIM: # if isaac sim...
        for _ in range(3):  # let isaac sim load a few steps for general process
            config.ISAAC_WORLD.step(render=True)

    try:  # try to run robot startup sequence
        neutral_position(1)
        time.sleep(3)
        IS_NEUTRAL = True  # set is_neutral to True

    except Exception as e:  # if there is an error, log error
        logging.error(f"(control_logic.py): Failed to move to neutral standing position in runRobot: {e}\n")

    ##### stream video, run inference, and control the robot #####

    try:  # try to run main robotic process

        while True:  # central loop to entire process, commenting out of importance

            if not config.USE_SIMULATION and not config.USE_ISAAC_SIM:  # if physical robot...
                mjpeg_buffer, streamed_frame, inference_frame = decode_real_frame( # run camera and decode frame
                    CAMERA_PROCESS,
                    mjpeg_buffer
                )
                command = None  # initially no command

            # Isaac Sim - collect frames and check for RL agent commands
            if config.USE_SIMULATION and config.USE_ISAAC_SIM:  # if isaac sim...

                mjpeg_buffer, streamed_frame, inference_frame = decode_isaac_frame(  # run camera and decode frame
                    CAMERA_PROCESS
                )
                command = None  # initially no command
                
                # Generate RL commands if queue is empty and robot is ready for new commands
                if COMMAND_QUEUE is not None and COMMAND_QUEUE.empty() and IS_COMPLETE:
                    from training.isaac_sim import get_rl_command_with_intensity, inject_rl_command_into_queue
                    command, intensity = get_rl_command_with_intensity()
                    inject_rl_command_into_queue(COMMAND_QUEUE, command, intensity)
                    #logging.debug(f"(control_logic.py): Generated RL command: {command} with intensity {intensity}\n")
                
                # Check RL command queue for Isaac Sim
                if COMMAND_QUEUE is not None and not COMMAND_QUEUE.empty():
                    command_data = COMMAND_QUEUE.get()  # Get command data from RL queue
                    if command_data is not None:
                        # Extract command and intensity from RL command data
                        rl_command = command_data.get('command')
                        rl_intensity = command_data.get('intensity')  # Get intensity from RL agent
                        
                        # Convert RL command format to web command format for consistency
                        if rl_command is None:
                            command = 'n'  # Neutral command
                        else:
                            command = rl_command  # Use RL command directly
                        
                        # Store intensity for later use (should always be present from RL agent)
                        if rl_intensity is not None:
                            config.RL_COMMAND_INTENSITY = rl_intensity
                        else:
                            # Fallback if intensity is missing (shouldn't happen)
                            config.RL_COMMAND_INTENSITY = 5
                            logging.warning(f"(control_logic.py): RL command missing intensity, using default 5\n")
                        
                        #if IS_COMPLETE:
                            #logging.info(f"(control_logic.py): Received RL command '{command}' with intensity {config.RL_COMMAND_INTENSITY} (WILL RUN).\n")
                        #else:
                            #logging.info(f"(control_logic.py): Received RL command '{command}' with intensity {config.RL_COMMAND_INTENSITY} (BLOCKED).\n")

                # TODO get start pose of robot, may need to move to foundational movement
                #prev_pose = get_world_pose(get_prim_at_path('/World/my_robot/base_link'))

            if config.CONTROL_MODE == 'web': # if web control enabled...

                if not config.USE_SIMULATION and not config.USE_ISAAC_SIM: # if physical robot...
                    internet.stream_to_backend(SOCK, streamed_frame)  # stream frame data to backend
                #internet.stream_to_backend(SOCK, streamed_frame)

                if COMMAND_QUEUE is not None and not COMMAND_QUEUE.empty(): # if command queue is not empty...
                    command = COMMAND_QUEUE.get() # get command from queue
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

            # WEB COMMAND HANDLING (includes RL commands for Isaac Sim)
            if command and IS_COMPLETE: # if command present and movement complete...
                #logging.debug(f"(control_logic.py): Running command: {command}...\n")
                threading.Thread(target=_handle_command, args=(command, inference_frame), daemon=True).start()

            # NEUTRAL POSITION HANDLING (for both modes)
            elif not command and IS_COMPLETE and not IS_NEUTRAL: # if no command and movement complete and not neutral...
                logging.debug(f"(control_logic.py): No command received, returning to neutral position...\n")
                threading.Thread(target=_handle_command, args=('n', inference_frame), daemon=True).start()

            # step simulation if enabled
            if config.USE_SIMULATION:
                if config.USE_ISAAC_SIM:
                    config.ISAAC_WORLD.step(render=True)
                    
                    # Process queued movements for Isaac Sim (avoid PhysX threading violations)
                    process_isaac_movement_queue()

                else:
                    pybullet.stepSimulation()

    except KeyboardInterrupt:  # if user ends program...
        logging.info("(control_logic.py): KeyboardInterrupt received, exiting.\n")

    except Exception as e:  # if something breaks and only God knows what it is...
        logging.error(f"(control_logic.py): Unexpected exception in main loop: {e}\n")
        exit(1)


########## HANDLE COMMANDS ##########

def _handle_command(command, frame):

    #logging.debug(f"(control_logic.py): Threading command: {command}...\n")

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
                CURRENT_LEG
            )
            logging.info(f"(control_logic.py): Executed radio commands: {command}\n")
            IS_COMPLETE = True
        except Exception as e:
            logging.error(f"(control_logic.py): Failed to execute radio commands: {e}\n")
            IS_NEUTRAL = False
            IS_COMPLETE = True

    elif config.CONTROL_MODE == 'web':
        try:
            #logging.debug(f"(control_logic.py): Executing keyboard command: {keys}\n")
            
            # Use RL intensity if available (for Isaac Sim), otherwise use default
            if config.USE_SIMULATION and config.USE_ISAAC_SIM and hasattr(config, 'RL_COMMAND_INTENSITY'):
                intensity = config.RL_COMMAND_INTENSITY
                # Clear the RL intensity after use
                delattr(config, 'RL_COMMAND_INTENSITY')
            else:
                intensity = config.DEFAULT_INTENSITY
            
            IS_NEUTRAL, CURRENT_LEG = _execute_keyboard_commands(
                keys,
                frame,
                IS_NEUTRAL,
                CURRENT_LEG,
                intensity
            )
            #logging.info(f"(control_logic.py): Executed keyboard command: {keys}\n")
            IS_COMPLETE = True
        except Exception as e:
            logging.error(f"(control_logic.py): Failed to execute keyboard command: {e}\n")
            IS_NEUTRAL = False
            IS_COMPLETE = True

##### keyboard commands for tuning mode and normal operation #####

def _execute_keyboard_commands(keys, frame, is_neutral, current_leg, intensity):

    global IMAGELESS_GAIT  # set IMAGELESS_GAIT as global to switch between modes via button press

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

    # combine all direction parts
    direction = None
    if direction_parts:
        direction = '+'.join(direction_parts)

    # neutral and special actions
    if 'n' in keys or not keys:
        neutral_position(10)
        is_neutral = True
    elif direction:
        #logging.debug(f"(control_logic.py): {keys}: {direction}\n")
        # Use trot_forward for all modes (now supports Isaac Sim queue system)
        move_direction(direction, frame, intensity, IMAGELESS_GAIT)
        #calibrate_joints_isaac()
        is_neutral = False
    else:
        logging.warning(f"(control_logic.py): Invalid command: {keys}.\n")

    return is_neutral, current_leg


##### radio commands for multi-channel processing #####

def _execute_radio_commands(commands, frame, is_neutral, current_leg):

    global IMAGELESS_GAIT # set IMAGELESS_GAIT as global to switch between modes via button press

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

    # combine all direction parts
    if direction_parts:
        direction = '+'.join(direction_parts)
        logging.debug(f"(control_logic.py): Radio commands: ({active_commands}:{direction})\n")
        if special_actions:
            logging.debug(f"(control_logic.py): Special actions: ({special_actions})\n")
        #move_direction(direction, frame, max_intensity, IMAGELESS_GAIT)
        move_direction(max_intensity)
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


########## ROBOTIC PROCESS ##########

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
