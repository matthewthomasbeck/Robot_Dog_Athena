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

from pxr import Gf

##### global variables #####

# Global variables for tracking robot states
PREV_ROBOT_POSITION = None
REWARD_TRACKING_ENABLED = False
EPISODE_COUNTER = 0  # Track episode number
EPISODE_STEP_COUNT = 0  # Track steps within current episode
TOTAL_FORWARD_PROGRESS = 0.0  # Track total forward progress across episodes

# Global variables for SB3 training
PPO_MODEL = None
MODEL_INITIALIZED = False
EXPERIENCE_BUFFER = []
LAST_OBSERVATION = None
LAST_ACTION = None
CURRENT_OBSERVATION = None
CURRENT_ACTION = None
TRAINING_STEP_COUNT = 0





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


########## RL COMMAND GENERATION ##########

def generate_rl_command(): # TODO legacy command to swap with apply_joint_angles_isaac()
    """
    Generate a command string for the RL agent to execute.
    Returns command strings compatible with _execute_keyboard_commands format.
    
    Returns:
        str: Command string like 'w', 'w+d', 'w+d+arrowleft+arrowup', or None for neutral
    
    Examples of possible outputs:
        - 'w' (forward)
        - 's+a' (backward + left)
        - 'w+d+arrowleft' (forward + right + rotate left)
        - 'w+d+arrowleft+arrowup' (forward + right + rotate left + tilt up)
        - None (neutral/no action)
    """
    import random
    
    # Define possible actions with their probabilities
    movement_actions = ['w', 's', 'a', 'd']  # forward, backward, left, right
    rotation_actions = ['arrowleft', 'arrowright']  # rotate left, rotate right
    tilt_actions = ['arrowup', 'arrowdown']  # tilt up, tilt down
    
    # Probability of including each type of action
    movement_prob = 0.7  # 70% chance of movement
    rotation_prob = 0.3  # 30% chance of rotation (can combine with movement)
    tilt_prob = 0.2     # 20% chance of tilt (can combine with movement/rotation)
    neutral_prob = 0.1  # 10% chance of neutral (no action)
    
    # Check for neutral action first
    if random.random() < neutral_prob:
        return None  # Neutral command
    
    command_parts = []
    
    # Add movement action
    if random.random() < movement_prob:
        # Decide between single movement or diagonal
        if random.random() < 0.3:  # 30% chance of diagonal movement
            # Choose two non-contradictory movements
            forward_back = random.choice(['w', 's'])
            left_right = random.choice(['a', 'd'])
            command_parts.extend([forward_back, left_right])
        else:
            # Single movement
            command_parts.append(random.choice(movement_actions))
    
    # Add rotation action (can combine with movement)
    if random.random() < rotation_prob:
        command_parts.append(random.choice(rotation_actions))
    
    # Add tilt action (can combine with movement and rotation)
    if random.random() < tilt_prob:
        command_parts.append(random.choice(tilt_actions))
    
    # If no actions were selected, return neutral
    if not command_parts:
        return None
    
    # Join parts with '+' to create command string
    return '+'.join(command_parts)


def get_rl_command_with_intensity(): # TODO currently only injects 'w' of intensity 10, keep commented logic
    """
    Generate both a command and intensity for the RL agent.
    
    Returns:
        tuple: (command_string, intensity_value)
            command_string: str or None (compatible with _execute_keyboard_commands)
            intensity_value: int between 1 and 10 (always 10 for neutral commands)
    """
    import random
    
    # SIMPLIFIED FOR FORWARD WALKING TRAINING - Only generate 'w' commands
    # TODO: Restore full command generation when ready for complex training
    
    # 90% chance of forward command, 10% chance of neutral
    if random.random() < 0.9:
        return 'w', 10  # Forward with max intensity
    else:
        return None, 10  # Neutral command
    
    # ORIGINAL FULL COMMAND GENERATION (commented out for now):
    # command = generate_rl_command()
    # 
    # # Special case: neutral commands always have intensity 10 (fast return to neutral)
    # if command is None:
    #     return command, 10
    # 
    # # Generate intensity between 1 and 10 (integers only) with varied randomness for movement commands
    # rand_value = random.random()
    # 
    # if rand_value < 0.15:  # 15% chance of low intensity (1-3)
    #     intensity = random.randint(1, 3)
    # elif rand_value < 0.70:  # 55% chance of moderate intensity (4-7)
    #     intensity = random.randint(4, 7)
    # else:  # 30% chance of high intensity (8-10)
    #     intensity = random.randint(8, 10)
    # 
    # return command, intensity


def inject_rl_command_into_queue(rl_command_queue, command, intensity):
    """
    Inject an RL-generated command into the RL command queue.
    This follows the same pattern as web commands being injected into COMMAND_QUEUE.
    
    Args:
        rl_command_queue: Queue object for RL commands
        command: str or None - Command string like 'w+d+arrowleft' or None for neutral
        intensity: int - Movement intensity (1 to 10)
    """
    if rl_command_queue is not None:
        # Package command with intensity (similar to web command format)
        command_data = {
            'command': command,
            'intensity': intensity
        }
        rl_command_queue.put(command_data)


########## COORDINATE FRAME SYSTEM ##########

def create_coordinate_frames():
    """
    Create visual coordinate frames for robot orientation tracking:
    1. Static world reference frame (fixed NSEW compass)
    2. Robot-attached local frame (moves with robot)
    """
    import utilities.config as config
    from pxr import UsdGeom, Gf, UsdShade
    import omni.usd
    
    # Get the current stage
    stage = omni.usd.get_context().get_stage()
    
    # Create static world reference frame (compass rose at origin, elevated above ground)
    world_frame_path = "/World/WorldReferenceFrame"
    world_frame_prim = UsdGeom.Xform.Define(stage, world_frame_path)
    
    # Position the world frame 30cm above ground
    world_frame_prim.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.30))
    
    # Create compass rose arrows for world frame
    _create_compass_arrows(stage, world_frame_path, scale=1.0, color=(1.0, 1.0, 1.0))  # White arrows
    

    
    # Store frame paths in config for later access
    config.WORLD_FRAME_PATH = world_frame_path
    config.ROBOT_BODY_PATH = "/World/robot_dog"  # Robot itself serves as its own reference frame


def _create_compass_arrows(stage, parent_path, scale=1.0, color=(1.0, 1.0, 1.0)):
    """
    Create compass rose arrows (N, S, E, W) under the given parent path.
    Args:
        stage: USD stage
        parent_path: Parent prim path
        scale: Size scaling factor
        color: RGB color tuple
    """
    from pxr import UsdGeom, Gf, UsdShade
    
    # Define arrow directions with unique colors for easy identification
    arrows = [
        ("North", (1.0, 0.0, 0.0), "X+", (1.0, 0.0, 0.0)),   # Forward/North = +X, RED
        ("South", (-1.0, 0.0, 0.0), "X-", (0.0, 1.0, 0.0)),  # Backward/South = -X, GREEN  
        ("East", (0.0, -1.0, 0.0), "Y-", (0.0, 0.0, 1.0)),   # Right/East = -Y, BLUE
        ("West", (0.0, 1.0, 0.0), "Y+", (1.0, 1.0, 0.0)),    # Left/West = +Y, YELLOW
        ("Up", (0.0, 0.0, 1.0), "Z+", (1.0, 0.0, 1.0))       # Up = +Z, MAGENTA
    ]
    
    for name, direction, label, arrow_color in arrows:
        # Create arrow as a cylinder (more visible than curves)
        arrow_path = f"{parent_path}/{name}Arrow"
        arrow_prim = UsdGeom.Cylinder.Define(stage, arrow_path)
        
        # Set cylinder properties (10cm long, 1cm thick)
        arrow_prim.GetRadiusAttr().Set(0.005)  # 1cm radius = 1cm thick
        arrow_prim.GetHeightAttr().Set(0.1)    # 10cm long
        arrow_prim.GetAxisAttr().Set("Z")      # Default axis is Z
        
        # Calculate position and rotation for each arrow
        arrow_length = 0.1  # 10cm
        dx, dy, dz = direction
        
        # Position arrows to extend FROM center outward (not centered AT center)
        # Each arrow's back end should touch the center, front end extends outward
        if name == "North":  # RED - should extend in +X direction
            mid_point = Gf.Vec3f(arrow_length * 0.5, 0.0, -0.05)  # Center of cylinder at +5cm X
            rotation = Gf.Rotation(Gf.Vec3d(0, 1, 0), 90)  # Rotate to point in +X
        elif name == "South":  # GREEN - should extend in -X direction
            mid_point = Gf.Vec3f(-arrow_length * 0.5, 0.0, -0.05)  # Center of cylinder at -5cm X
            rotation = Gf.Rotation(Gf.Vec3d(0, 1, 0), -90)  # Rotate to point in -X
        elif name == "East":  # BLUE - should extend in -Y direction
            mid_point = Gf.Vec3f(0.0, -arrow_length * 0.5, -0.05)  # Center of cylinder at -5cm Y
            rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), 90)  # Rotate to point in -Y
        elif name == "West":  # YELLOW - should extend in +Y direction
            mid_point = Gf.Vec3f(0.0, arrow_length * 0.5, -0.05)  # Center of cylinder at +5cm Y
            rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)  # Rotate to point in +Y
        else:  # "Up" - MAGENTA - should extend in +Z direction
            mid_point = Gf.Vec3f(0.0, 0.0, arrow_length * 0.05)  # Center of cylinder at +5cm Z
            rotation = Gf.Rotation()  # No rotation needed for Z-axis
        
        # Apply transform
        transform_matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(mid_point)) * Gf.Matrix4d().SetRotate(rotation)
        arrow_prim.AddTransformOp().Set(transform_matrix)
        
        # Set color using displayColor (use individual arrow color, not the passed color parameter)
        arrow_prim.GetDisplayColorAttr().Set([arrow_color])


########## ISAAC SIM QUEUE PROCESSING ##########

def process_isaac_movement_queue():
    """
    Process queued movements for Isaac Sim in the main thread to avoid PhysX violations.
    This function should be called from the main loop after each simulation step.
    """
    if not config.USE_SIMULATION or not config.USE_ISAAC_SIM:
        return
    
    # Import here to avoid circular imports
    from movement.fundamental_movement import ISAAC_MOVEMENT_QUEUE, ISAAC_CALIBRATION_COMPLETE, _apply_single_joint_position_isaac
    import queue
    
    # Process any new movement data
    while not ISAAC_MOVEMENT_QUEUE.empty():
        try:
            movement_data = ISAAC_MOVEMENT_QUEUE.get_nowait()
            
            # Check if this is calibration data
            if isinstance(movement_data, dict) and movement_data.get('type') == 'calibration':
                # Apply the joint position directly
                joint_name = movement_data['joint_name']
                angle_rad = movement_data['angle_rad']
                velocity = movement_data.get('velocity', 0.5)  # Default to 0.5 if not specified
                _apply_single_joint_position_isaac(joint_name, angle_rad, velocity)
                
                # Signal that this calibration movement is complete
                ISAAC_CALIBRATION_COMPLETE.set()
                
        except queue.Empty:
            break
