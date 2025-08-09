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
    delta_pos = curr_pos - prev_pos  # (dx, dy, dz)

    # orientation delta
    prev_rot = Gf.Quatf(*previous_pose[1])  # (w, x, y, z)
    curr_rot = Gf.Quatf(*current_pose[1])
    delta_rot = curr_rot * prev_rot.GetInverse()
    delta_yaw = delta_rot.GetAxisAngle()[0] * delta_rot.GetAxisAngle()[1][2]  # yaw around Z

    reward = 0.0
    command_keys = command.split('+') if isinstance(command, str) else []

    # Translational movements
    if 'w' in command_keys:
        reward += delta_pos[0] * intensity  # +X is forward
    if 's' in command_keys:
        reward -= delta_pos[0] * intensity
    if 'a' in command_keys:
        reward += delta_pos[1] * intensity  # +Y is left
    if 'd' in command_keys:
        reward -= delta_pos[1] * intensity

    # Rotational movements
    if 'arrowleft' in command_keys:
        reward += delta_yaw * intensity
    if 'arrowright' in command_keys:
        reward -= delta_yaw * intensity

    # Tilting (Up/Down): reward changes in pitch via z-axis body angle or delta_z
    delta_z = delta_pos[2]

    if 'arrowup' in command_keys:
        reward += delta_z * intensity  # raise front / tilt up
    if 'arrowdown' in command_keys:
        reward -= delta_z * intensity  # lower front / tilt down

    # Penalty for unintended drift if no command is active
    if not command_keys:
        noise_penalty = numpy.linalg.norm(delta_pos[:2]) * 0.05
        reward -= noise_penalty

    return reward


########## SIMULATION VARIABLES ##########

def set_simulation_variables(robot_id, joint_map):
    """
    Set global simulation variables for PyBullet.
    Args:
        robot_id: PyBullet robot body ID
        joint_map: Dictionary mapping joint names to indices
    """
    global ROBOT_ID, JOINT_MAP
    ROBOT_ID = robot_id
    JOINT_MAP = joint_map


########## RL AGENT ACTION FUNCTIONS ##########

def get_rl_action_standard(state, commands, intensity, frame):
    """
    Placeholder for RL agent's policy WITH image input (standard gait adjustment).
    Args:
        state: The current state of the robot/simulation (to be defined).
        commands: The movement commands.
        intensity: The movement intensity.
        frame: The current image frame (for vision-based agent).
    Returns:
        target_angles: dict of target joint angles for each leg (similar to SERVO_CONFIG structure).
        mid_angles: dict of mid joint angles for each leg (similar to SERVO_CONFIG structure).
        movement_rates: dict of movement rate parameters for each leg.
    TODO: Replace this with a call to your RL agent's policy/model (with image input).
    """
    # For now, just return the current angles as both mid and target
    target_angles = {}
    mid_angles = {}
    movement_rates = {}
    
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        target_angles[leg_id] = {}
        mid_angles[leg_id] = {}
        movement_rates[leg_id] = {'speed': 1000, 'acceleration': 255}
        
        for joint_name in ['hip', 'upper', 'lower']:
            current_angle = config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE']
            target_angles[leg_id][joint_name] = current_angle
            mid_angles[leg_id][joint_name] = current_angle
    
    return target_angles, mid_angles, movement_rates


def get_rl_action_blind(state, commands, intensity):
    """
    Placeholder for RL agent's policy WITHOUT image input (imageless gait adjustment).
    Args:
        state: The current state of the robot/simulation (to be defined).
        commands: The movement commands.
        intensity: The movement intensity.
    Returns:
        target_angles: dict of target joint angles for each leg (similar to SERVO_CONFIG structure).
        mid_angles: dict of mid joint angles for each leg (similar to SERVO_CONFIG structure).
        movement_rates: dict of movement rate parameters for each leg.
    TODO: Replace this with a call to your RL agent's policy/model (no image input).
    """
    # For now, just return the current angles as both mid and target (no movement)
    target_angles = {}
    mid_angles = {}
    movement_rates = {}
    
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        target_angles[leg_id] = {}
        mid_angles[leg_id] = {}
        movement_rates[leg_id] = {'speed': 1000, 'acceleration': 255}
        
        for joint_name in ['hip', 'upper', 'lower']:
            current_angle = config.SERVO_CONFIG[leg_id][joint_name]['CURRENT_ANGLE']
            target_angles[leg_id][joint_name] = current_angle
            mid_angles[leg_id][joint_name] = current_angle
    
    return target_angles, mid_angles, movement_rates


########## RL COMMAND GENERATION ##########

def generate_rl_command():
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


def get_rl_command_with_intensity():
    """
    Generate both a command and intensity for the RL agent.
    
    Returns:
        tuple: (command_string, intensity_value)
            command_string: str or None (compatible with _execute_keyboard_commands)
            intensity_value: int between 1 and 10 (always 10 for neutral commands)
    """
    import random
    
    command = generate_rl_command()
    
    # Special case: neutral commands always have intensity 10 (fast return to neutral)
    if command is None:
        return command, 10
    
    # Generate intensity between 1 and 10 (integers only) with varied randomness for movement commands
    rand_value = random.random()
    
    if rand_value < 0.15:  # 15% chance of low intensity (1-3)
        intensity = random.randint(1, 3)
    elif rand_value < 0.70:  # 55% chance of moderate intensity (4-7)
        intensity = random.randint(4, 7)
    else:  # 30% chance of high intensity (8-10)
        intensity = random.randint(8, 10)
    
    return command, intensity


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




def run_rl_episode_step(rl_command_queue):
    """
    Execute one step of RL training by generating a command and putting it in the queue.
    This follows the same pattern as web commands.
    
    Args:
        rl_command_queue: Queue object for RL commands
    
    Returns:
        dict: Episode step information including:
            - command: The generated command
            - intensity: The command intensity
    """
    # Generate RL command and intensity
    command, intensity = get_rl_command_with_intensity()
    
    # Put command into queue (same as web command flow)
    inject_rl_command_into_queue(rl_command_queue, command, intensity)
    
    # Return step information for RL training
    return {
        'command': command,
        'intensity': intensity
    }


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
    
    # TODO: Robot-attached local frame (disabled for now - has positioning issues)
    # robot_frame_path = "/World/robot_dog/RobotReferenceFrame"
    # robot_frame_prim = UsdGeom.Xform.Define(stage, robot_frame_path)
    # _create_compass_arrows(stage, robot_frame_path, scale=0.5, color=(1.0, 0.0, 0.0))  # Red arrows
    
    # Store frame paths in config for later access
    config.WORLD_FRAME_PATH = world_frame_path
    # config.ROBOT_FRAME_PATH = robot_frame_path  # Disabled for now
    config.ROBOT_BODY_PATH = "/World/robot_dog"


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
            mid_point = Gf.Vec3f(arrow_length * 0.5, 0.0, 0.05)  # Center of cylinder at +5cm X
            rotation = Gf.Rotation(Gf.Vec3d(0, 1, 0), 90)  # Rotate to point in +X
        elif name == "South":  # GREEN - should extend in -X direction
            mid_point = Gf.Vec3f(-arrow_length * 0.5, 0.0, 0.05)  # Center of cylinder at -5cm X
            rotation = Gf.Rotation(Gf.Vec3d(0, 1, 0), -90)  # Rotate to point in -X
        elif name == "East":  # BLUE - should extend in -Y direction
            mid_point = Gf.Vec3f(0.0, -arrow_length * 0.5, 0.05)  # Center of cylinder at -5cm Y
            rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), 90)  # Rotate to point in -Y
        elif name == "West":  # YELLOW - should extend in +Y direction
            mid_point = Gf.Vec3f(0.0, arrow_length * 0.5, 0.05)  # Center of cylinder at +5cm Y
            rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)  # Rotate to point in +Y
        else:  # "Up" - MAGENTA - should extend in +Z direction
            mid_point = Gf.Vec3f(0.0, 0.0, arrow_length * 0.05)  # Center of cylinder at +5cm Z
            rotation = Gf.Rotation()  # No rotation needed for Z-axis
        
        # Apply transform
        transform_matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(mid_point)) * Gf.Matrix4d().SetRotate(rotation)
        arrow_prim.AddTransformOp().Set(transform_matrix)
        
        # Set color using displayColor (use individual arrow color, not the passed color parameter)
        arrow_prim.GetDisplayColorAttr().Set([arrow_color])


########## ROBOT POSE EXTRACTION ##########

def get_robot_pose():
    """
    Extract robot's current position and orientation from Isaac Sim.
    Returns:
        tuple: ((x, y, z), (qw, qx, qy, qz)) - position and quaternion
    """
    import utilities.config as config
    from omni.isaac.core.utils.prims import get_prim_at_path
    from omni.isaac.core.utils.transformations import get_world_pose
    
    try:
        # Get robot body prim
        robot_prim = get_prim_at_path(config.ROBOT_BODY_PATH)
        if robot_prim is None:
            raise RuntimeError(f"Robot prim not found at {config.ROBOT_BODY_PATH}")
        
        # Get world pose (position and rotation)
        position, rotation = get_world_pose(config.ROBOT_BODY_PATH)
        
        # Convert to standard format: position as tuple, rotation as quaternion tuple
        pos_tuple = (float(position[0]), float(position[1]), float(position[2]))
        rot_tuple = (float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3]))  # (w, x, y, z)
        
        return pos_tuple, rot_tuple
        
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Failed to get robot pose: {e}")
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)  # Default pose


def get_robot_transform_matrix():
    """
    Get robot's transform matrix for extracting local coordinate axes.
    Returns:
        numpy.ndarray: 4x4 transform matrix, or None if failed
    """
    import numpy as np
    import utilities.config as config
    from omni.isaac.core.utils.transformations import get_world_pose
    from scipy.spatial.transform import Rotation
    
    try:
        position, rotation = get_world_pose(config.ROBOT_BODY_PATH)
        
        # Convert quaternion to rotation matrix
        quat_wxyz = [rotation[0], rotation[1], rotation[2], rotation[3]]  # (w, x, y, z)
        r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # scipy wants (x, y, z, w)
        rot_matrix = r.as_matrix()
        
        # Build 4x4 transform matrix
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = [position[0], position[1], position[2]]
        
        return transform
        
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Failed to get robot transform matrix: {e}")
        return None


def get_robot_local_axes():
    """
    Extract robot's local coordinate axes in world space.
    Returns:
        dict: {
            'forward': numpy.array([x, y, z]),  # Robot's forward direction (+X local)
            'left': numpy.array([x, y, z]),     # Robot's left direction (+Y local)  
            'up': numpy.array([x, y, z])        # Robot's up direction (+Z local)
        }
    """
    import numpy as np
    
    transform = get_robot_transform_matrix()
    if transform is None:
        # Return default axes if transform failed
        return {
            'forward': np.array([1.0, 0.0, 0.0]),
            'left': np.array([0.0, 1.0, 0.0]),
            'up': np.array([0.0, 0.0, 1.0])
        }
    
    # Extract local axes from rotation matrix
    return {
        'forward': transform[:3, 0],  # First column = local +X = forward
        'left': transform[:3, 1],     # Second column = local +Y = left
        'up': transform[:3, 2]        # Third column = local +Z = up
    }


########## ROBOT STATE DETECTION ##########

def is_robot_fallen(tilt_threshold_degrees=45.0):
    """
    Check if robot has fallen over by comparing its up vector to world up.
    Args:
        tilt_threshold_degrees: Maximum allowed tilt before considering fallen
    Returns:
        bool: True if robot is considered fallen
    """
    import numpy as np
    
    # Get robot's up vector
    axes = get_robot_local_axes()
    robot_up = axes['up']
    world_up = np.array([0.0, 0.0, 1.0])
    
    # Calculate angle between robot up and world up
    dot_product = np.dot(robot_up, world_up)
    # Clamp to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg > tilt_threshold_degrees


def calculate_forward_progress(previous_pose, current_pose, robot_forward_direction):
    """
    Calculate how much the robot moved in its intended forward direction.
    Args:
        previous_pose: tuple ((x, y, z), (qw, qx, qy, qz)) from previous timestep
        current_pose: tuple ((x, y, z), (qw, qx, qy, qz)) from current timestep  
        robot_forward_direction: numpy.array([x, y, z]) - robot's current forward vector
    Returns:
        float: Distance moved in forward direction (positive = forward, negative = backward)
    """
    import numpy as np
    
    # Calculate position change
    prev_pos = np.array(previous_pose[0])
    curr_pos = np.array(current_pose[0])
    movement_vector = curr_pos - prev_pos
    
    # Project movement onto robot's forward direction
    forward_progress = np.dot(movement_vector, robot_forward_direction)
    
    return float(forward_progress)


def calculate_movement_efficiency(previous_pose, current_pose, intended_direction_vector):
    """
    Calculate how efficiently the robot moved toward its intended direction.
    Args:
        previous_pose: tuple ((x, y, z), (qw, qx, qy, qz)) from previous timestep
        current_pose: tuple ((x, y, z), (qw, qx, qy, qz)) from current timestep
        intended_direction_vector: numpy.array([x, y, z]) - normalized intended movement direction
    Returns:
        dict: {
            'alignment': float,      # How aligned actual movement was with intended (0-1)
            'magnitude': float,      # Total distance moved
            'efficiency': float      # Combined efficiency score (0-1)
        }
    """
    import numpy as np
    
    # Calculate actual movement
    prev_pos = np.array(previous_pose[0])
    curr_pos = np.array(current_pose[0])
    movement_vector = curr_pos - prev_pos
    movement_magnitude = np.linalg.norm(movement_vector)
    
    if movement_magnitude < 1e-6:  # Essentially no movement
        return {
            'alignment': 1.0,      # Perfect alignment when not moving
            'magnitude': 0.0,
            'efficiency': 0.0      # No efficiency when not moving
        }
    
    # Normalize movement vector
    movement_direction = movement_vector / movement_magnitude
    
    # Calculate alignment with intended direction
    alignment = np.dot(movement_direction, intended_direction_vector)
    alignment = np.clip(alignment, -1.0, 1.0)  # Ensure valid range
    alignment = (alignment + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
    
    # Efficiency combines alignment with movement magnitude
    efficiency = alignment * min(movement_magnitude / 0.1, 1.0)  # Cap efficiency at reasonable movement speed
    
    return {
        'alignment': float(alignment),
        'magnitude': float(movement_magnitude), 
        'efficiency': float(efficiency)
    }


def get_robot_orientation_metrics():
    """
    Get comprehensive robot orientation metrics for reward calculation.
    Returns:
        dict: Complete robot state information for RL training
    """
    import time
    
    pose = get_robot_pose()
    axes = get_robot_local_axes()
    fallen = is_robot_fallen()
    
    return {
        'pose': pose,
        'local_axes': axes,
        'is_fallen': fallen,
        'timestamp': time.time()
    }


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
