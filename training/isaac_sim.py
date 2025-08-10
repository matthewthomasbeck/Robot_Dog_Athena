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


def get_rl_action_blind(current_angles, commands, intensity):
    """
    SB3 PPO RL agent's policy WITHOUT image input (imageless gait adjustment).
    Args:
        current_angles: Current joint angles in SERVO_CONFIG format
        commands: The movement commands (e.g., 'w', 'w+arrowleft', etc.)
        intensity: The movement intensity (1-10)
    Returns:
        target_angles: dict of target joint angles for each leg (similar to SERVO_CONFIG structure).
        mid_angles: dict of mid joint angles for each leg (similar to SERVO_CONFIG structure).
        movement_rates: dict of movement rate parameters for each leg.
    """
    import numpy as np
    import utilities.config as config
    
    try:
        ##### INITIALIZE SB3 PPO MODEL (ONCE) #####
        global PPO_MODEL, MODEL_INITIALIZED
        
        if 'PPO_MODEL' not in globals():
            try:
                # Import SB3
                from stable_baselines3 import PPO
                from stable_baselines3.common.vec_env import DummyVecEnv
                from stable_baselines3.common.env_util import make_vec_env
                import gymnasium as gym
                from gymnasium import spaces
                
                # Create simple gym environment
                class SimpleWalkingEnv:
                    def __init__(self):
                        # Observation space: 21D (12 joint angles + 8 commands + 1 intensity)
                        self.observation_space = spaces.Box(low=-10, high=10, shape=(21,), dtype=np.float32)
                        # Action space: 36D (12 target + 12 mid + 12 speeds)
                        self.action_space = spaces.Box(low=-1, high=1, shape=(36,), dtype=np.float32)
                
                # Create environment
                env = DummyVecEnv([lambda: SimpleWalkingEnv()])
                
                # Create PPO model
                PPO_MODEL = PPO("MlpPolicy", env, verbose=0, 
                               learning_rate=3e-4, n_steps=2048, batch_size=64)
                MODEL_INITIALIZED = True
                
                import logging
                logging.info("(isaac_sim.py): SB3 PPO model initialized for RL training.\n")
                
            except ImportError as e:
                import logging
                logging.warning(f"(isaac_sim.py): SB3 not available, using random actions: {e}\n")
                PPO_MODEL = None
                MODEL_INITIALIZED = False
        
        ##### BUILD OBSERVATION FOR SB3 #####
        
        # 1. Extract joint angles (12D)
        joint_angles = []
        for leg_id in ['FL', 'FR', 'BL', 'BR']:
            for joint_name in ['hip', 'upper', 'lower']:
                angle = current_angles[leg_id][joint_name]
                joint_angles.append(float(angle))
        
        # 2. Encode commands (8D one-hot)
        if isinstance(commands, list):
            command_list = commands  # Already a list
        elif isinstance(commands, str):
            command_list = commands.split('+') if commands else []
        else:
            command_list = []  # Handle None or other types
        command_encoding = [
            1.0 if 'w' in command_list else 0.0,
            1.0 if 's' in command_list else 0.0,
            1.0 if 'a' in command_list else 0.0,
            1.0 if 'd' in command_list else 0.0,
            1.0 if 'arrowleft' in command_list else 0.0,
            1.0 if 'arrowright' in command_list else 0.0,
            1.0 if 'arrowup' in command_list else 0.0,
            1.0 if 'arrowdown' in command_list else 0.0
        ]
        
        # 3. Normalize intensity (1D)
        intensity_normalized = float(intensity) / 10.0
        
        # 4. Combine into observation (21D total)
        observation = np.array(joint_angles + command_encoding + [intensity_normalized], dtype=np.float32)
        
        ##### SB3 PPO INFERENCE OR RANDOM FALLBACK #####
        
        if PPO_MODEL is not None and MODEL_INITIALIZED:
            # Use SB3 PPO model for action prediction
            action, _states = PPO_MODEL.predict(observation, deterministic=False)
        else:
            # Fallback to random actions if SB3 not available
            action = np.random.uniform(-1.0, 1.0, 36)  # 36D action space
        
        # Store observation and action for learning (global access for perception loop)
        global CURRENT_OBSERVATION, CURRENT_ACTION
        CURRENT_OBSERVATION = observation.copy()
        CURRENT_ACTION = action.copy()
        
        ##### CONVERT ACTION TO SERVO_CONFIG FORMAT #####
        
        target_angles = {}
        mid_angles = {}
        movement_rates = {}
        
        action_idx = 0
        
        for leg_id in ['FL', 'FR', 'BL', 'BR']:
            target_angles[leg_id] = {}
            mid_angles[leg_id] = {}
            movement_rates[leg_id] = {'speed': 1.0, 'acceleration': 0.5}
            
            for joint_name in ['hip', 'upper', 'lower']:
                # Get joint limits from config
                servo_data = config.SERVO_CONFIG[leg_id][joint_name]
                min_angle = servo_data['FULL_BACK_ANGLE']
                max_angle = servo_data['FULL_FRONT_ANGLE']
                
                # Ensure correct order (min < max)
                if min_angle > max_angle:
                    min_angle, max_angle = max_angle, min_angle
                
                # Convert target action (-1 to 1) to joint angle (min to max)
                target_action = action[action_idx]
                target_angle = min_angle + (target_action + 1.0) * 0.5 * (max_angle - min_angle)
                target_angle = np.clip(target_angle, min_angle, max_angle)
                target_angles[leg_id][joint_name] = float(target_angle)
                
                # Convert mid action (-1 to 1) to joint angle (min to max)
                mid_action = action[action_idx + 12]
                mid_angle = min_angle + (mid_action + 1.0) * 0.5 * (max_angle - min_angle)
                mid_angle = np.clip(mid_angle, min_angle, max_angle)
                mid_angles[leg_id][joint_name] = float(mid_angle)
                
                action_idx += 1
        
        # Convert movement rates (12D normalized 0-1 to actual speeds)
        # PPO now outputs direct rad/s values (no scaling needed)
        rate_idx = 0
        for leg_id in ['FL', 'FR', 'BL', 'BR']:
            # Speed action (-1 to 1) converted to direct rad/s values
            speed_action = action[24 + rate_idx]  # Last 12 actions are speeds
            # START WITH SLOW LEARNING SPEEDS: Map to 0.01-0.5 rad/s for initial stability
            # URDF limit is 9.52 rad/s, so we stay well below that during learning
            speed = 0.01 + (speed_action + 1.0) * 0.5 * (0.5 - 0.01)  # Map to 0.01-0.5 rad/s
            speed = np.clip(speed, 0.01, 0.5)  # Never exceed 0.5 rad/s during learning
            # Safety check: Never exceed URDF velocity limit of 9.52 rad/s
            speed = min(speed, 9.52)
            
            movement_rates[leg_id]['speed'] = float(speed)
            movement_rates[leg_id]['acceleration'] = 0.5  # Fixed for now
            rate_idx += 1
            
            # Note: We're using the same speed for all 3 joints per leg for simplicity
            # This could be expanded to 12 different speeds later
        
        return target_angles, mid_angles, movement_rates
        
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Error in get_rl_action_blind: {e}")
        
        # Fallback: return current angles as both mid and target (no movement)
    target_angles = {}
    mid_angles = {}
    movement_rates = {}
    
    for leg_id in ['FL', 'FR', 'BL', 'BR']:
        target_angles[leg_id] = {}
        mid_angles[leg_id] = {}
        movement_rates[leg_id] = {'speed': 1.0, 'acceleration': 0.5}
        
        for joint_name in ['hip', 'upper', 'lower']:
            current_angle = current_angles[leg_id][joint_name]
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
            - episode_done: Whether episode should be reset
    """
    # Check if episode should be reset (robot has fallen)
    episode_done = is_robot_fallen()
    
    if episode_done:
        # Reset episode - put robot back to neutral position
        reset_episode()
        # Return neutral command to give robot time to stabilize
        return {
            'command': None,  # Neutral command 
            'intensity': 10,
            'episode_done': True
        }
    
    # Generate RL command and intensity
    command, intensity = get_rl_command_with_intensity()
    
    # Put command into queue (same as web command flow)
    inject_rl_command_into_queue(rl_command_queue, command, intensity)
    
    # Return step information for RL training
    return {
        'command': command,
        'intensity': intensity,
        'episode_done': False
    }


def reset_episode():
    """
    Reset the RL episode by resetting Isaac Sim world and moving robot to neutral position.
    """
    global PREV_ROBOT_POSITION, REWARD_TRACKING_ENABLED, EPISODE_COUNTER
    
    import logging
    from movement.fundamental_movement import neutral_position
    
    try:
        logging.info(f"(isaac_sim.py): Episode {EPISODE_COUNTER + 1} starting - Robot fallen, resetting Isaac Sim world.\n")
        
        # Reset Isaac Sim world (position, velocity, physics state)
        import utilities.config as config
        if config.USE_SIMULATION and config.USE_ISAAC_SIM:
            # Reset the world - this resets robot position, velocities, and physics state
            config.ISAAC_WORLD.reset()
            
            # Give Isaac Sim a few steps to stabilize after world reset
            for _ in range(5):
                config.ISAAC_WORLD.step(render=True)
        
        # Move robot to neutral position (joint angles)
        neutral_position(10)  # High intensity for quick reset
        
        # Give Isaac Sim more steps to stabilize after neutral position
        if config.USE_SIMULATION and config.USE_ISAAC_SIM:
            for _ in range(5):
                config.ISAAC_WORLD.step(render=True)
        
        # Reset reward tracking after world and position reset
        PREV_ROBOT_POSITION = get_robot_position()
        REWARD_TRACKING_ENABLED = True
        
        # Increment episode counter and reset step counter
        EPISODE_COUNTER += 1
        EPISODE_STEP_COUNT = 0
        
        # Log episode summary before resetting step count
        if EPISODE_STEP_COUNT > 0:
            logging.info(f"(isaac_sim.py): Episode {EPISODE_COUNTER} completed with {EPISODE_STEP_COUNT} forward steps. Total forward progress: {TOTAL_FORWARD_PROGRESS:.4f}")
        else:
            logging.info(f"(isaac_sim.py): Episode {EPISODE_COUNTER} failed - no forward movement. Total forward progress: {TOTAL_FORWARD_PROGRESS:.4f}")
        
        logging.info(f"(isaac_sim.py): Episode {EPISODE_COUNTER} reset complete - World and robot state reset.\n")
        
        # Log learning progress every 10 episodes
        if EPISODE_COUNTER % 10 == 0:
            logging.info(f"(isaac_sim.py): LEARNING PROGRESS - Episodes: {EPISODE_COUNTER}, Total Forward Progress: {TOTAL_FORWARD_PROGRESS:.4f}")
        
    except Exception as e:
        logging.error(f"(isaac_sim.py): Failed to reset episode: {e}\n")


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


########## SIMPLE FORWARD WALKING FUNCTIONS ##########

def get_robot_position():
    """
    Get robot's current position (x, y, z).
    Returns:
        tuple: (x, y, z) position
    """
    import utilities.config as config
    
    try:
        positions, rotations = config.ISAAC_ROBOT.get_world_poses()
        # get_world_poses() returns arrays - get first element (index 0)
        position = positions[0]  # First robot position
        return (float(position[0]), float(position[1]), float(position[2]))
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Failed to get robot position: {e}\n")
        return (0.0, 0.0, 0.0)


def get_robot_forward_direction():
    """
    Get robot's current forward direction (face points West initially, toward Yellow).
    Robot's forward = direction the face is pointing = local +Y direction.
    Returns:
        numpy.array: [x, y, z] unit vector showing forward direction
    """
    import numpy as np
    import utilities.config as config
    from scipy.spatial.transform import Rotation
    
    try:
        positions, rotations = config.ISAAC_ROBOT.get_world_poses()
        # get_world_poses() returns arrays - get first element (index 0)
        rotation = rotations[0]  # First robot rotation
        
        # Convert quaternion to rotation object
        quat_wxyz = [rotation[0], rotation[1], rotation[2], rotation[3]]  # (w, x, y, z)
        r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # scipy wants (x, y, z, w)
        
        # Robot's forward direction is local +Y (initially points toward Yellow/West)
        local_forward = np.array([0.0, 1.0, 0.0])  # Local +Y
        world_forward = r.apply(local_forward)
        
        return world_forward
        
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Failed to get robot forward direction: {e}")
        return np.array([0.0, 1.0, 0.0])  # Default: pointing West


def is_robot_fallen():
    """
    Check if robot has tilted more than 45 degrees from upright.
    Returns:
        bool: True if robot has fallen over
    """
    import numpy as np
    import utilities.config as config
    from scipy.spatial.transform import Rotation
    
    try:
        positions, rotations = config.ISAAC_ROBOT.get_world_poses()
        # get_world_poses() returns arrays - get first element (index 0)
        rotation = rotations[0]  # First robot rotation
        
        # Convert quaternion to rotation object
        quat_wxyz = [rotation[0], rotation[1], rotation[2], rotation[3]]  # (w, x, y, z)
        r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # scipy wants (x, y, z, w)
        
        # Robot's up direction is local +Z
        local_up = np.array([0.0, 0.0, 1.0])  # Local +Z
        robot_up = r.apply(local_up)
        world_up = np.array([0.0, 0.0, 1.0])  # World +Z
        
        # Calculate angle between robot up and world up
        dot_product = np.dot(robot_up, world_up)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure valid range
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg > 45.0  # Fallen if tilted more than 45 degrees
        
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Failed to check if robot fallen: {e}\n")
        return False  # Assume upright if error


def calculate_forward_movement(prev_position, curr_position, forward_direction):
    """
    Calculate how much robot moved in its forward direction.
    Args:
        prev_position: (x, y, z) from previous step
        curr_position: (x, y, z) from current step  
        forward_direction: numpy.array([x, y, z]) robot's forward vector
    Returns:
        float: Distance moved forward (positive = good, negative = backward)
    """
    import numpy as np
    
    # Calculate movement vector
    prev_pos = np.array(prev_position)
    curr_pos = np.array(curr_position)
    movement = curr_pos - prev_pos
    
    # Project movement onto forward direction
    forward_distance = np.dot(movement, forward_direction)
    
    return float(forward_distance)


def compute_simple_walking_reward(prev_position, curr_position, command, intensity):
    """
    Simple reward function for training robot to walk forward.
    Args:
        prev_position: (x, y, z) from previous step
        curr_position: (x, y, z) from current step
        command: str - command executed ('w', 'n', etc.)
        intensity: int - movement intensity (1-10)
    Returns:
        float: Reward value (positive = good, negative = bad)
    """
    import logging
    
    reward = 0.0
    
    # Check if robot has fallen
    if is_robot_fallen():
        reward -= 100.0  # Large penalty for falling
        logging.debug(f"(isaac_sim.py): Robot fallen! Penalty: -100.0\n")
        return reward
    
    # Only reward forward movement for 'w' commands
    if command == 'w':
        # Get robot's current forward direction
        forward_dir = get_robot_forward_direction()
        
        # Calculate how much robot moved forward
        forward_progress = calculate_forward_movement(prev_position, curr_position, forward_dir)
        
                # Reward forward movement, penalize backward movement
        if forward_progress > 0:
            reward += forward_progress * intensity * 10  # Scale by intensity and amplify
            logging.debug(f"(isaac_sim.py): Forward progress: {forward_progress:.4f}, reward: +{reward:.2f}")
            
            # Track total forward progress across episodes
            global TOTAL_FORWARD_PROGRESS
            TOTAL_FORWARD_PROGRESS += forward_progress
            logging.debug(f"(isaac_sim.py): Total forward progress: {TOTAL_FORWARD_PROGRESS:.4f}")
        else:
            reward += forward_progress * intensity * 5  # Smaller penalty for backward movement
            logging.debug(f"(isaac_sim.py): Backward movement: {forward_progress:.4f}, reward: {reward:.2f}")
        
        # Penalty for not moving forward when commanded
        if abs(forward_progress) < 0.001:  # Very small movement threshold
            penalty = -5.0  # Stronger penalty for standing still when should move forward
            reward += penalty
            logging.debug(f"(isaac_sim.py): No forward movement on 'w' command, penalty: {penalty}")
            
            # Additional penalty that increases over time if robot keeps standing still
            if command == 'w':
                standing_still_penalty = -1.0  # Small penalty for each step of standing still
                reward += standing_still_penalty
                logging.debug(f"(isaac_sim.py): Standing still penalty: {standing_still_penalty}")
    
 
    
    # Neutral commands get small positive reward for stability
    elif command == 'n' or command is None:
        if not is_robot_fallen():
            reward += 0.1  # Small reward for staying upright
    
    # Large reward for staying upright and moving forward successfully
    if command == 'w' and not is_robot_fallen():
        # Bonus for staying upright while executing forward command
        reward += 5.0
        logging.debug(f"(isaac_sim.py): Staying upright during forward movement! Bonus: +5.0")
    
    # Reward for staying upright longer (encourages longer episodes)
    if not is_robot_fallen():
        # Small reward that accumulates over time
        reward += 0.05
        logging.debug(f"(isaac_sim.py): Staying upright! Episode length reward: +0.05")
        
        # Only count steps when robot is actually moving (not just standing still)
        global EPISODE_STEP_COUNT
        if command == 'w' and abs(forward_progress) > 0.001:  # Only count steps when moving forward
            EPISODE_STEP_COUNT += 1
            logging.debug(f"(isaac_sim.py): Forward movement step! Step count: {EPISODE_STEP_COUNT}")
            
            if EPISODE_STEP_COUNT % 10 == 0:  # Every 10 forward steps
                milestone_bonus = min(EPISODE_STEP_COUNT * 0.1, 5.0)  # Cap at 5.0
                reward += milestone_bonus
                logging.debug(f"(isaac_sim.py): Forward movement milestone! Step {EPISODE_STEP_COUNT}, bonus: +{milestone_bonus:.2f}")
            
            # Episode success reward (robot successfully moves forward for extended period)
            if EPISODE_STEP_COUNT >= 20:  # 20 forward steps = successful episode
                success_bonus = 100.0  # Large bonus for completing forward movement episode
                reward += success_bonus
                logging.info(f"(isaac_sim.py): FORWARD MOVEMENT SUCCESS! Robot moved forward for {EPISODE_STEP_COUNT} steps! Bonus: +{success_bonus}")
    
    # Add velocity penalty to encourage slower, more controlled movements
    try:
        # Get the current action that was applied (includes velocities)
        if 'CURRENT_ACTION' in globals() and CURRENT_ACTION is not None:
            # The last 12 values in the action are movement rates (velocities)
            velocities = CURRENT_ACTION[24:36]  # Last 12 actions are speeds
            
            # Calculate average velocity and penalize high speeds
            avg_velocity = sum(abs(v) for v in velocities) / len(velocities)
            
            # Stronger penalty for high velocities (encourages much slower, more controlled movement)
            # Scale penalty to be more impactful - 25% penalty on average velocity
            velocity_penalty = avg_velocity * 0.25  # 25% penalty on average velocity
            reward -= velocity_penalty
            
            logging.debug(f"(isaac_sim.py): Velocity penalty: -{velocity_penalty:.4f} (avg vel: {avg_velocity:.4f})\n")
    except Exception as e:
        # If we can't get velocity info, just continue without penalty
        pass
        
    return reward



def start_reward_tracking():
    """Initialize reward tracking system."""
    global PREV_ROBOT_POSITION, REWARD_TRACKING_ENABLED
    PREV_ROBOT_POSITION = get_robot_position()
    REWARD_TRACKING_ENABLED = True
    import logging
    # logging.info("(isaac_sim.py): Reward tracking started.\n")

def get_step_reward(command, intensity):
    """
    Get reward for current step and update tracking.
    Args:
        command: str - command executed
        intensity: int - movement intensity
    Returns:
        float: Reward for this step
    """
    global PREV_ROBOT_POSITION, REWARD_TRACKING_ENABLED
    
    if not REWARD_TRACKING_ENABLED:
        start_reward_tracking()
        return 0.0  # No reward for first step
    
    # Get current position
    curr_position = get_robot_position()
    
    # Calculate reward
    reward = compute_simple_walking_reward(PREV_ROBOT_POSITION, curr_position, command, intensity)
    
    # Update previous position for next step
    PREV_ROBOT_POSITION = curr_position
    
    return reward


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

def collect_experience_and_train(observation, action, reward, next_observation, done):
    """
    Collect experience and train PPO model periodically.
    Called from the perception loop after each step.
    """
    global PPO_MODEL, EXPERIENCE_BUFFER, TRAINING_STEP_COUNT
    
    if PPO_MODEL is None or not MODEL_INITIALIZED:
        return  # Skip if model not ready
    
    try:
        import numpy as np
        
        # Store experience
        experience = {
            'observation': observation,
            'action': action, 
            'reward': reward,
            'next_observation': next_observation,
            'done': done
        }
        EXPERIENCE_BUFFER.append(experience)
        TRAINING_STEP_COUNT += 1
        
        # Train every 64 steps (batch_size from PPO config)
        if TRAINING_STEP_COUNT >= 64:
            
            # Create simple environment for training
            class TrainingEnv:
                def __init__(self, experiences):
                    self.experiences = experiences
                    self.current_idx = 0
                    from gymnasium import spaces
                    self.observation_space = spaces.Box(low=-10, high=10, shape=(21,), dtype=np.float32)
                    self.action_space = spaces.Box(low=-1, high=1, shape=(36,), dtype=np.float32)
                
                def reset(self):
                    if self.current_idx < len(self.experiences):
                        obs = self.experiences[self.current_idx]['observation']
                        return obs, {}
                    return np.zeros(21, dtype=np.float32), {}
                
                def step(self, action):
                    if self.current_idx < len(self.experiences):
                        exp = self.experiences[self.current_idx]
                        self.current_idx += 1
                        return exp['next_observation'], exp['reward'], exp['done'], False, {}
                    return np.zeros(21, dtype=np.float32), 0.0, True, False, {}
            
            # Train the model with collected experiences
            try:
                # Use PPO's built-in learning with collected experiences
                # PPO will use its own environment, we just need to trigger learning
                PPO_MODEL.learn(total_timesteps=len(EXPERIENCE_BUFFER), reset_num_timesteps=False)
                
                import logging
                logging.info(f"(isaac_sim.py): PPO trained on {len(EXPERIENCE_BUFFER)} experiences. Total steps: {TRAINING_STEP_COUNT}\n")
                
                # Clear buffer and reset counter
                EXPERIENCE_BUFFER = []
                TRAINING_STEP_COUNT = 0
                
            except Exception as e:
                import logging
                logging.error(f"(isaac_sim.py): Failed to train PPO model: {e}\n")
                
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Error collecting experience: {e}\n")


def store_step_for_learning(observation, action, reward):
    """
    Store the current step data for learning. Called from perception loop.
    """
    global LAST_OBSERVATION, LAST_ACTION
    
    # Store for next step
    LAST_OBSERVATION = observation.copy() if observation is not None else None
    LAST_ACTION = action.copy() if action is not None else None


def get_last_step_data():
    """
    Get the stored step data for learning.
    """
    global LAST_OBSERVATION, LAST_ACTION
    return LAST_OBSERVATION, LAST_ACTION


def update_learning_from_perception_loop(reward, episode_done):
    """
    Called from _perception_loop to update learning with current step data.
    Integrates with existing reward calculation and episode management.
    """
    global CURRENT_OBSERVATION, CURRENT_ACTION, LAST_OBSERVATION, LAST_ACTION
    
    try:
        # Check if we have current step data
        if CURRENT_OBSERVATION is None or CURRENT_ACTION is None:
            return  # No data to learn from
        
        # Check if we have previous step data for experience
        if LAST_OBSERVATION is not None and LAST_ACTION is not None:
            # We have a complete experience: (last_obs, last_action, reward, current_obs, done)
            collect_experience_and_train(
                observation=LAST_OBSERVATION,
                action=LAST_ACTION, 
                reward=reward,
                next_observation=CURRENT_OBSERVATION,
                done=episode_done
            )
        
        # Store current step as last step for next iteration
        LAST_OBSERVATION = CURRENT_OBSERVATION.copy()
        LAST_ACTION = CURRENT_ACTION.copy()
        
        # Clear current step
        CURRENT_OBSERVATION = None
        CURRENT_ACTION = None
        
    except Exception as e:
        import logging
        logging.error(f"(isaac_sim.py): Error in update_learning_from_perception_loop: {e}\n")
