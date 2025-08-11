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





##################################################
############### TRAINING FUNCTIONS ###############
##################################################


########## FALLEN ROBOT ##########

def is_robot_fallen():  # TODO this function does a good job at telling when the robot has fallen, DO NOT TOUCH
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
        logging.error(f"(training.py): Failed to check if robot fallen: {e}\n")
        return False  # Assume upright if error


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


########## STANDARD RL AGENT INTERFACE ##########

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


########## BLIND RL AGENT INTERFACE ##########

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
                logging.info("(training.py): SB3 PPO model initialized for RL training.\n")

            except ImportError as e:
                import logging
                logging.warning(f"(training.py): SB3 not available, using random actions: {e}\n")
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
        logging.error(f"(training.py): Error in get_rl_action_blind: {e}")

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