"""
Calibration module for Robot Dog RL Training
Provides conversion functions between PyBullet angles and real servo values
"""

import math
from utilities.config import SERVO_CONFIG, LINK_CONFIG
from utilities.mathematics import Kinematics

# Initialize kinematics with the same link config as real robot
k = Kinematics(LINK_CONFIG)

def angle_to_servo_value(angle_rad, leg_id, joint_type):
    """
    Convert PyBullet angle (radians) to servo value for real robot
    Args:
        angle_rad: Joint angle in radians
        leg_id: 'FL', 'FR', 'BL', 'BR'
        joint_type: 'hip', 'upper', 'lower'
    Returns:
        int: Servo value for set_target function
    """
    servo_data = SERVO_CONFIG[leg_id][joint_type]
    
    # Get the angle range for this joint
    full_front_angle = servo_data['FULL_FRONT_ANGLE']
    full_back_angle = servo_data['FULL_BACK_ANGLE']
    
    # Get the servo value range
    full_front_servo = servo_data['FULL_FRONT']
    full_back_servo = servo_data['FULL_BACK']
    
    # Linear interpolation from angle to servo value
    if full_front_angle != full_back_angle:
        ratio = (angle_rad - full_front_angle) / (full_back_angle - full_front_angle)
        servo_value = full_front_servo + ratio * (full_back_servo - full_front_servo)
    else:
        servo_value = servo_data['NEUTRAL']
    
    return int(servo_value)

def servo_value_to_angle(servo_value, leg_id, joint_type):
    """
    Convert servo value to PyBullet angle (radians)
    Args:
        servo_value: Servo value from real robot
        leg_id: 'FL', 'FR', 'BL', 'BR'
        joint_type: 'hip', 'upper', 'lower'
    Returns:
        float: Joint angle in radians
    """
    servo_data = SERVO_CONFIG[leg_id][joint_type]
    
    # Get the angle range for this joint
    full_front_angle = servo_data['FULL_FRONT_ANGLE']
    full_back_angle = servo_data['FULL_BACK_ANGLE']
    
    # Get the servo value range
    full_front_servo = servo_data['FULL_FRONT']
    full_back_servo = servo_data['FULL_BACK']
    
    # Linear interpolation from servo value to angle
    if full_front_servo != full_back_servo:
        ratio = (servo_value - full_front_servo) / (full_back_servo - full_front_servo)
        angle_rad = full_front_angle + ratio * (full_back_angle - full_front_angle)
    else:
        angle_rad = 0.0
    
    return angle_rad

def foot_position_to_joint_angles(x, y, z, leg_id):
    """
    Convert foot position to joint angles using the same IK as real robot
    Args:
        x, y, z: Foot position coordinates
        leg_id: 'FL', 'FR', 'BL', 'BR'
    Returns:
        tuple: (hip_angle, upper_angle, lower_angle) in radians
    """
    # Use the same IK as your real robot
    hip_angle, upper_angle, lower_angle = k.inverse_kinematics(x, y, z)
    
    # Convert from degrees to radians (assuming your IK returns degrees)
    hip_rad = math.radians(hip_angle)
    upper_rad = math.radians(upper_angle)
    lower_rad = math.radians(lower_angle)
    
    return hip_rad, upper_rad, lower_rad

def joint_angles_to_foot_position(hip_angle, upper_angle, lower_angle, leg_id):
    """
    Convert joint angles to foot position using forward kinematics
    Args:
        hip_angle, upper_angle, lower_angle: Joint angles in radians
        leg_id: 'FL', 'FR', 'BL', 'BR'
    Returns:
        dict: {'x': x, 'y': y, 'z': z} foot position
    """
    # Convert from radians to degrees (assuming your FK expects degrees)
    hip_deg = math.degrees(hip_angle)
    upper_deg = math.degrees(upper_angle)
    lower_deg = math.degrees(lower_angle)
    
    # Use forward kinematics to get foot position
    # Note: You'll need to implement this in your mathematics module
    # For now, this is a placeholder
    x, y, z = k.forward_kinematics(hip_deg, upper_deg, lower_deg)
    
    return {'x': x, 'y': y, 'z': z}

def get_joint_limits(leg_id, joint_type):
    """
    Get joint limits for PyBullet
    Args:
        leg_id: 'FL', 'FR', 'BL', 'BR'
        joint_type: 'hip', 'upper', 'lower'
    Returns:
        tuple: (lower_limit, upper_limit) in radians
    """
    servo_data = SERVO_CONFIG[leg_id][joint_type]
    return servo_data['FULL_FRONT_ANGLE'], servo_data['FULL_BACK_ANGLE']

def get_neutral_foot_position(leg_id):
    """
    Get neutral foot position for a leg
    Args:
        leg_id: 'FL', 'FR', 'BL', 'BR'
    Returns:
        dict: {'x': x, 'y': y, 'z': z} neutral position
    """
    from utilities.config import FL_NEUTRAL, FR_NEUTRAL, BL_NEUTRAL, BR_NEUTRAL
    
    neutral_positions = {
        'FL': FL_NEUTRAL,
        'FR': FR_NEUTRAL,
        'BL': BL_NEUTRAL,
        'BR': BR_NEUTRAL
    }
    
    return neutral_positions[leg_id]

def validate_foot_position(x, y, z, leg_id):
    """
    Check if a foot position is reachable by the leg
    Args:
        x, y, z: Foot position coordinates
        leg_id: 'FL', 'FR', 'BL', 'BR'
    Returns:
        bool: True if position is reachable
    """
    try:
        hip_angle, upper_angle, lower_angle = foot_position_to_joint_angles(x, y, z, leg_id)
        
        # Check if all joint angles are within limits
        hip_lower, hip_upper = get_joint_limits(leg_id, 'hip')
        upper_lower, upper_upper = get_joint_limits(leg_id, 'upper')
        lower_lower, lower_upper = get_joint_limits(leg_id, 'lower')
        
        hip_rad = math.radians(hip_angle)
        upper_rad = math.radians(upper_angle)
        lower_rad = math.radians(lower_angle)
        
        return (hip_lower <= hip_rad <= hip_upper and
                upper_lower <= upper_rad <= upper_upper and
                lower_lower <= lower_rad <= lower_upper)
    except:
        return False 