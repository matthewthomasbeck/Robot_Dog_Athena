import pybullet as p
import pybullet_data
import time
import os
import math
from training.calibration import (
    angle_to_servo_value, 
    foot_position_to_joint_angles,
    get_joint_limits,
    get_neutral_foot_position,
    validate_foot_position
)

# --- CONFIG ---
ANGLE_STEP = 0.01  # radians per button press

# --- SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", [0, 0, 0])
p.setGravity(0, 0, 0)

# Camera sliders
cam_yaw_slider = p.addUserDebugParameter("Camera Yaw", -180, 180, 45)
cam_pitch_slider = p.addUserDebugParameter("Camera Pitch", -89, 89, -30)
cam_dist_slider = p.addUserDebugParameter("Camera Distance", 0, 5, 2)
cam_target_x_slider = p.addUserDebugParameter("Camera Target X", -1, 1, 0)
cam_target_y_slider = p.addUserDebugParameter("Camera Target Y", -1, 1, 0)
cam_target_z_slider = p.addUserDebugParameter("Camera Target Z", 0, 2, 1)

urdf_path = "/Users/matthewthomasbeck/Library/Mobile Documents/com~apple~CloudDocs/Projects/Robot_Dog/Robot_Dog/training/urdf/robot_dog.urdf"
if not os.path.isfile(urdf_path):
    print(f"URDF file not found at: {urdf_path}")
    exit()

robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 1],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True  # <--- This keeps the base absolutely fixed
)
print(f"Robot ID: {robot_id}")

num_joints = p.getNumJoints(robot_id)
joint_info = []
leg_names = set()
joint_types = set()

# Parse joint names and build mapping
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]
    joint_info.append({
        'index': i,
        'name': joint_name,
        'type': joint_type,
        'lower': info[8],
        'upper': info[9],
    })
    # Expecting names like FL_hip, FR_upper, etc.
    parts = joint_name.split('_')
    if len(parts) >= 2:
        leg_names.add(parts[0])
        joint_types.add(parts[1])

leg_names = sorted(list(leg_names))
joint_types = sorted(list(joint_types))

# Build a mapping: (leg, joint_type) -> joint_index
joint_map = {}
for j in joint_info:
    parts = j['name'].split('_')
    if len(parts) >= 2:
        joint_map[(parts[0], parts[1])] = j['index']

# --- UI ELEMENTS ---
LEG_LABELS = ['FL', 'FR', 'BL', 'BR']
JOINT_LABELS = ['hip', 'upper', 'lower']
leg_slider = p.addUserDebugParameter("LEG (1=FL, 2=FR, 3=BL, 4=BR)", 1, 4, 1)
joint_slider = p.addUserDebugParameter("JOINT (1=hip, 2=upper, 3=lower)", 1, 3, 1)

# Track state
current_angle = 0.0
last_leg_idx = -1
last_joint_idx = -1

print("\nAvailable leg/type joints:")
for (leg, jt), idx in joint_map.items():
    print(f"  {leg}_{jt}: index {idx}")

print("\nKeyboard Controls:")
print("  Legs: 1=FL, 2=FR, 3=BL, 4=BR")
print("  Joints: h=hip, u=upper, l=lower")
print("  Angle: e=+, d=-, space=zero, f/b: FULL_FRONT/BACK")
print("  Test: t=test foot position to joint angles")

while True:
    # --- Camera controls ---
    cam_yaw = p.readUserDebugParameter(cam_yaw_slider)
    cam_pitch = p.readUserDebugParameter(cam_pitch_slider)
    cam_dist = p.readUserDebugParameter(cam_dist_slider)
    cam_target_x = p.readUserDebugParameter(cam_target_x_slider)
    cam_target_y = p.readUserDebugParameter(cam_target_y_slider)
    cam_target_z = p.readUserDebugParameter(cam_target_z_slider)
    p.resetDebugVisualizerCamera(
        cameraDistance=cam_dist,
        cameraYaw=cam_yaw,
        cameraPitch=cam_pitch,
        cameraTargetPosition=[cam_target_x, cam_target_y, cam_target_z]
    )

    # --- Leg/joint selection and control ---
    leg_idx = int(p.readUserDebugParameter(leg_slider)) - 1  # 0-based
    joint_idx = int(p.readUserDebugParameter(joint_slider)) - 1  # 0-based
    if 0 <= leg_idx < 4 and 0 <= joint_idx < 3:
        leg = LEG_LABELS[leg_idx]
        joint_type = JOINT_LABELS[joint_idx]
        joint_key = (leg, joint_type)
        joint_index = joint_map.get(joint_key, None)
    else:
        joint_index = None

    if joint_index is None:
        p.addUserDebugText(f"No joint: {leg}_{joint_type}", [0, 0, 1.5], textColorRGB=[1, 0, 0], replaceItemUniqueId=0)
        time.sleep(0.05)
        continue

    # If leg/joint changed, reset angle to current joint position
    if leg_idx != last_leg_idx or joint_idx != last_joint_idx:
        current_angle = p.getJointState(robot_id, joint_index)[0]
        last_leg_idx = leg_idx
        last_joint_idx = joint_idx
        print(f"Selected joint: {leg}_{joint_type} (index {joint_index})")
        print(f"Current angle: {current_angle:.3f} rad ({math.degrees(current_angle):.2f} deg)")
        
        # Get joint limits from calibration
        lower_limit, upper_limit = get_joint_limits(leg, joint_type)
        print(f"Calibrated limits: {lower_limit:.3f} to {upper_limit:.3f} rad")

    # --- Keyboard control ---
    keys = p.getKeyboardEvents()

    # Leg selection: 1, 2, 3, 4
    if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
        p.removeUserDebugItem(leg_slider)
        leg_slider = p.addUserDebugParameter("LEG (1=FL, 2=FR, 3=BL, 4=BR)", 1, 4, 1)
    if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
        p.removeUserDebugItem(leg_slider)
        leg_slider = p.addUserDebugParameter("LEG (1=FL, 2=FR, 3=BL, 4=BR)", 1, 4, 2)
    if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
        p.removeUserDebugItem(leg_slider)
        leg_slider = p.addUserDebugParameter("LEG (1=FL, 2=FR, 3=BL, 4=BR)", 1, 4, 3)
    if ord('4') in keys and keys[ord('4')] & p.KEY_WAS_TRIGGERED:
        p.removeUserDebugItem(leg_slider)
        leg_slider = p.addUserDebugParameter("LEG (1=FL, 2=FR, 3=BL, 4=BR)", 1, 4, 4)

    # Joint selection: h, u, l
    if ord('h') in keys and keys[ord('h')] & p.KEY_WAS_TRIGGERED:
        p.removeUserDebugItem(joint_slider)
        joint_slider = p.addUserDebugParameter("JOINT (1=hip, 2=upper, 3=lower)", 1, 3, 1)
    if ord('u') in keys and keys[ord('u')] & p.KEY_WAS_TRIGGERED:
        p.removeUserDebugItem(joint_slider)
        joint_slider = p.addUserDebugParameter("JOINT (1=hip, 2=upper, 3=lower)", 1, 3, 2)
    if ord('l') in keys and keys[ord('l')] & p.KEY_WAS_TRIGGERED:
        p.removeUserDebugItem(joint_slider)
        joint_slider = p.addUserDebugParameter("JOINT (1=hip, 2=upper, 3=lower)", 1, 3, 3)

    # Angle control: e, d
    if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
        current_angle += ANGLE_STEP
    if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
        current_angle -= ANGLE_STEP

    # Spacebar to reset to zero
    if p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
        current_angle = 0.0
    
    # F/B to set FULL_FRONT/BACK using calibrated values
    if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
        lower_limit, upper_limit = get_joint_limits(leg, joint_type)
        current_angle = lower_limit
    if ord('b') in keys and keys[ord('b')] & p.KEY_WAS_TRIGGERED:
        lower_limit, upper_limit = get_joint_limits(leg, joint_type)
        current_angle = upper_limit
    
    # T to test foot position to joint angles
    if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
        # Test with neutral foot position
        neutral_pos = get_neutral_foot_position(leg)
        print(f"\n=== Testing {leg} foot position to joint angles ===")
        print(f"Neutral foot position: {neutral_pos}")
        
        try:
            hip_rad, upper_rad, lower_rad = foot_position_to_joint_angles(
                neutral_pos['x'], neutral_pos['y'], neutral_pos['z'], leg
            )
            print(f"Calculated joint angles:")
            print(f"  Hip: {hip_rad:.3f} rad ({math.degrees(hip_rad):.1f}°)")
            print(f"  Upper: {upper_rad:.3f} rad ({math.degrees(upper_rad):.1f}°)")
            print(f"  Lower: {lower_rad:.3f} rad ({math.degrees(lower_rad):.1f}°)")
            
            # Convert to servo values
            hip_servo = angle_to_servo_value(hip_rad, leg, 'hip')
            upper_servo = angle_to_servo_value(upper_rad, leg, 'upper')
            lower_servo = angle_to_servo_value(lower_rad, leg, 'lower')
            print(f"Servo values: Hip={hip_servo}, Upper={upper_servo}, Lower={lower_servo}")
            
        except Exception as e:
            print(f"Error: {e}")

    # Clamp to joint limits
    lower = joint_info[joint_index]['lower']
    upper = joint_info[joint_index]['upper']
    if current_angle < lower:
        current_angle = lower
    if current_angle > upper:
        current_angle = upper

    # Set joint position
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=joint_index,
        controlMode=p.POSITION_CONTROL,
        targetPosition=current_angle,
        force=4.41
    )

    # Display current angle and servo value
    servo_value = angle_to_servo_value(current_angle, leg, joint_type)
    p.addUserDebugText(
        f"{leg}_{joint_type}: {current_angle:.3f} rad ({math.degrees(current_angle):.1f}°) | Servo: {servo_value}\n[1-4: legs | h/u/l: joints | e/d: +/- | space: zero | f/b: limits | t: test IK]",
        [0, 0, 1.5], textColorRGB=[0, 0, 1], replaceItemUniqueId=1)

    p.stepSimulation()
    time.sleep(1.0 / 240.0)
