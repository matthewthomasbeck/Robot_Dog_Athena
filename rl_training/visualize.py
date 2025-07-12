import pybullet as p
import pybullet_data
import time
import os
import math

# --- CONFIG ---
ANGLE_STEP = 0.05  # radians per button press

# --- SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", [0, 0, 0])
p.setGravity(0, 0, 0)

urdf_path = "/Users/matthewthomasbeck/Library/Mobile Documents/com~apple~CloudDocs/Projects/Robot_Dog/Training/urdf/robot_dog.urdf"
if not os.path.isfile(urdf_path):
    print(f"URDF file not found at: {urdf_path}")
    exit()

robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 1],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
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
leg_dropdown = p.addUserDebugParameter("Leg", 0, len(leg_names)-1, 0)
joint_dropdown = p.addUserDebugParameter("Joint", 0, len(joint_types)-1, 0)
plus_button = p.addUserDebugParameter("+ (Increase Angle)", 0, 1, 0)
minus_button = p.addUserDebugParameter("- (Decrease Angle)", 0, 1, 0)

# Track state
current_angle = 0.0
last_leg_idx = -1
last_joint_idx = -1
last_plus = 0
last_minus = 0

while True:
    leg_idx = int(p.readUserDebugParameter(leg_dropdown))
    joint_idx = int(p.readUserDebugParameter(joint_dropdown))
    leg = leg_names[leg_idx]
    joint_type = joint_types[joint_idx]
    joint_key = (leg, joint_type)
    joint_index = joint_map.get(joint_key, None)
    if joint_index is None:
        p.addUserDebugText(f"No joint: {leg}_{joint_type}", [0,0,1.5], textColorRGB=[1,0,0], replaceItemUniqueId=0)
        time.sleep(0.05)
        continue

    # If leg/joint changed, reset angle to current joint position
    if leg_idx != last_leg_idx or joint_idx != last_joint_idx:
        current_angle = p.getJointState(robot_id, joint_index)[0]
        last_leg_idx = leg_idx
        last_joint_idx = joint_idx
        print(f"Selected joint: {leg}_{joint_type} (index {joint_index})")
        print(f"Current angle: {current_angle:.3f} rad ({math.degrees(current_angle):.2f} deg)")

    # Read button states
    plus_val = p.readUserDebugParameter(plus_button)
    minus_val = p.readUserDebugParameter(minus_button)
    # Button logic: only increment when pressed (value goes from 0 to 1)
    if plus_val > 0.5 and last_plus <= 0.5:
        current_angle += ANGLE_STEP
        print(f"[+] {leg}_{joint_type}: {current_angle:.3f} rad ({math.degrees(current_angle):.2f} deg)")
    if minus_val > 0.5 and last_minus <= 0.5:
        current_angle -= ANGLE_STEP
        print(f"[-] {leg}_{joint_type}: {current_angle:.3f} rad ({math.degrees(current_angle):.2f} deg)")
    last_plus = plus_val
    last_minus = minus_val

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

    # Display current angle
    p.addUserDebugText(f"{leg}_{joint_type}: {current_angle:.3f} rad ({math.degrees(current_angle):.2f} deg)", [0,0,1.5], textColorRGB=[0,0,1], replaceItemUniqueId=1)

    p.stepSimulation()
    time.sleep(1.0 / 240.0)
