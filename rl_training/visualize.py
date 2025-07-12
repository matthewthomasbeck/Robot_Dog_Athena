import pybullet as p
import pybullet_data
import time
import os

# Connect to PyBullet
p.connect(p.GUI)

# (Optional) Set PyBullet's default search path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 1. Load a plane
plane_id = p.loadURDF("plane.urdf", [0, 0, 0])

# 2. Gravity
p.setGravity(0, 0, 0)

# Absolute path to URDF
urdf_path = "/Users/matthewthomasbeck/Library/Mobile Documents/com~apple~CloudDocs/Projects/Robot_Dog/Training/urdf/robot_dog.urdf"
if not os.path.isfile(urdf_path):
    print(f"URDF file not found at: {urdf_path}")
    exit()

# 3. Load the robot
robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 1],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
)
print(f"Robot ID: {robot_id}")

# Identify all continuous joints
num_joints = p.getNumJoints(robot_id)
continuous_joint_indices = []

for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]
    # Check if it's a continuous joint
    # (PyBullet uses 0 for REVOLUTE, 1 for PRISMATIC, 2 for SPHERICAL,
    #  3 for PLANAR, 4 for FIXED; 'continuous' also is 0 in some versions
    #  but we'll check the 'jointUpperLimit < jointLowerLimit' if needed.)
    if joint_type == p.JOINT_REVOLUTE and info[8] < info[9]:
        # This indicates it might be a limited revolute joint
        pass
    elif joint_type == p.JOINT_REVOLUTE and info[8] > info[9]:
        # This condition is sometimes used to indicate a continuous joint
        continuous_joint_indices.append(i)
    print(f"Joint index {i} | name: {joint_name} | type: {joint_type}")

# If we didn't find any continuous joints, let's just warn and exit
if not continuous_joint_indices:
    print("No continuous joints found in the robot!")
    exit()

# --- CREATE SLIDERS ---
# Slider to toggle continuous joint rotation on/off
toggle_id = p.addUserDebugParameter("Enable Continuous Joints", 0, 1, 0)

# Three sliders for camera settings
yaw_id = p.addUserDebugParameter("Camera Yaw", -180, 180, 90.947)
pitch_id = p.addUserDebugParameter("Camera Pitch", -90, 90, 10.421)
zoom_id = p.addUserDebugParameter("Camera Zoom", 0, 1, 0.221)

# Main simulation loop
try:
    while True:
        # 1) Check the continuous joints toggle
        toggle_val = p.readUserDebugParameter(toggle_id)

        # If ON => spin at 1 rad/s, else set velocity = 0
        target_vel = 1.0 if toggle_val > 0.5 else 0.0

        # Apply control to all continuous joints
        for joint_index in continuous_joint_indices:
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_vel,
                force=4.41
            )

        # 2) Read camera sliders and update the camera
        yaw_val = p.readUserDebugParameter(yaw_id)
        pitch_val = p.readUserDebugParameter(pitch_id)
        zoom_val = p.readUserDebugParameter(zoom_id)

        p.resetDebugVisualizerCamera(
            cameraDistance=zoom_val,
            cameraYaw=yaw_val,
            cameraPitch=pitch_val,
            cameraTargetPosition=[0, 0, 1]
        )

        p.stepSimulation()
        time.sleep(1.0 / 240.0)

except KeyboardInterrupt:
    p.disconnect()
