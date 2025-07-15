#!/usr/bin/env python3
"""
Clean PyBullet simulation test
Tests the movement pipeline using trot_forward gait
"""

import time
import logging
import numpy as np
import pybullet as p
import pybullet_data
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utilities.config import USE_SIMULATION
from movement.fundamental_movement import *
from movement.walking.forward import *
from movement.standing.standing import *

def initialize_simulation():
    """Initialize PyBullet simulation"""
    print("Initializing PyBullet simulation...")
    
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Load robot URDF
    urdf_path = os.path.join("training", "urdf", "robot_dog.urdf")
    robot_id = p.loadURDF(urdf_path, [0, 0, 0.1], useFixedBase=False)  # Changed to False to allow body movement
    
    # Create joint mapping
    joint_map = {}
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')
        if '_' in joint_name and joint_name.split('_')[0] in ['FL', 'FR', 'BL', 'BR']:
            parts = joint_name.split('_')
            if len(parts) == 2 and parts[1] in ['hip', 'upper', 'lower']:
                joint_map[(parts[0], parts[1])] = i
    
    print(f"Robot ID: {robot_id}")
    print(f"Joint map: {joint_map}")
    
    # Set simulation variables in movement module
    set_simulation_variables(robot_id, joint_map)
    
    return robot_id, joint_map

def setup_camera_controls():
    """Setup camera control sliders"""
    cam_yaw_slider = p.addUserDebugParameter("Camera Yaw", -180, 180, 0)
    cam_pitch_slider = p.addUserDebugParameter("Camera Pitch", -89, 89, -30)
    cam_dist_slider = p.addUserDebugParameter("Camera Distance", 0.1, 10, 2)
    cam_target_x_slider = p.addUserDebugParameter("Camera Target X", -2, 2, 0)
    cam_target_y_slider = p.addUserDebugParameter("Camera Target Y", -2, 2, 0)
    cam_target_z_slider = p.addUserDebugParameter("Camera Target Z", -2, 2, 0)
    
    return (cam_yaw_slider, cam_pitch_slider, cam_dist_slider, 
            cam_target_x_slider, cam_target_y_slider, cam_target_z_slider)

def update_camera(cam_controls):
    """Update camera based on slider values"""
    try:
        cam_yaw, cam_pitch, cam_dist, cam_target_x, cam_target_y, cam_target_z = cam_controls
        cam_yaw_val = p.readUserDebugParameter(cam_yaw)
        cam_pitch_val = p.readUserDebugParameter(cam_pitch)
        cam_dist_val = p.readUserDebugParameter(cam_dist)
        cam_target_x_val = p.readUserDebugParameter(cam_target_x)
        cam_target_y_val = p.readUserDebugParameter(cam_target_y)
        cam_target_z_val = p.readUserDebugParameter(cam_target_z)
        
        p.resetDebugVisualizerCamera(
            cameraDistance=cam_dist_val,
            cameraYaw=cam_yaw_val,
            cameraPitch=cam_pitch_val,
            cameraTargetPosition=[cam_target_x_val, cam_target_y_val, cam_target_z_val]
        )
    except p.error:
        pass

def test_standing():
    """Test standing positions"""
    print("\n=== Testing Standing Positions ===")
    
    print("1. Moving to neutral position...")
    try:
        neutral_position(1)
        time.sleep(2)
        print("✓ Neutral position successful")
    except Exception as e:
        print(f"✗ Neutral position failed: {e}")
    
    print("2. Moving to tippytoes position...")
    try:
        tippytoes_position(1)
        time.sleep(2)
        print("✓ Tippytoes position successful")
    except Exception as e:
        print(f"✗ Tippytoes position failed: {e}")

def test_trot_forward():
    """Test trot_forward gait continuously at intensity 10"""
    print("\n=== Testing Trot Forward Gait (Continuous Loop) ===")
    print("Running trot_forward with intensity 10 in continuous loop...")
    print("Press Ctrl+C to stop the movement")
    
    try:
        while True:
            trot_forward(10)
            time.sleep(0.1)  # Small delay between cycles
    except Exception as e:
        print(f"✗ Trot forward failed: {e}")

def main():
    """Main test function"""
    print("=== PyBullet Simulation Test ===")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize simulation
    robot_id, joint_map = initialize_simulation()
    
    # Setup camera controls
    cam_controls = setup_camera_controls()
    
    print("\n=== Camera Controls ===")
    print("Use the sliders in the PyBullet GUI to control the camera view")
    
    # Run standing test first
    test_standing()
    
    # Step simulation to settle
    for _ in range(20):
        p.stepSimulation()
        time.sleep(0.1)
    
    # Run continuous trot_forward test
    test_trot_forward()
    
    print("\n=== Test Complete ===")
    print("Press Ctrl+C to exit...")
    
    # Keep simulation running
    try:
        while True:
            update_camera(cam_controls)
            p.stepSimulation()
            time.sleep(1.0/240.0)
    except KeyboardInterrupt:
        print("\nShutting down...")
        p.disconnect()

if __name__ == "__main__":
    main() 