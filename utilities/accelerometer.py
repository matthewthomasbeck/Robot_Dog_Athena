##################################################################################
# Copyright (c) 2025 Matthew Thomas Beck                                         #
#                                                                                #
# Licensed under the Creative Commons Attribution-NonCommercial 4.0              #
# International (CC BY-NC 4.0). Personal and educational use is permitted.       #
# Commercial use by companies or for-profit entities is prohibited.              #
##################################################################################





############################################################
############### IMPORT / CREATE DEPENDENCIES ###############
############################################################


########## IMPORT DEPENDENCIES ##########

##### import config #####

import utilities.config as config

##### import necessary libraries #####

import logging
import smbus


########## CREATE DEPENDENCIES ##########

##### setup variables #####

BUS = smbus.SMBus(1) # used for I2C communication





#######################################################
############### ACCELEROMETER FUNCTIONS ###############
#######################################################


########## INITIALIZE ACCELEROMETER ##########

def initialize_accelerometer(): # function to initialize accelerometer

    try: # try to initialize accelerometer

        ##### write to registers #####

        logging.debug(f"(accelerometer.py): Initializing accelerometer...\n")

        BUS.write_byte_data(config.ACCELEROMETER_CONFIG['MPU_6050_ADDRESS'], config.ACCELEROMETER_CONFIG['SMPLRT_DIV'], 7)	# write to sample rate register
        BUS.write_byte_data(config.ACCELEROMETER_CONFIG['MPU_6050_ADDRESS'], config.ACCELEROMETER_CONFIG['PWR_MGMT_1'], 1) # write to power management register
        BUS.write_byte_data(config.ACCELEROMETER_CONFIG['MPU_6050_ADDRESS'], config.ACCELEROMETER_CONFIG['CONFIG_REGISTER'], 0) # write to configuration register
        BUS.write_byte_data(config.ACCELEROMETER_CONFIG['MPU_6050_ADDRESS'], config.ACCELEROMETER_CONFIG['GYRO_CONFIG'], 24) # write to gyro configuration register
        BUS.write_byte_data(config.ACCELEROMETER_CONFIG['MPU_6050_ADDRESS'], config.ACCELEROMETER_CONFIG['INT_ENABLE'], 1) # write to interrupt enable register

        logging.info(f"(accelerometer.py): Accelerometer initialized successfully.\n")

    except Exception as e: # if error initializing accelerometer...
        logging.error(f"(accelerometer.py): Error initializing accelerometer: {e}\n")
        return None


########## READ ALL DATA ##########

def get_all_data(): # function to read all data from accelerometer (and gyroscope)

    ##### read accelerometer data #####

    acc_x = get_orientation_datapoint(config.ACCELEROMETER_CONFIG['ACCEL_XOUT_H'])
    acc_y = get_orientation_datapoint(config.ACCELEROMETER_CONFIG['ACCEL_YOUT_H'])
    acc_z = get_orientation_datapoint(config.ACCELEROMETER_CONFIG['ACCEL_ZOUT_H'])

    ##### read gyroscope data #####

    gyro_x = get_orientation_datapoint(config.ACCELEROMETER_CONFIG['GYRO_XOUT_H'])
    gyro_y = get_orientation_datapoint(config.ACCELEROMETER_CONFIG['GYRO_YOUT_H'])
    gyro_z = get_orientation_datapoint(config.ACCELEROMETER_CONFIG['GYRO_ZOUT_H'])

    ##### calculate data #####

    shift = acc_x/16384.0 # negative value shifts to the right
    move = -acc_y/16384.0 # flipped sign to correct for upside-down orientation, positive value moves forward
    translate = acc_z/16384.0  # positive value translates up

    yaw = -gyro_x/131.0 # flipped sign to correct for upside-down orientation, positive value yaw right
    roll = gyro_y/131.0 # positive value roll right
    pitch = -gyro_z/131.0 # flipped sign to correct for upside-down orientation, positive value pitch up

    return shift, move, translate, yaw, roll, pitch


########## ISAAC LAB ORIENTATION VECTORS ##########

def get_orientation_vectors():
    """
    Construct IMU-derived vectors in the format expected by the Isaac Lab-style observation.

    Returns:
        A dict with:
        - base_lin_vel: [3] (m/s)  (zeroed; no odometry)
        - base_ang_vel: [3] (rad/s) [wx, wy, wz] in body frame
        - projected_gravity: [3] (unit vector) gravity direction in body frame

    Notes:
        - Uses your calibrated `get_all_data()` mapping.
        - Axis mapping to Isaac Lab body frame: X=forward, Y=left, Z=up.
          Your calibrated signals imply: [X, Y, Z] = [move, shift, translate].
        - Gyro is returned as [wx, wy, wz] = [roll, pitch, yaw] converted to rad/s.
    """

    shift, move, translate, yaw, roll, pitch = get_all_data()

    # Isaac Lab body frame convention: +X forward, +Y left, +Z up
    # Your calibrated accel: move=forward, shift=left/right, translate=up/down
    ax = float(move)
    ay = float(shift)
    az = float(translate)

    # Projected gravity: normalize accelerometer vector (works well when gravity dominates).
    norm = (ax * ax + ay * ay + az * az) ** 0.5
    if norm > 1e-6:
        projected_gravity = [ax / norm, ay / norm, az / norm]
    else:
        # Default to upright if something goes wrong
        projected_gravity = [0.0, 0.0, -1.0]

    # Angular velocity: deg/s -> rad/s; and map to [wx, wy, wz] = [roll, pitch, yaw]
    deg2rad = 3.141592653589793 / 180.0
    wx = float(roll) * deg2rad
    wy = float(pitch) * deg2rad
    wz = float(yaw) * deg2rad
    base_ang_vel = [wx, wy, wz]

    # No base linear velocity estimate available (no odometry)
    base_lin_vel = [0.0, 0.0, 0.0]

    return {
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "projected_gravity": projected_gravity,
        # keep raw around for debugging if needed
        "raw": {
            "shift": shift,
            "move": move,
            "translate": translate,
            "yaw": yaw,
            "roll": roll,
            "pitch": pitch,
        },
    }


########## READ INDIVIDUAL DATA ##########

def get_orientation_datapoint(addr): # function to read orientation data from accelerometer (and gyroscope)

    try: # try to read orientation data from accelerometer (and gyroscope)

        ##### set variables #####

        high = BUS.read_byte_data(config.ACCELEROMETER_CONFIG['MPU_6050_ADDRESS'], addr) # read higher byte
        low = BUS.read_byte_data(config.ACCELEROMETER_CONFIG['MPU_6050_ADDRESS'], addr+1) # read lower byte
        value = ((high << 8) | low) # concatenate higher and lower byte

        if (value > 32768): # if value is greater than 32768...
            value = value - 65536 # subtract 65536 from value

        return value

    except Exception as e: # if error reading orientation data...
        logging.error(f"(accelerometer.py): Error reading orientation data: {e}\n")
        return None
