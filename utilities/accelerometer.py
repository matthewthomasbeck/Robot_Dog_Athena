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
import time


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

        BUS.write_byte_data(config.ACCELEROMETER_CONFIG['ACCELEROMETER_ADDRESS'], config.ACCELEROMETER_CONFIG['SMPLRT_DIV'], 7)	# write to sample rate register
        BUS.write_byte_data(config.ACCELEROMETER_CONFIG['ACCELEROMETER_ADDRESS'], config.ACCELEROMETER_CONFIG['PWR_MGMT_1'], 1) # write to power management register
        BUS.write_byte_data(config.ACCELEROMETER_CONFIG['ACCELEROMETER_ADDRESS'], config.ACCELEROMETER_CONFIG['CONFIG_REGISTER'], 0) # write to configuration register
        BUS.write_byte_data(config.ACCELEROMETER_CONFIG['ACCELEROMETER_ADDRESS'], config.ACCELEROMETER_CONFIG['GYRO_CONFIG'], 24) # write to gyro configuration register
        BUS.write_byte_data(config.ACCELEROMETER_CONFIG['ACCELEROMETER_ADDRESS'], config.ACCELEROMETER_CONFIG['INT_ENABLE'], 1) # write to interrupt enable register

        logging.info(f"(accelerometer.py): Accelerometer initialized successfully.\n")

    except Exception as e: # if error initializing accelerometer...
        logging.error(f"(accelerometer.py): Error initializing accelerometer: {e}\n")
        return None


########## READ ALL DATA ##########

def get_all_data(): # function to read all data from accelerometer (and gyroscope)

    ##### read accelerometer data #####

    acc_x = get_device_data(ACCEL_XOUT_H)
	acc_y = get_device_data(ACCEL_YOUT_H)
	acc_z = get_device_data(ACCEL_ZOUT_H)

    ##### read gyroscope data #####

	gyro_x = get_device_data(GYRO_XOUT_H)
	gyro_y = get_device_data(GYRO_YOUT_H)
	gyro_z = get_device_data(GYRO_ZOUT_H)

    ##### calculate data #####

	shift = acc_x/16384.0 # negative value shifts to the right
	move = -acc_y/16384.0 # flipped sign to correct for upside-down orientation, positive value moves forward
	translate = acc_z/16384.0  # positive value translates up
	
	yaw = -gyro_x/131.0 # flipped sign to correct for upside-down orientation, positive value yaw right
	roll = gyro_y/131.0 # positive value roll right
	pitch = -gyro_z/131.0 # flipped sign to correct for upside-down orientation, positive value pitch up

    return shift, move, translate, yaw, roll, pitch


########## READ INDIVIDUAL DATA ##########

def get_individual_data(addr): # function to read orientation data from accelerometer (and gyroscope)

    try: # try to read orientation data from accelerometer (and gyroscope)

        ##### set variables #####

        high = bus.read_byte_data(Device_Address, addr) # read higher byte
        low = bus.read_byte_data(Device_Address, addr+1) # read lower byte
        value = ((high << 8) | low) # concatenate higher and lower byte
        
        if (value > 32768): # if value is greater than 32768...
            value = value - 65536 # subtract 65536 from value

        return value

    except Exception as e: # if error reading orientation data...
        logging.error(f"(accelerometer.py): Error reading orientation data: {e}\n")
        return None
