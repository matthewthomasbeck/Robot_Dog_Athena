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

import RPi.GPIO as GPIO # import RPi.GPIO library for GPIO control
import pigpio # import pigpio library for PWM control
import time # import time library for time functions
import logging # import logging library for debugging

##### import config #####

import utilities.config as config # dynamically import config


########## CREATE DEPENDENCIES ##########

##### set channel data #####

# iterate through channels and assign neutral values of 1500
CHANNEL_DATA = {info['gpio_pin']: 1500 for info in config.RECEIVER_CHANNELS.values()}





###################################################
############### INITIALIZE RECEIVER ###############
###################################################


########## PWM DECODER ##########

class PWMDecoder: # class to decode PWM signals

    ##### initialize PWM signal decoding #####

    logging.debug("(receiver.py): Initializing PWM signal decoding...\n")

    def __init__(self, pi, gpio, callback): # function to initialize pwm signal decoding

        self.pi = pi # set pi
        self.gpio = gpio # set gpio
        self.callback = callback # set callback
        self.last_tick = None # set last tick
        self.tick = None # set tick
        self.pi.set_mode(gpio, pigpio.INPUT) # set gpio to input
        self.cb = self.pi.callback(gpio, pigpio.EITHER_EDGE, self._cbf) # set callback

    ##### decode PWM signal #####

    logging.debug("(receiver.py): Decoding PWM signal...\n")

    def _cbf(self, gpio, level, tick): # function to decode pwm signal

        if self.last_tick is not None: # if last tick is not None...
            self.tick = tick # set tick
            self.callback(gpio, tick - self.last_tick) # call callback with gpio and tick - last tick

        self.last_tick = tick # set last tick to tick

    ##### cancel PWM signal decoding #####

    def cancel(self): # function to cancel pwm signal decoding
        self.cb.cancel() # cancel callback


########## INITIALIZE RECEIVER ##########

def initialize_receiver(): # function to initialize receiver

    ##### initialize receiver #####

    logging.debug("(receiver.py): Initializing receiver...\n")

    try:
        GPIO.cleanup() # clean up GPIO to ensure no previous state affects the receiver
        GPIO.setmode(GPIO.BCM) # set GPIO mode to BCM as standard

        try: # try to clean up any old pigpio session

            temp_pi = pigpio.pi() # create temporary pigpio instance
            if temp_pi.connected: # if pigpio instance is connected...
                temp_pi.stop() # stop any old pigpio instance
                logging.debug("(receiver.py): Cleaned up old pigpio instance.\n")

        except Exception as e:
            logging.warning(f"(receiver.py): No previous pigpio instance to stop: {e}\n")

        pi = pigpio.pi() # create a fresh pigpio instance
        if not pi.connected: # if pigpio instance is not connected...
            logging.error("(receiver.py): Failed to connect to pigpio daemon. Exiting...\n")
            exit(1)

        decoders = [] # list to hold PWM decoders
        for channel in config.RECEIVER_CHANNELS.values(): # iterate through channels in config
            decoder = PWMDecoder(pi, channel['gpio_pin'], _pwm_callback) # create PWM decoder for each channel
            decoders.append(decoder) # append decoder to list

        logging.info("(receiver.py): Receiver initialized with pigpio + PWM decoders.\n")

        return pi, CHANNEL_DATA # return pigpio instance, list of decoders, and channel data

    except Exception as e:
        logging.error(f"(receiver.py): Receiver initialization failed: {e}\n")
        exit(1)


########## PWM CALLBACK ##########

def _pwm_callback(gpio, pulseWidth):  # function to set pulse width to channel data

    ##### set pulse width to channel data #####

    CHANNEL_DATA[gpio] = pulseWidth  # set channel data to pulse width


########## JOYSTICK INTENSITY ##########

def _get_joystick_intensity(pulse_width, min_val, max_val): # return a value from 1-10 based on intensity

    ##### calculate intensity based on pulse width #####

    # very annoying, leave commented
    #logging.debug(f"(receiver.py): Calculating joystick intensity for pulse width {pulse_width}...\n")

    try: # try to calculate joystick intensity

        if pulse_width < min_val:
            intensity = int(10 * (min_val - pulse_width) / (min_val - 1000)) # scale intensity downwards
        elif pulse_width > max_val:
            intensity = int(10 * (pulse_width - max_val) / (2000 - max_val)) # scale intensity upwards
        else:
            return 0 # neutral position, no intensity

        return max(1, min(intensity, 10))  # Ensure value is between 1-10

    except Exception as e:
        logging.error(f"(receiver.py): Failed to calculate joystick intensity: {e}\n")
        return 0


########## COMMAND INTERPRETER ##########

def interpret_commands(channel_data):
    ##### interpret commands from channel data #####

    # logging.debug("(receiver.py): Interpreting commands from channel data...\n") # very annoying, leave commented
    current_time = time.time()
    commands = {
        'channel_0': ('NEUTRAL', 0),
        'channel_1': ('NEUTRAL', 0),
        'channel_2': ('NEUTRAL', 0),
        'channel_7': ('NEUTRAL', 0)
    }

    ##### tilt channel 0 #####
    pin = config.RECEIVER_CHANNELS['channel_0']['gpio_pin']
    if channel_data[pin] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.RECEIVER_CHANNELS['channel_0']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_0']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_0']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_0']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_0']['counter'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel_0'] = ('TILT DOWN', 0)
            config.RECEIVER_CHANNELS['channel_0']['counter'] = 0
    elif channel_data[pin] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.RECEIVER_CHANNELS['channel_0']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_0']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_0']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_0']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_0']['counter'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel_0'] = ('TILT UP', 0)
            config.RECEIVER_CHANNELS['channel_0']['counter'] = 0

    ##### trigger channel 1 #####
    pin = config.RECEIVER_CHANNELS['channel_1']['gpio_pin']
    if channel_data[pin] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.RECEIVER_CHANNELS['channel_1']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_1']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_1']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_1']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_1']['counter'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel_1'] = ('NEUTRAL', 0)
            config.RECEIVER_CHANNELS['channel_1']['counter'] = 0
    elif channel_data[pin] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.RECEIVER_CHANNELS['channel_1']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_1']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_1']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_1']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_1']['counter'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel_1'] = ('TRIGGER SHOOT', 0)
            config.RECEIVER_CHANNELS['channel_1']['counter'] = 0

    ##### squat channel 2 #####
    pin = config.RECEIVER_CHANNELS['channel_2']['gpio_pin']
    if channel_data[pin] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.RECEIVER_CHANNELS['channel_2']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_2']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_2']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_2']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_2']['counter'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel_2'] = ('SQUAT DOWN', 0)
            config.RECEIVER_CHANNELS['channel_2']['counter'] = 0
    elif channel_data[pin] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.RECEIVER_CHANNELS['channel_2']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_2']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_2']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_2']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_2']['counter'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel_2'] = ('SQUAT UP', 0)
            config.RECEIVER_CHANNELS['channel_2']['counter'] = 0

    ##### rotate channel 3 #####
    pin = config.RECEIVER_CHANNELS['channel_3']['gpio_pin']
    if channel_data[pin] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.RECEIVER_CHANNELS['channel_3']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_3']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_3']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_3']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_3']['counter'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            intensity = _get_joystick_intensity(channel_data[pin], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH'])
            commands['channel_3'] = ('ROTATE LEFT', intensity)
            config.RECEIVER_CHANNELS['channel_3']['counter'] = 0
    elif config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'] <= channel_data[pin] <= config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.RECEIVER_CHANNELS['channel_3']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_3']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_3']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_3']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_3']['counter'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            commands['channel_3'] = ('NEUTRAL', 0)
            config.RECEIVER_CHANNELS['channel_3']['counter'] = 0
    elif channel_data[pin] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.RECEIVER_CHANNELS['channel_3']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_3']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_3']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_3']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_3']['counter'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            intensity = _get_joystick_intensity(channel_data[pin], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH'])
            commands['channel_3'] = ('ROTATE RIGHT', intensity)
            config.RECEIVER_CHANNELS['channel_3']['counter'] = 0

    ##### look channel 4 #####
    pin = config.RECEIVER_CHANNELS['channel_4']['gpio_pin']
    if channel_data[pin] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.RECEIVER_CHANNELS['channel_4']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_4']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_4']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_4']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_4']['counter'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            intensity = _get_joystick_intensity(channel_data[pin], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH'])
            commands['channel_4'] = ('LOOK DOWN', intensity)
            config.RECEIVER_CHANNELS['channel_4']['counter'] = 0
    elif config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'] <= channel_data[pin] <= config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.RECEIVER_CHANNELS['channel_4']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_4']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_4']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_4']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_4']['counter'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            commands['channel_4'] = ('NEUTRAL', 0)
            config.RECEIVER_CHANNELS['channel_4']['counter'] = 0
    elif channel_data[pin] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.RECEIVER_CHANNELS['channel_4']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_4']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_4']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_4']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_4']['counter'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            intensity = _get_joystick_intensity(channel_data[pin], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH'])
            commands['channel_4'] = ('LOOK UP', intensity)
            config.RECEIVER_CHANNELS['channel_4']['counter'] = 0

    ##### move channel 5 #####
    pin = config.RECEIVER_CHANNELS['channel_5']['gpio_pin']
    if channel_data[pin] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.RECEIVER_CHANNELS['channel_5']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_5']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_5']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_5']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_5']['counter'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            commands['channel_5'] = ('MOVE FORWARD', _get_joystick_intensity(channel_data[pin], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']))
            config.RECEIVER_CHANNELS['channel_5']['counter'] = 0
    elif config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'] <= channel_data[pin] <= config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.RECEIVER_CHANNELS['channel_5']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_5']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_5']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_5']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_5']['counter'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            commands['channel_5'] = ('NEUTRAL', 0)
            config.RECEIVER_CHANNELS['channel_5']['counter'] = 0
    elif channel_data[pin] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.RECEIVER_CHANNELS['channel_5']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_5']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_5']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_5']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_5']['counter'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            commands['channel_5'] = ('MOVE BACKWARD', _get_joystick_intensity(channel_data[pin], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']))
            config.RECEIVER_CHANNELS['channel_5']['counter'] = 0

    ##### shift channel 6 #####
    pin = config.RECEIVER_CHANNELS['channel_6']['gpio_pin']
    if channel_data[pin] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.RECEIVER_CHANNELS['channel_6']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_6']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_6']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_6']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_6']['counter'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            commands['channel_6'] = ('SHIFT LEFT', _get_joystick_intensity(channel_data[pin], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']))
            config.RECEIVER_CHANNELS['channel_6']['counter'] = 0
    elif config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'] <= channel_data[pin] <= config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.RECEIVER_CHANNELS['channel_6']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_6']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_6']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_6']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_6']['counter'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            commands['channel_6'] = ('NEUTRAL', 0)
            config.RECEIVER_CHANNELS['channel_6']['counter'] = 0
    elif channel_data[pin] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.RECEIVER_CHANNELS['channel_6']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_6']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_6']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_6']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_6']['counter'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            commands['channel_6'] = ('SHIFT RIGHT', _get_joystick_intensity(channel_data[pin], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']))
            config.RECEIVER_CHANNELS['channel_6']['counter'] = 0

    ##### extra channel 7 #####
    pin = config.RECEIVER_CHANNELS['channel_7']['gpio_pin']
    if channel_data[pin] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.RECEIVER_CHANNELS['channel_7']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_7']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_7']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_7']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_7']['counter'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel_7'] = ('+', 0)
            config.RECEIVER_CHANNELS['channel_7']['counter'] = 0
    elif channel_data[pin] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.RECEIVER_CHANNELS['channel_7']['timestamp'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.RECEIVER_CHANNELS['channel_7']['counter'] = 0
        config.RECEIVER_CHANNELS['channel_7']['counter'] += 1
        config.RECEIVER_CHANNELS['channel_7']['timestamp'] = current_time
        if config.RECEIVER_CHANNELS['channel_7']['counter'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel_7'] = ('-', 0)
            config.RECEIVER_CHANNELS['channel_7']['counter'] = 0

    return commands
