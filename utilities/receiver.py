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

# iterate through GPIO_PINS and assign neutral values of 1500
CHANNEL_DATA = {pin: 1500 for pin in list(config.GPIO_PINS.values())}





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
        pi = pigpio.pi() # create pigpio instance
        if not pi.connected: # if pigpio instance is not connected...
            logging.error("(receiver.py): Failed to connect to pigpio daemon. Exiting...\n")
            exit(1)

        GPIO.setmode(GPIO.BCM) # set GPIO mode to BCM as standard
        decoders = [] # list to hold PWM decoders
        for pin in config.GPIO_PINS.values(): # iterate through GPIO pins
            decoder = PWMDecoder(pi, pin, _pwm_callback) # create PWM decoder for each pin
            decoders.append(decoder) # add decoder to list

        logging.info("(receiver.py): Receiver initialized with pigpio + PWM decoders.\n")

        return pi, decoders, CHANNEL_DATA # return pigpio instance, list of decoders, and channel data

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

def interpret_commands(channel_data): # function to interpret commands from PWM signal data

    ##### interpret commands from PWM signal data #####

    #logging.debug("(receiver.py): Interpreting commands from channel data...\n") # very annoying, leave commented
    current_time = time.time()  # set current time to current time
    commands = { # create dictionary of commands

        'channel-0': ('NEUTRAL', 0), # set channel 0 to neutral for silence
        'channel-1': ('NEUTRAL', 0), # set channel 1 to neutral for silence
        'channel-2': ('NEUTRAL', 0), # set channel 2 to neutral for silence
        'channel-7': ('NEUTRAL', 0), # set channel 7 to neutral for silence
    }

    ##### tilt channel 0 #####

    if channel_data[config.GPIO_PINS['tiltUpDownChannel0']] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-0'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-0'] = 0
        config.CHANNEL_COUNTERS['channel-0'] += 1
        config.CHANNEL_TIMESTAMPS['channel-0'] = current_time
        if config.CHANNEL_COUNTERS['channel-0'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel-0'] = ('TILT DOWN', 0)
            config.CHANNEL_COUNTERS['channel-0'] = 0
    elif channel_data[config.GPIO_PINS['tiltUpDownChannel0']] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-0'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-0'] = 0
        config.CHANNEL_COUNTERS['channel-0'] += 1
        config.CHANNEL_TIMESTAMPS['channel-0'] = current_time
        if config.CHANNEL_COUNTERS['channel-0'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel-0'] = ('TILT UP', 0)
            config.CHANNEL_COUNTERS['channel-0'] = 0

    ##### trigger channel 1 #####

    if channel_data[config.GPIO_PINS['triggerShootChannel1']] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-1'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-1'] = 0
        config.CHANNEL_COUNTERS['channel-1'] += 1
        config.CHANNEL_TIMESTAMPS['channel-1'] = current_time
        if config.CHANNEL_COUNTERS['channel-1'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel-1'] = ('NEUTRAL', 0)
            config.CHANNEL_COUNTERS['channel-1'] = 0
    elif channel_data[config.GPIO_PINS['triggerShootChannel1']] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-1'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-1'] = 0
        config.CHANNEL_COUNTERS['channel-1'] += 1
        config.CHANNEL_TIMESTAMPS['channel-1'] = current_time
        if config.CHANNEL_COUNTERS['channel-1'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel-1'] = ('TRIGGER SHOOT', 0)
            config.CHANNEL_COUNTERS['channel-1'] = 0

    ##### squat channel 2 #####

    if channel_data[config.GPIO_PINS['squatUpDownChannel2']] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-2'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-2'] = 0
        config.CHANNEL_COUNTERS['channel-2'] += 1
        config.CHANNEL_TIMESTAMPS['channel-2'] = current_time
        if config.CHANNEL_COUNTERS['channel-2'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel-2'] = ('SQUAT DOWN', 0)
            config.CHANNEL_COUNTERS['channel-2'] = 0
    elif channel_data[config.GPIO_PINS['squatUpDownChannel2']] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-2'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-2'] = 0
        config.CHANNEL_COUNTERS['channel-2'] += 1
        config.CHANNEL_TIMESTAMPS['channel-2'] = current_time
        if config.CHANNEL_COUNTERS['channel-2'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel-2'] = ('SQUAT UP', 0)
            config.CHANNEL_COUNTERS['channel-2'] = 0

    ##### rotation channel 3 #####

    if channel_data[config.GPIO_PINS['rotateLeftRightChannel3']] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-3'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-3'] = 0
        config.CHANNEL_COUNTERS['channel-3'] += 1
        config.CHANNEL_TIMESTAMPS['channel-3'] = current_time

        if config.CHANNEL_COUNTERS['channel-3'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            intensity = _get_joystick_intensity(channel_data[config.GPIO_PINS['rotateLeftRightChannel3']], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH'])
            commands['channel-3'] = ('ROTATE LEFT', intensity)
            config.CHANNEL_COUNTERS['channel-3'] = 0

    elif config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'] <= channel_data[config.GPIO_PINS['rotateLeftRightChannel3']] <= config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-3'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-3'] = 0
        config.CHANNEL_COUNTERS['channel-3'] += 1
        config.CHANNEL_TIMESTAMPS['channel-3'] = current_time

        if config.CHANNEL_COUNTERS['channel-3'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            commands['channel-3'] = ('NEUTRAL', 0)
            config.CHANNEL_COUNTERS['channel-3'] = 0

    elif channel_data[config.GPIO_PINS['rotateLeftRightChannel3']] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-3'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-3'] = 0
        config.CHANNEL_COUNTERS['channel-3'] += 1
        config.CHANNEL_TIMESTAMPS['channel-3'] = current_time

        if config.CHANNEL_COUNTERS['channel-3'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            intensity = _get_joystick_intensity(channel_data[config.GPIO_PINS['rotateLeftRightChannel3']], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH'])
            commands['channel-3'] = ('ROTATE RIGHT', intensity)
            config.CHANNEL_COUNTERS['channel-3'] = 0

    ##### look channel 4 #####

    if channel_data[config.GPIO_PINS['lookUpDownChannel4']] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-4'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-4'] = 0
        config.CHANNEL_COUNTERS['channel-4'] += 1
        config.CHANNEL_TIMESTAMPS['channel-4'] = current_time

        if config.CHANNEL_COUNTERS['channel-4'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            intensity = _get_joystick_intensity(channel_data[config.GPIO_PINS['lookUpDownChannel4']], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH'])
            commands['channel-4'] = ('LOOK DOWN', intensity)
            config.CHANNEL_COUNTERS['channel-4'] = 0

    elif config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'] <= channel_data[config.GPIO_PINS['lookUpDownChannel4']] <= config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-4'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-4'] = 0
        config.CHANNEL_COUNTERS['channel-4'] += 1
        config.CHANNEL_TIMESTAMPS['channel-4'] = current_time

        if config.CHANNEL_COUNTERS['channel-4'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            commands['channel-4'] = ('NEUTRAL', 0)
            config.CHANNEL_COUNTERS['channel-4'] = 0

    elif channel_data[config.GPIO_PINS['lookUpDownChannel4']] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-4'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-4'] = 0
        config.CHANNEL_COUNTERS['channel-4'] += 1
        config.CHANNEL_TIMESTAMPS['channel-4'] = current_time

        if config.CHANNEL_COUNTERS['channel-4'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            intensity = _get_joystick_intensity(channel_data[config.GPIO_PINS['lookUpDownChannel4']], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH'])
            commands['channel-4'] = ('LOOK UP', intensity)
            config.CHANNEL_COUNTERS['channel-4'] = 0

    ##### move channel 5 #####

    if channel_data[config.GPIO_PINS['moveForwardBackwardChannel5']] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-5'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-5'] = 0
        config.CHANNEL_COUNTERS['channel-5'] += 1
        config.CHANNEL_TIMESTAMPS['channel-5'] = current_time

        if config.CHANNEL_COUNTERS['channel-5'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            intensity = _get_joystick_intensity(channel_data[config.GPIO_PINS['moveForwardBackwardChannel5']], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH'])
            commands['channel-5'] = ('MOVE FORWARD', intensity)
            config.CHANNEL_COUNTERS['channel-5'] = 0

    elif config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'] <= channel_data[config.GPIO_PINS['moveForwardBackwardChannel5']] <= config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-5'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-5'] = 0
        config.CHANNEL_COUNTERS['channel-5'] += 1
        config.CHANNEL_TIMESTAMPS['channel-5'] = current_time

        if config.CHANNEL_COUNTERS['channel-5'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            commands['channel-5'] = ('NEUTRAL', 0)
            config.CHANNEL_COUNTERS['channel-5'] = 0

    elif channel_data[config.GPIO_PINS['moveForwardBackwardChannel5']] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-5'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-5'] = 0
        config.CHANNEL_COUNTERS['channel-5'] += 1
        config.CHANNEL_TIMESTAMPS['channel-5'] = current_time

        if config.CHANNEL_COUNTERS['channel-5'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            intensity = _get_joystick_intensity(channel_data[config.GPIO_PINS['moveForwardBackwardChannel5']], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH'])
            commands['channel-5'] = ('MOVE BACKWARD', intensity)
            config.CHANNEL_COUNTERS['channel-5'] = 0

    ##### shift channel 6 #####

    if channel_data[config.GPIO_PINS['shiftLeftRightChannel6']] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-6'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-6'] = 0
        config.CHANNEL_COUNTERS['channel-6'] += 1
        config.CHANNEL_TIMESTAMPS['channel-6'] = current_time

        if config.CHANNEL_COUNTERS['channel-6'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            intensity = _get_joystick_intensity(channel_data[config.GPIO_PINS['shiftLeftRightChannel6']], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH'])
            commands['channel-6'] = ('SHIFT LEFT', intensity)
            config.CHANNEL_COUNTERS['channel-6'] = 0

    elif config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'] <= channel_data[config.GPIO_PINS['shiftLeftRightChannel6']] <= config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-6'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-6'] = 0
        config.CHANNEL_COUNTERS['channel-6'] += 1
        config.CHANNEL_TIMESTAMPS['channel-6'] = current_time

        if config.CHANNEL_COUNTERS['channel-6'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            commands['channel-6'] = ('NEUTRAL', 0)
            config.CHANNEL_COUNTERS['channel-6'] = 0

    elif channel_data[config.GPIO_PINS['shiftLeftRightChannel6']] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-6'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-6'] = 0
        config.CHANNEL_COUNTERS['channel-6'] += 1
        config.CHANNEL_TIMESTAMPS['channel-6'] = current_time

        if config.CHANNEL_COUNTERS['channel-6'] >= config.SIGNAL_TUNING_CONFIG['JOYSTICK_THRESHOLD']:
            intensity = _get_joystick_intensity(channel_data[config.GPIO_PINS['shiftLeftRightChannel6']], config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW'], config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH'])
            commands['channel-6'] = ('SHIFT RIGHT', intensity)
            config.CHANNEL_COUNTERS['channel-6'] = 0

    ##### extra channel 7 #####

    if channel_data[config.GPIO_PINS['extraChannel7']] < config.SIGNAL_TUNING_CONFIG['DEADBAND_LOW']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-7'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-7'] = 0
        config.CHANNEL_COUNTERS['channel-7'] += 1
        config.CHANNEL_TIMESTAMPS['channel-7'] = current_time
        if config.CHANNEL_COUNTERS['channel-7'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel-7'] = ('+', 0)
            config.CHANNEL_COUNTERS['channel-7'] = 0
    elif channel_data[config.GPIO_PINS['extraChannel7']] > config.SIGNAL_TUNING_CONFIG['DEADBAND_HIGH']:
        if current_time - config.CHANNEL_TIMESTAMPS['channel-7'] > config.SIGNAL_TUNING_CONFIG['TIME_FRAME']:
            config.CHANNEL_COUNTERS['channel-7'] = 0
        config.CHANNEL_COUNTERS['channel-7'] += 1
        config.CHANNEL_TIMESTAMPS['channel-7'] = current_time
        if config.CHANNEL_COUNTERS['channel-7'] >= config.SIGNAL_TUNING_CONFIG['TOGGLE_THRESHOLD']:
            commands['channel-7'] = ('-', 0)
            config.CHANNEL_COUNTERS['channel-7'] = 0

    return commands # return channel requests
