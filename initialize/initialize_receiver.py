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

##### control libraries #####

import pigpio # import pigpio library for PWM control
import time # import time library for time functions


########## CREATE DEPENDENCIES ##########

##### set counters and timestamps for each channel #####

CHANNEL_COUNTERS = { # set channel counters to 0 for command requests

    'channel-0': 0, 'channel-1': 0, 'channel-2': 0, 'channel-3': 0, 'channel-4': 0, 'channel-5': 0, 'channel-6': 0,
    'channel-7': 0
}

CHANNEL_TIMESTAMPS = { # set channel timestamps to current time for command requests

    'channel-0': time.time(), 'channel-1': time.time(), 'channel-2': time.time(), 'channel-3': time.time(),
    'channel-4': time.time(), 'channel-5': time.time(), 'channel-6': time.time(), 'channel-7': time.time()
}

##### command registry hyperparameters #####

JOYSTICK_THRESHOLD = 40 # number of times condition must be met to trigger a request on a joystick channel
TOGGLE_THRESHOLD = 40 # number of times condition must be met to trigger a request on a button channel
TIME_FRAME = 0.10017 # time frame for condition to be met, default: 0.100158
DEADBAND_HIGH = 1600 # deadband high for PWM signal
DEADBAND_LOW = 1400 # deadband low for PWM signal

##### declare movement channel GPIO pins #####

tiltUpDownChannel0 = 17 # default: 17
triggerShootChannel1 = 27 # default: 27
squatUpDownChannel2 = 22 # default: 22
rotateLeftRightChannel3 = 5 # default: 5
lookUpDownChannel4 = 6 # default: 6
moveForwardBackwardChannel5 = 13 # default: 13
shiftLeftRightChannel6 = 19 # default: 19

##### declare extra channel GPIO pins #####

extraChannel7 = 26 # default: 26

##### declare utilized pwm pins #####

PWM_PINS = [ # define PWM pins

    tiltUpDownChannel0, # default: 17
    triggerShootChannel1, # default: 27
    squatUpDownChannel2, # default: 22
    rotateLeftRightChannel3, # default: 5
    lookUpDownChannel4, # default: 6
    moveForwardBackwardChannel5, # default: 13
    shiftLeftRightChannel6, # default: 19
    extraChannel7, # default: 26
]





###################################################
############### INITIALIZE RECEIVER ###############
###################################################


########## PWM DECODER ##########

class PWMDecoder: # class to decode PWM signals

    #####  initialize PWM signal decoding #####

    def __init__(self, pi, gpio, callback): # function to initialize pwm signal decoding

        self.pi = pi # set pi
        self.gpio = gpio # set gpio
        self.callback = callback # set callback
        self.last_tick = None # set last tick
        self.tick = None # set tick
        self.pi.set_mode(gpio, pigpio.INPUT) # set gpio to input
        self.cb = self.pi.callback(gpio, pigpio.EITHER_EDGE, self._cbf) # set callback

    ##### decode PWM signal #####

    def _cbf(self, gpio, level, tick): # function to decode pwm signal

        if self.last_tick is not None: # if last tick is not None...

            self.tick = tick # set tick
            self.callback(gpio, tick - self.last_tick) # call callback with gpio and tick - last tick

        self.last_tick = tick # set last tick to tick

    ##### cancel PWM signal decoding #####

    def cancel(self): # function to cancel pwm signal decoding

        self.cb.cancel() # cancel callback


########## COMMAND INTERPRETER ##########

def interpretCommands(channel_data): # function to interpret commands from PWM signal data

    ##### set variables #####

    commands = { # create dictionary of commands

        'channel-0': 'NEUTRAL', # set channel 0 to neutral for silence
        'channel-1': 'NEUTRAL', # set channel 1 to neutral for silence
        'channel-2': 'NEUTRAL', # set channel 2 to neutral for silence
        'channel-3': 'NEUTRAL', # set channel 3 to neutral for silence
        'channel-4': 'NEUTRAL', # set channel 4 to neutral for silence
        'channel-5': 'NEUTRAL', # set channel 5 to neutral for silence
        'channel-6': 'NEUTRAL', # set channel 6 to neutral for silence
        'channel-7': 'NEUTRAL' # set channel 7 to neutral for silence
    }

    current_time = time.time() # set current time to current time

    ##### tilt channel 0 #####

    if channel_data[tiltUpDownChannel0] < DEADBAND_LOW: # if the pulse width is less than DEADBAND_LOW...
        if current_time - CHANNEL_TIMESTAMPS['channel-0'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-0'] = 0
        CHANNEL_COUNTERS['channel-0'] += 1
        CHANNEL_TIMESTAMPS['channel-0'] = current_time
        if CHANNEL_COUNTERS['channel-0'] >= TOGGLE_THRESHOLD:
            commands['channel-0'] = 'TILT DOWN'
            CHANNEL_COUNTERS['channel-0'] = 0
    elif channel_data[tiltUpDownChannel0] > DEADBAND_HIGH: # if the pulse width is greater than DEADBAND_HIGH...
        if current_time - CHANNEL_TIMESTAMPS['channel-0'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-0'] = 0
        CHANNEL_COUNTERS['channel-0'] += 1
        CHANNEL_TIMESTAMPS['channel-0'] = current_time
        if CHANNEL_COUNTERS['channel-0'] >= TOGGLE_THRESHOLD:
            commands['channel-0'] = 'TILT UP'
            CHANNEL_COUNTERS['channel-0'] = 0

    ##### trigger channel 1 #####

    if channel_data[triggerShootChannel1] < DEADBAND_LOW: # if the pulse width is less than DEADBAND_LOW...
        if current_time - CHANNEL_TIMESTAMPS['channel-1'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-1'] = 0
        CHANNEL_COUNTERS['channel-1'] += 1
        CHANNEL_TIMESTAMPS['channel-1'] = current_time
        if CHANNEL_COUNTERS['channel-1'] >= TOGGLE_THRESHOLD:
            commands['channel-1'] = 'NEUTRAL'
            CHANNEL_COUNTERS['channel-1'] = 0
    elif channel_data[triggerShootChannel1] > DEADBAND_HIGH: # if the pulse width is greater than DEADBAND_HIGH...
        if current_time - CHANNEL_TIMESTAMPS['channel-1'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-1'] = 0
        CHANNEL_COUNTERS['channel-1'] += 1
        CHANNEL_TIMESTAMPS['channel-1'] = current_time
        if CHANNEL_COUNTERS['channel-1'] >= TOGGLE_THRESHOLD:
            commands['channel-1'] = 'TRIGGER SHOOT'
            CHANNEL_COUNTERS['channel-1'] = 0

    ##### squat channel 2 #####

    if channel_data[squatUpDownChannel2] < DEADBAND_LOW: # if the pulse width is less than DEADBAND_LOW...
        if current_time - CHANNEL_TIMESTAMPS['channel-2'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-2'] = 0
        CHANNEL_COUNTERS['channel-2'] += 1
        CHANNEL_TIMESTAMPS['channel-2'] = current_time
        if CHANNEL_COUNTERS['channel-2'] >= TOGGLE_THRESHOLD:
            commands['channel-2'] = 'SQUAT DOWN'
            CHANNEL_COUNTERS['channel-2'] = 0
    elif channel_data[squatUpDownChannel2] > DEADBAND_HIGH: # if the pulse width is greater than DEADBAND_HIGH...
        if current_time - CHANNEL_TIMESTAMPS['channel-2'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-2'] = 0
        CHANNEL_COUNTERS['channel-2'] += 1
        CHANNEL_TIMESTAMPS['channel-2'] = current_time
        if CHANNEL_COUNTERS['channel-2'] >= TOGGLE_THRESHOLD:
            commands['channel-2'] = 'SQUAT UP'
            CHANNEL_COUNTERS['channel-2'] = 0

    ##### rotation channel 3 #####

    if channel_data[rotateLeftRightChannel3] < DEADBAND_LOW: # if the pulse width is less than DEADBAND_LOW...
        if current_time - CHANNEL_TIMESTAMPS['channel-3'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-3'] = 0
        CHANNEL_COUNTERS['channel-3'] += 1
        CHANNEL_TIMESTAMPS['channel-3'] = current_time
        if CHANNEL_COUNTERS['channel-3'] >= JOYSTICK_THRESHOLD:
            commands['channel-3'] = 'ROTATE LEFT' # request left rotation
            CHANNEL_COUNTERS['channel-3'] = 0
    elif channel_data[rotateLeftRightChannel3] > DEADBAND_HIGH: # if the pulse width is greater than DEADBAND_HIGH...
        if current_time - CHANNEL_TIMESTAMPS['channel-3'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-3'] = 0
        CHANNEL_COUNTERS['channel-3'] += 1
        CHANNEL_TIMESTAMPS['channel-3'] = current_time
        if CHANNEL_COUNTERS['channel-3'] >= JOYSTICK_THRESHOLD:
            commands['channel-3'] = 'ROTATE RIGHT' # request right rotation
            CHANNEL_COUNTERS['channel-3'] = 0

    ##### look channel 4 #####

    if channel_data[lookUpDownChannel4] < DEADBAND_LOW: # if the pulse width is less than DEADBAND_LOW...
        if current_time - CHANNEL_TIMESTAMPS['channel-4'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-4'] = 0
        CHANNEL_COUNTERS['channel-4'] += 1
        CHANNEL_TIMESTAMPS['channel-4'] = current_time
        if CHANNEL_COUNTERS['channel-4'] >= JOYSTICK_THRESHOLD:
            commands['channel-4'] = 'LOOK DOWN' # request looking down
            CHANNEL_COUNTERS['channel-4'] = 0
    elif channel_data[lookUpDownChannel4] > DEADBAND_HIGH: # if the pulse width is greater than DEADBAND_HIGH...
        if current_time - CHANNEL_TIMESTAMPS['channel-4'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-4'] = 0
        CHANNEL_COUNTERS['channel-4'] += 1
        CHANNEL_TIMESTAMPS['channel-4'] = current_time
        if CHANNEL_COUNTERS['channel-4'] >= JOYSTICK_THRESHOLD:
            commands['channel-4'] = 'LOOK UP' # request looking up
            CHANNEL_COUNTERS['channel-4'] = 0

    ##### move channel 5 #####

    if channel_data[moveForwardBackwardChannel5] < DEADBAND_LOW: # if the pulse width is less than DEADBAND_LOW...
        if current_time - CHANNEL_TIMESTAMPS['channel-5'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-5'] = 0
        CHANNEL_COUNTERS['channel-5'] += 1
        CHANNEL_TIMESTAMPS['channel-5'] = current_time
        if CHANNEL_COUNTERS['channel-5'] >= JOYSTICK_THRESHOLD:
            commands['channel-5'] = 'MOVE FORWARD' # request movement forward
            CHANNEL_COUNTERS['channel-5'] = 0
    elif channel_data[moveForwardBackwardChannel5] > DEADBAND_HIGH: # if the pulse width is greater than DEADBAND_HIGH..
        if current_time - CHANNEL_TIMESTAMPS['channel-5'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-5'] = 0
        CHANNEL_COUNTERS['channel-5'] += 1
        CHANNEL_TIMESTAMPS['channel-5'] = current_time
        if CHANNEL_COUNTERS['channel-5'] >= JOYSTICK_THRESHOLD:
            commands['channel-5'] = 'MOVE BACKWARD' # request movement backward
            CHANNEL_COUNTERS['channel-5'] = 0

    ##### shift channel 6 #####

    if channel_data[shiftLeftRightChannel6] < DEADBAND_LOW: # if the pulse width is less than DEADBAND_LOW...
        if current_time - CHANNEL_TIMESTAMPS['channel-6'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-6'] = 0
        CHANNEL_COUNTERS['channel-6'] += 1
        CHANNEL_TIMESTAMPS['channel-6'] = current_time
        if CHANNEL_COUNTERS['channel-6'] >= JOYSTICK_THRESHOLD:
            commands['channel-6'] = 'SHIFT LEFT' # request shift left
            CHANNEL_COUNTERS['channel-6'] = 0
    elif channel_data[shiftLeftRightChannel6] > DEADBAND_HIGH: # if the pulse width is greater than DEADBAND_HIGH...
        if current_time - CHANNEL_TIMESTAMPS['channel-6'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-6'] = 0
        CHANNEL_COUNTERS['channel-6'] += 1
        CHANNEL_TIMESTAMPS['channel-6'] = current_time
        if CHANNEL_COUNTERS['channel-6'] >= JOYSTICK_THRESHOLD:
            commands['channel-6'] = 'SHIFT RIGHT' # request shift right
            CHANNEL_COUNTERS['channel-6'] = 0

    ##### extra channel 7 #####

    if channel_data[extraChannel7] < DEADBAND_LOW: # if the pulse width is less than DEADBAND_LOW...
        if current_time - CHANNEL_TIMESTAMPS['channel-7'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-7'] = 0
        CHANNEL_COUNTERS['channel-7'] += 1
        CHANNEL_TIMESTAMPS['channel-7'] = current_time
        if CHANNEL_COUNTERS['channel-7'] >= TOGGLE_THRESHOLD:
            commands['channel-7'] = '+'
            CHANNEL_COUNTERS['channel-7'] = 0
    elif channel_data[extraChannel7] > DEADBAND_HIGH: # if the pulse width is greater than DEADBAND_HIGH...
        if current_time - CHANNEL_TIMESTAMPS['channel-7'] > TIME_FRAME:
            CHANNEL_COUNTERS['channel-7'] = 0
        CHANNEL_COUNTERS['channel-7'] += 1
        CHANNEL_TIMESTAMPS['channel-7'] = current_time
        if CHANNEL_COUNTERS['channel-7'] >= TOGGLE_THRESHOLD:
            commands['channel-7'] = '-'
            CHANNEL_COUNTERS['channel-7'] = 0

    return commands # return command requests
