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

import socket
import sys
import termios
import tty
import os
import signal


########## CREATE DEPENDENCIES ##########

##### define necessary paths #####

SOCKET_PATH = '/tmp/robot.sock'  # path for Unix socket used for communication with robot
old_settings = None # variable to store old terminal settings for restoration after use





##############################################################
############### INTERNET CONNECTIVITY FUNCTION ###############
##############################################################


def restore_terminal():
    global old_settings
    if old_settings:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
        old_settings = None

def read_key():
    return sys.stdin.read(1)

def run_client():
    global old_settings

    if not os.path.exists(SOCKET_PATH):
        print("ERROR (ssh_control_client.py): Socket not found; is the robot running?\n")
        return

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(SOCKET_PATH)
        print("Connected to robot. Use W/A/S/D or arrow keys. Ctrl+C to quit.\n")

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)

        try:
            while True:
                key = read_key()
                if key == '\x03':  # raw Ctrl+C
                    print("Ctrl+C detected — exiting.\n")
                    break
                if key == '\x1b':
                    key += sys.stdin.read(2)  # arrow keys
                client.sendall(key.encode())
        finally:
            restore_terminal()
            print("Terminal restored.\n")


########## RUN SSH CONTROL CLIENT ##########

run_client() # run client on start