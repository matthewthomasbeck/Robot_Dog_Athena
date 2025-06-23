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

##### import libraries #####

import os  # import os for file operations
import socket  # import socket for Unix socket communication
import logging  # import logging for debugging


########## CREATE DEPENDENCIES ##########

##### define necessary paths #####

SOCKET_PATH = '/tmp/robot.sock'





##############################################################
############### INTERNET CONNECTIVITY FUNCTION ###############
##############################################################


########## SET UP SOCKET FOR USE ##########

def setup_unix_socket(): # function to set up a Unix socket for communication

    if os.path.exists(SOCKET_PATH): # if socket already exists...
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) # create a Unix socket
    server.bind(SOCKET_PATH) # bind socket to specified path
    server.listen(1) # listen for incoming connections
    os.chmod(SOCKET_PATH, 0o666) # set permissions to allow read/write for all users

    return server


########## PROMPT USER ON SSH ##########

def detect_ssh_and_prompt_mode(): # function to detect if SSH is available and prompt user for mode

    ##### check for SSH sessions #####

    logging.debug("(internet.py): Checking for SSH sessions...\n") # log checking for SSH sessions
    who_output = os.popen("who").read() # get output of 'who' command to check for SSH sessions
    tty_path = None # initialize tty_path to None

    for line in who_output.splitlines(): # iterate through each line of output

        if "pts/" in line: # if line contains a pseudo-terminal session...
            tty_path = f"/dev/{line.split()[1]}" # extract tty path
            break

    ##### prompt user for SSH mode #####

    if tty_path: # if a tty path was found...

        try: # attempt to prompt user for SSH mode

            with open(tty_path, 'w') as tty_out, open(tty_path, 'r') as tty_in: # open tty for writing and reading

                tty_out.write("(internet.py): Control mode or Tune mode? [c/t]: ")
                tty_out.flush() # flush output to ensure prompt is displayed
                choice = tty_in.read(1).strip().lower() # read user input
                mode = 'ssh-control' if choice == 'c' else 'ssh-tune'
                tty_out.write(f"\nMode set to {mode}. Now run: python3 ~/Projects/Robot_Dog/ssh_control_client.py\n")
                tty_out.flush() # flush output again

                return mode

        except Exception as e:
            logging.error(f"(internet.py): Failed to prompt for SSH mode: {e}\n")

    return 'radio' # default to radio mode if no SSH session found
