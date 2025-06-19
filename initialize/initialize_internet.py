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

import os
import socket
import logging  # import logging for debugging


########## CREATE DEPENDENCIES ##########

##### define necessary paths #####

SOCKET_PATH = '/tmp/robot.sock'


##############################################################
############### INTERNET CONNECTIVITY FUNCTION ###############
##############################################################


########## SET UP SOCKET FOR USE ##########

def setup_unix_socket():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    os.chmod(SOCKET_PATH, 0o666)
    return server


########## PROMPT USER ON SSH ##########

def detect_ssh_and_prompt_mode():
    who_output = os.popen("who").read()
    tty_path = None
    for line in who_output.splitlines():
        if "pts/" in line:
            tty_path = f"/dev/{line.split()[1]}"
            break

    if tty_path:
        try:
            with open(tty_path, 'w') as tty_out, open(tty_path, 'r') as tty_in:
                tty_out.write("Control mode or Tune mode? [c/t]: ")
                tty_out.flush()
                choice = tty_in.read(1).strip().lower()
                mode = 'ssh-control' if choice == 'c' else 'ssh-tune'
                tty_out.write(f"\nMode set to {mode}. Now run: python3 ~/Projects/Robot_Dog/ssh_control_client.py\n")
                tty_out.flush()
                return mode
        except Exception as e:
            logging.error(f"ERROR (initialize_internet.py): Failed to prompt for SSH mode: {e}\n")
    return 'radio'  # fallback
