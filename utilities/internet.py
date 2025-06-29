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

import os  # import os for file operations
import socket  # import socket for Unix socket communication
import logging  # import logging for debugging
import threading  # import threading for concurrent operations
from queue import Queue  # import Queue for command queue management

##### import config #####

from utilities.config import LOOP_RATE_HZ, INTERNET_CONFIG  # import logging configuration from config module


########## CREATE DEPENDENCIES ##########

##### create global variables #####

sock = None  # global socket variable for EC2 connection, initialized to None
_send_lock = threading.Lock()  # lock for sending data to EC2, initialized to a new threading lock





##############################################################
############### INTERNET CONNECTIVITY FUNCTION ###############
##############################################################

########## CONNECT TO EC2 ##########

def initialize_ec2_socket(): # function to connect to EC2 instance via socket

    ##### initialize global socket #####

    logging.debug("(internet.py): Initializing EC2 socket...\n")

    global sock

    if not sock: # if sock is not already initialized...
        sock = socket.socket() # create a new socket object

        try:
            sock.connect((INTERNET_CONFIG['EC2_PUBLIC_IP'], INTERNET_CONFIG['EC2_PORT']))
            logging.info("Connected to EC2 instance.\n")
            return sock

        except Exception as e:
            logging.error(f"(internet.py): Failed to connect to EC2: {e}\n")
            sock = None
            return None


########## STREAM FRAME DATA TO EC2 ##########

def stream_to_ec2(sock, frame_data): # function to send frame data to EC2 instance

    ##### send frame data to EC2 #####

    logging.debug("(internet.py): Streaming frame data to EC2...\n")

    if frame_data is not None:
        try:
            with _send_lock:
                sock.sendall(len(frame_data).to_bytes(4, 'big')) # send frame length first
                sock.sendall(frame_data) # send frame data to EC2 instance
                logging.info("Frame data sent to EC2 successfully.\n")

        except Exception as e:
            logging.error(f"(internet.py): Error sending data to EC2: {e}\n")

    else:
        logging.warning("(internet.py): No frame data to send.\n")


########## INITIALIZE COMMAND QUEUE ##########

def initialize_command_queue(SOCK): # function to create a command queue for receiving commands from EC2 instance

    logging.debug("(internet.py): Initializing command queue...\n") # log initialization of command queue

    if SOCK is None:
        logging.error("(internet.py): No EC2 socket—command queue not started.")
        return None

    try:
        command_queue = Queue() # create a new command queue
        threading.Thread(target=listen_for_commands, args=(SOCK, command_queue), daemon=True).start()
        logging.info("Command queue initialized successfully.\n")
        return command_queue # return the command queue for further processing

    except Exception as e:
        logging.error(f"(internet.py): Failed to initialize command queue: {e}\n")
        return None


########## RECEIVE COMMANDS FROM EC2 ##########

def listen_for_commands(sock, command_queue): # function to listen for commands from EC2 instance
    while True:
        try:
            length_bytes = sock.recv(4)
            if not length_bytes:
                continue
            length = int.from_bytes(length_bytes, 'big')
            command = sock.recv(length).decode()
            command_queue.put(command)
        except Exception as e:
            logging.error(f"(internet.py): Error receiving command from EC2: {e}")


########## INITIALIZE SOCKET ##########

def initialize_socket(): # function to set up a Unix socket for SSH communication

    if os.path.exists(INTERNET_CONFIG['SSH_SOCKET_PATH']): # if socket already exists...
        os.remove(INTERNET_CONFIG['SSH_SOCKET_PATH'])

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) # create a Unix socket
    server.bind(INTERNET_CONFIG['SSH_SOCKET_PATH']) # bind socket to specified path
    server.listen(1) # listen for incoming connections
    os.chmod(INTERNET_CONFIG['SSH_SOCKET_PATH'], 0o666) # set permissions to allow read/write for all users

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
