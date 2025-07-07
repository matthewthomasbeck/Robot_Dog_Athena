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

sock = None  # global socket variable for backend connection, initialized to None
_send_lock = threading.Lock()  # lock for sending data to backend, initialized to a new threading lock





##############################################################
############### INTERNET CONNECTIVITY FUNCTION ###############
##############################################################

########## CONNECT TO BACKEND ##########

def initialize_backend_socket(): # function to connect to backend via socket

    ##### initialize global socket #####

    logging.debug("(internet.py): Initializing backend socket...\n")

    global sock

    if not sock: # if sock is not already initialized...
        sock = socket.socket() # create a new socket object

        try:
            sock.connect((INTERNET_CONFIG['BACKEND_PUBLIC_IP'], INTERNET_CONFIG['BACKEND_PORT']))
            logging.info("Connected to website backend.\n")
            return sock

        except Exception as e:
            logging.error(f"(internet.py): Failed to connect to website backend: {e}\n")
            sock = None
            return None


########## STREAM FRAME DATA TO BACKEND ##########

def stream_to_backend(socket_param, frame_data): # function to send frame data to backend

    ##### send frame data to backend #####

    logging.debug("(internet.py): Streaming frame data to website backend...\n")

    if frame_data is not None and socket_param is not None:
        try:
            with _send_lock:
                # Send frame length first (4 bytes)
                frame_length = len(frame_data)
                socket_param.sendall(frame_length.to_bytes(4, 'big'))
                # Send frame data
                socket_param.sendall(frame_data)
                logging.info(f"Frame data sent to website backend successfully. Frame size: {frame_length} bytes\n")

        except Exception as e:
            logging.error(f"(internet.py): Error sending data to website backend: {e}\n")
            # Try to reconnect if connection is lost
            try:
                socket_param.close()
                global sock
                sock = None
                sock = initialize_backend_socket()
            except Exception as reconnect_error:
                logging.error(f"(internet.py): Failed to reconnect: {reconnect_error}\n")

    else:
        if frame_data is None:
            logging.warning("(internet.py): No frame data to send.\n")
        if socket_param is None:
            logging.warning("(internet.py): No socket connection to EC2.\n")


########## INITIALIZE COMMAND QUEUE ##########

def initialize_command_queue(SOCK): # function to create a command queue for receiving commands from backend

    logging.debug("(internet.py): Initializing command queue...\n") # log initialization of command queue

    if SOCK is None:
        logging.error("(internet.py): No website backend socket—command queue not started.")
        return None

    try:
        command_queue = Queue() # create a new command queue
        threading.Thread(target=listen_for_commands, args=(SOCK, command_queue), daemon=True).start()
        logging.info("Command queue initialized successfully.\n")
        return command_queue # return the command queue for further processing

    except Exception as e:
        logging.error(f"(internet.py): Failed to initialize command queue: {e}\n")
        return None


########## RECEIVE COMMANDS FROM BACKEND ##########

def listen_for_commands(sock, command_queue):

    logging.info("(internet.py): Listening for commands from website backend...\n") # log listening for commands

    while True:
        try:
            length_bytes = sock.recv(4) # receive first 4 bytes for length of command
            if not length_bytes:
                continue
            length = int.from_bytes(length_bytes, 'big') # convert bytes to integer length
            command = sock.recv(length).decode() # receive command data based on length
            command_queue.put(command) # put command into the queue for processing
            logging.info(f"(internet.py): Received command: {command}")

        except Exception as e:
            logging.error(f"(internet.py): Error receiving command from website backend: {e}\n")

            if isinstance(e, OSError) and e.errno == 9: # if socket is bad...

                logging.error("(internet.py): Socket is bad, exiting command listener thread.\n")
                break # exit thread
            break

        finally:
            logging.error("(internet.py): listen_for_commands thread exiting!")


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
