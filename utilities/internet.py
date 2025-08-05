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

SOCK = None  # global socket variable for backend connection, initialized to None
_send_lock = threading.Lock()  # lock for sending data to backend, initialized to a new threading lock





##############################################################
############### INTERNET CONNECTIVITY FUNCTION ###############
##############################################################

########## CONNECT TO BACKEND ##########

def initialize_backend_socket(): # function to connect to backend via socket

    ##### initialize global socket #####

    global SOCK
    logging.debug("(internet.py): Initializing backend socket...\n")
    if not SOCK: # if sock is not already initialized...
        SOCK = socket.socket() # create a new socket object

        try:
            SOCK.connect((INTERNET_CONFIG['BACKEND_PUBLIC_IP'], INTERNET_CONFIG['BACKEND_PORT']))
            logging.info("(internet.py): Connected to website backend.\n")
            return SOCK

        except Exception as e:
            logging.error(f"(internet.py): Failed to connect to website backend: {e}\n")
            SOCK = None
            return None


########## STREAM FRAME DATA TO BACKEND ##########

def stream_to_backend(socket_param, frame_data): # function to send frame data to backend

    ##### send frame data to backend #####

    global SOCK
    if frame_data is not None and socket_param is not None:
        #logging.debug("(internet.py): Streaming frame data to website backend...\n")
        try:
            with _send_lock:
                # Send frame length first (4 bytes)
                frame_length = len(frame_data)
                socket_param.sendall(frame_length.to_bytes(4, 'big'))
                # Send frame data
                socket_param.sendall(frame_data)
                #logging.debug(
                    #f"(internet.py); Frame data sent to website backend successfully of size: {frame_length} bytes\n"
                #)

        except Exception as e:
            logging.error(f"(internet.py): Error sending data to website backend: {e}\n")
            # Try to reconnect if connection is lost
            try:
                socket_param.close()
                SOCK = None
                SOCK = initialize_backend_socket()
            except Exception as reconnect_error:
                logging.error(f"(internet.py): Failed to reconnect: {reconnect_error}\n")

    else:
        if frame_data is None:
            logging.debug("(internet.py): No frame data to send.\n")
            pass
        if socket_param is None:
            logging.warning("(internet.py): No socket connection to EC2.\n")


########## INITIALIZE COMMAND QUEUE ##########

def initialize_command_queue(local_sock): # function to create a command queue for receiving commands from backend

    logging.debug("(internet.py): Initializing command queue...\n") # log initialization of command queue

    if local_sock is None:
        logging.error("(internet.py): No website backend socket—command queue not started.\n")
        return None

    try:
        command_queue = Queue() # create a new command queue
        threading.Thread(target=listen_for_commands, args=(local_sock, command_queue), daemon=True).start()
        logging.info("(internet.py): Command queue initialized successfully.\n")
        return command_queue # return the command queue for further processing

    except Exception as e:
        logging.error(f"(internet.py): Failed to initialize command queue: {e}\n")
        return None


########## RECEIVE COMMANDS FROM BACKEND ##########

def listen_for_commands(local_sock, command_queue):
    logging.debug("(internet.py): Listening for commands from website backend...\n")
    while True:
        try:
            length_bytes = local_sock.recv(4)
            logging.debug(f"(internet.py): Received length_bytes: {length_bytes}\n")
            if not length_bytes:
                logging.warning("(internet.py): Socket closed or no data received for length. Exiting thread.\n")
                break
            length = int.from_bytes(length_bytes, 'big')
            command_bytes = b''
            while len(command_bytes) < length:
                chunk = local_sock.recv(length - len(command_bytes))
                if not chunk:
                    logging.warning("(internet.py): Socket closed while reading command. Exiting thread.\n")
                    break
                command_bytes += chunk
            if len(command_bytes) < length:
                break
            command = command_bytes.decode()
            command_queue.put(command)
            logging.debug(f"(internet.py): Received command: {command}\n")
        except Exception as e:
            logging.error(f"(internet.py): Error receiving command from website backend: {e}\n")
            break
        finally:
            logging.warning("(internet.py): Encountering thread issues (thread exiting)!\n")
            try:
                # TODO this is a really shitty way to solve this problem, I need to see if the thread issue is caused by
                # TODO some kind of camera overflow, an unstable internet connection, or something else
                pass
                #os.system("sudo systemctl restart robot_dog.service")
            except Exception as e:
                pass
                #logging.error(f"(internet.py): Failed to restart robot_dog service: {e}\n")
            pass
