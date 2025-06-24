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
import subprocess  # import subprocess for running shell commands
import threading  # import threading for running Flask in a separate thread
import time  # import time for flask waiting
from flask import Flask, Response  # import for web server video streaming
from collections import deque  # import deque to forward MJPEG data to flask

##### import config #####

from utilities.config import LOOP_RATE_HZ, INTERNET_CONFIG  # import logging configuration from config module


########## CREATE DEPENDENCIES ##########

##### create flask objects #####
APP = Flask(__name__)  # create flask app instance for video streaming
JPEG_FRAME_QUEUE = deque(maxlen=5)  # store a minimum of 10 JPEG frames in queue for video streaming





##############################################################
############### INTERNET CONNECTIVITY FUNCTION ###############
##############################################################


########## INITIALIZE FLASK ##########

def initialize_flask(): # function to initialize flask server for video streaming

    ##### destroy any lingering flask processes #####

    logging.debug("(internet.py): Initializing Flask server...\n")  # log initializing Flask server

    try:
        _kill_lingering_flask() # kill old flask servers
        JPEG_FRAME_QUEUE.clear() # clear the JPEG frame queue
        logging.info("(internet.py): Cleaned up old Flask processes and cleared JPEG frame queue.\n")

    except Exception as e:
        logging.warning(f"(internet.py): Failed to clean up old flask processes or queue: {e}\n")

    ##### start flask server #####

    try:
        flask_thread = threading.Thread(target=_start_flask, daemon=True) # create a thread to run flask server
        flask_thread.start() # start the flask server
        logging.info("(internet.py): Flask initialized and started.\n")
        logging.info(f"(internet.py): JPEG_FRAME_QUEUE id in internet.py: {id(JPEG_FRAME_QUEUE)}\n")

    except Exception as e:
        logging.error(f"(internet.py): Failed to start Flask server: {e}\n")
        return False


########## KILL LINGERING FLASK PROCESSES ##########

def _kill_lingering_flask(): # function to kill any lingering flask processes

    ##### kill old flask processes #####

    logging.debug("(internet.py): Killing any lingering Flask processes...\n")  # log killing old Flask processes

    try:
        # use pgrep to find all processes running control_logic.py
        result = subprocess.run(["pgrep", "-f", "control_logic.py"], stdout=subprocess.PIPE, text=True)

        for pid in result.stdout.splitlines(): # iterate through each PID found
            if str(os.getpid()) != pid: # ensure not to kill itself
                kill_result = subprocess.run(["kill", "-9", pid]) # kill the process with SIGKILL
                if kill_result.returncode != 0:  # check if kill command was successful
                    logging.warning(f"(internet.py): Failed to kill PID {pid}\n")

        logging.info("(internet.py): Cleaned up old Flask processes.\n")

    except Exception as e:
        logging.warning(f"(internet.py): Could not clean up Flask processes: {e}\n")


########## SET UP VIDEO FEED ##########

@APP.route("/video_feed")
def video_feed(): # route to serve video feed from the camera

    def stream(): # generator function to stream video frames

        while True:

            if JPEG_FRAME_QUEUE: # if there are frames in the queue...
                frame = JPEG_FRAME_QUEUE[-1] # get the last frame in the queue
                yield (
                        b"--frame\r\n"  # MJPEG boundary delimiter
                        b"Content-Type: image/jpeg\r\n\r\n"  # MIME header for the frame
                        + frame  # actual JPEG binary data
                        + b"\r\n"  # end of the frame block
                )

            else: # if there are no frames in the queue...
                time.sleep(0.01) # wait for some more

            time.sleep(1/LOOP_RATE_HZ) # throttle stream to match camera and robot actions per second

    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame') # return video stream as a response


########## START FLASK ##########

def _start_flask(): # function to start flask server in a separate thread

    ##### start flask server #####

    logging.debug("(internet.py): Starting Flask server...\n")  # log starting Flask server

    try:
        APP.run(host="0.0.0.0", port=5000) # run flask app on all interfaces at port 5000

    except Exception as e:
        logging.error(f"(internet.py): Failed to start Flask server: {e}\n")

    logging.info("(internet.py): Flask server started.\n")  # log starting Flask server


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
