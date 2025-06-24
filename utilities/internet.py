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

from utilities.config import INTERNET_CONFIG  # import logging configuration from config module


########## CREATE DEPENDENCIES ##########

##### create flask objects #####
APP = Flask(__name__)  # create flask app instance for video streaming
JPEG_FRAME_QUEUE = deque(maxlen=10)  # store a minimum of 10 JPEG frames in queue for video streaming





##############################################################
############### INTERNET CONNECTIVITY FUNCTION ###############
##############################################################


########## INITIALIZE FLASK ##########

def initialize_flask():
    kill_lingering_flask()
    JPEG_FRAME_QUEUE.clear()
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    logging.debug("(internet.py): Flask initialized and started.")


########## KILL LINGERING FLASK SERVER ##########

def kill_lingering_flask_OLD():

    try:

        # aggressively shut down any lingering flask processes
        result = subprocess.run(["pgrep", "-f", "flask"], stdout=subprocess.PIPE, text=True)
        for pid in result.stdout.splitlines():
            subprocess.run(["kill", "-9", pid])
        logging.debug("(internet.py): Killed lingering Flask processes.")
    except Exception as e:
        logging.warning(f"(internet.py): Could not kill Flask processes: {e}")


def kill_lingering_flask():
    try:
        result = subprocess.run(
            ["pgrep", "-f", "control_logic.py"], stdout=subprocess.PIPE, text=True
        )
        for pid in result.stdout.splitlines():
            if str(os.getpid()) != pid:  # don't kill yourself
                kill_result = subprocess.run(["kill", "-9", pid])
                if kill_result.returncode != 0:
                    logging.warning(f"(internet.py): Failed to kill PID {pid}")
        logging.debug("(internet.py): Cleaned up old Flask processes.")
    except Exception as e:
        logging.warning(f"(internet.py): Could not clean up Flask processes: {e}")


########## SET UP VIDEO FEED ##########
@APP.route("/video_feed")
def video_feed():
    def stream():
        while True:
            if JPEG_FRAME_QUEUE:
                frame = JPEG_FRAME_QUEUE[-1]
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                time.sleep(0.01)
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


########## START FLASK ##########

def start_flask():
    APP.run(host="0.0.0.0", port=5000)


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
