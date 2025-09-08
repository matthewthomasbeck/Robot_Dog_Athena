##################################################################################
# Copyright (c) 2025 Matthew Thomas Beck                                         #
#                                                                                #
# Licensed under the Creative Commons Attribution-NonCommercial 4.0              #
# International (CC BY-NC 4.0). Personal and educational use is permitted.       #
# Commercial use by companies or for-profit entities is prohibited.              #
##################################################################################





############################################################
############### IMPORT / CREATE DEPENDENCIES ###############
############################################################


########## IMPORT DEPENDENCIES ##########

##### import necessary libraries #####

import os # import os to check if log file exists and to rename it
import sys # import sys to output logs to console
import logging # import logging for logging messages

##### import config #####

from utilities.config import LOG_CONFIG # import config to get logging settings





##################################################
############### INITIALIZE LOGGING ###############
##################################################


########## SET UP LOGGING ##########

# function to set up logging for entire system
def initialize_logging(
        log_path=LOG_CONFIG['LOG_PATH'],
        log_level=LOG_CONFIG['LOG_LEVEL']
):

    ##### set old log file to backup #####

    if os.path.exists(log_path):
        os.rename(log_path, f"{log_path}.bak") # rename existing log file to a backup

    ##### set up new log #####

    try: # try to set up logging

        logger = logging.getLogger()  # get root logger
        logger.setLevel(log_level)  # set logging level to user-specified level
        for handler in logger.handlers[:]:  # remove all existing handlers to avoid duplicates
            logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_path, mode='w')  # create a file handler to write logs to specified file
        file_handler.setLevel(log_level)  # set logging level for file handler to user-specified level

        # set formatter for file handler to include timestamps and log levels
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)  # add the file handler to logger
        console_handler = logging.StreamHandler(sys.stdout)  # create a console handler to output logs to console

        # set formatter for console handler to only include messages without timestamps
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)  # add console handler to logger
        logger.info("(log.py): Logging setup complete.\n")

        return logger

    except Exception as e:
        print(f"Failed to set up logging: {e}\n", file=sys.stderr)
        return