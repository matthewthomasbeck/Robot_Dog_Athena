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

##### import config #####

import utilities.config as config

##### import necessary libraries #####

import glob
import re
import os
import logging
import torch
import training.episodes as episodes





#############################################
############### MODEL MANAGER ###############
#############################################

########## MODEL MANAGEMENT FUNCTIONS ##########

##### find latest model #####

def find_latest_model():
    """Find the latest saved model file"""
    
    # Look for model files in the models directory (both old and new formats)
    old_pattern = os.path.join(config.MODELS_DIRECTORY, "ppo_steps_*_episode_*_reward_*.pth")
    new_pattern = os.path.join(config.MODELS_DIRECTORY, "*_*.pth")  # New format: steps_avg.pth
    
    model_files = glob.glob(old_pattern) + glob.glob(new_pattern)
    
    if not model_files:
        return None
    
    # Extract step numbers and find the latest
    latest_model = None
    latest_steps = 0
    
    for model_file in model_files:
        filename = os.path.basename(model_file)
        
        # Try old format first: "ppo_steps_700000_episode_47474_avg_-109.76.pth"
        match = re.search(r'ppo_steps_(\d+)_episode_', filename)
        if match:
            steps = int(match.group(1))
            if steps > latest_steps:
                latest_steps = steps
                latest_model = model_file
        else:
            # Try new format: "1300000_-5.22.pth"
            match = re.search(r'^(\d+)_', filename)
            if match:
                steps = int(match.group(1))
                if steps > latest_steps:
                    latest_steps = steps
                    latest_model = model_file
    
    return latest_model

##### save trained model #####

def save_model(filepath, ppo_policy, agent_data, total_steps):
    """Save the current PPO model"""
    if ppo_policy:

        logging.info(f"💾 Saving multi-agent PPO model to: {filepath}")
        logging.info(f"   📊 Total steps: {total_steps}")
        logging.info(f"   📊 Number of agents: {len(agent_data)}")
        logging.info(f"   📊 Current average score: {episodes.AVERAGE_SCORE:.4f}\n")

        # Create the checkpoint data
        checkpoint = {
            'actor_critic_state_dict': ppo_policy.actor_critic.state_dict(),
            'optimizer_state_dict': ppo_policy.optimizer.state_dict(),
            'total_steps': total_steps,
            'agent_data': agent_data,
            'average_score': episodes.AVERAGE_SCORE,
        }

        # Verify checkpoint data before saving
        logging.info(f"   🔍 Checkpoint contains {len(checkpoint)} keys:")
        for key, value in checkpoint.items():
            if 'state_dict' in key:
                if isinstance(value, dict):
                    logging.info(f"      ✅ {key}: {len(value)} layers\n")
                else:
                    logging.warning(f"      ❌ {key}: Invalid type {type(value)}\n")
            else:
                logging.info(f"      📊 {key}: {value}\n")

        # Save the model
        torch.save(checkpoint, filepath)

        # Verify the file was created and has content
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            logging.info(f"   ✅ Model saved successfully! File size: {file_size:.1f} MB\n")
        else:
            logging.error(f"   ❌ Failed to save model - file not created.\n")

        logging.info(f"Multi-agent model saved to: {filepath}")
        logging.info(f"   📊 Current average score: {episodes.AVERAGE_SCORE:.3f}\n")
    else:
        logging.error(f"❌ Cannot save model - PPO policy not initialized.\n")


##### load the model #####

def load_model(filepath, ppo_policy):
    """Load a PPO model from file"""
    if ppo_policy and os.path.exists(filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        ppo_policy.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        ppo_policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore training state
        total_steps = checkpoint.get('total_steps', 0)
        agent_data = checkpoint.get('agent_data', {})
        episodes.AVERAGE_SCORE = checkpoint.get('average_score', 0.0)

        # CRITICAL: If agent_data is empty or doesn't match expected number of robots, reinitialize it
        num_robots = config.MULTI_ROBOT_CONFIG['num_robots']
        if not agent_data or len(agent_data) != num_robots:
            logging.warning(f"Agent data from checkpoint is invalid (got {len(agent_data)} agents, expected {num_robots}), reinitializing...")
            # Import here to avoid circular imports
            from training.agents import initialize_agent_data
            agent_data = initialize_agent_data()

        logging.info(f"Multi-agent model loaded from: {filepath}")
        logging.info(f"  - Total steps: {total_steps}")
        logging.info(f"  - Number of agents: {len(agent_data)}")
        logging.info(f"  - Average score: {episodes.AVERAGE_SCORE:.4f}\n")

        return True, total_steps, agent_data

    return False, 0, {}
