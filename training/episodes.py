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
import training.rewards as rewards

########## CREATE DEPENDENCIES ##########

##### initialize global variables #####

AVERAGE_SCORE = 0.0 # called in training.py
EPISODE_SCORES = []





##################################################
############### EPISODE MANAGEMENT ###############
##################################################


########## EPISODE FUNCTIONS ##########

##### start episode #####

def start_episode(agent_data, robot_id):
    """Start a new training episode for a specific robot"""

    agent = agent_data[robot_id]
    agent['episode_counter'] += 1
    setattr(rewards, f'EPISODE_STEP_{robot_id}', 0)
    agent['episode_reward'] = 0.0
    agent['episode_states'] = []
    agent['episode_actions'] = []
    agent['episode_rewards'] = []
    agent['episode_values'] = []
    agent['episode_log_probs'] = []
    agent['episode_dones'] = []

    logging.debug(f"🚀 Robot {robot_id} starting episode {agent['episode_counter']}\n")
    # Show initial orientation at episode start
    track_orientation(robot_id)
    print(f"Robot {robot_id} TRACKING ORIENTATION\n")

##### end episode #####

def end_episode(agent_data, robot_id):
    """End current episode for a specific robot and save progress"""

    global EPISODE_SCORES, AVERAGE_SCORE

    agent = agent_data[robot_id]
    episode_step = getattr(rewards, f'EPISODE_STEP_{robot_id}', 0)
    
    logging.debug(f"🎯 Robot {robot_id} Episode {agent['episode_counter']} ended:")
    logging.debug(f"   📊 Steps: {episode_step}")
    logging.debug(f"   📊 Final Reward: {agent['episode_reward']:.3f}\n")

    # Track episode scores for average calculation
    EPISODE_SCORES.append(agent['episode_reward'])
    if len(EPISODE_SCORES) > 100:  # Keep only last 100 episodes for recent average
        EPISODE_SCORES.pop(0)
    
    # Calculate running average
    AVERAGE_SCORE = sum(EPISODE_SCORES) / len(EPISODE_SCORES)
    logging.info(f"   📊 Average Score (last {len(EPISODE_SCORES)} episodes): {AVERAGE_SCORE:.3f}\n")

    # Update agent's average score
    agent['episode_scores'].append(agent['episode_reward'])
    if len(agent['episode_scores']) > 20:  # Keep only last 20 episodes per agent
        agent['episode_scores'].pop(0)
    
    agent['average_score'] = sum(agent['episode_scores']) / len(agent['episode_scores'])
    logging.info(f"   📊 Robot {robot_id} Average Score (last {len(agent['episode_scores'])} episodes): {agent['average_score']:.3f}\n")


##### check/reset episode #####

def check_and_reset_episode_if_needed(agent_data, robot_id):
    """
    Check if episode should be reset for a specific robot and trigger reset if needed.
    This function should be called from the main thread to integrate episode management.

    Returns:
        bool: True if episode was reset, False otherwise
    """

    import utilities.config as config

    agent = agent_data[robot_id]
    episode_step = getattr(rewards, f'EPISODE_STEP_{robot_id}', 0)

    # Check if episode has reached max steps
    if episode_step >= config.TRAINING_CONFIG['max_steps_per_episode']:
        logging.debug(f"🎯 Robot {robot_id} Episode {agent['episode_counter']} completed after {episode_step} steps.\n")
        # Save model before resetting
        end_episode(agent_data, robot_id)
        
        # Monitor learning progress before resetting
        monitor_learning_progress(robot_id)
        
        reset_episode(agent_data, robot_id)
        return True

    return False

##### reset episode #####

def reset_episode(agent_data, robot_id):
    """
    Reset the current episode for a specific robot by resetting Isaac Sim world and moving robot to neutral position.
    This is the critical function that was working in working_robot_reset.py
    """

    try:
        agent = agent_data[robot_id]
        episode_step = getattr(rewards, f'EPISODE_STEP_{robot_id}', 0)
        
        logging.info(f"(training.py): Robot {robot_id} Episode {agent['episode_counter']} ending - Episode complete, resetting Isaac Sim world.\n")

        # CRITICAL: Save the model before resetting (if episode had any progress)
        if episode_step > 0:
            end_episode(agent_data, robot_id)

        # CRITICAL: Small delay to ensure all experiences are processed
        import time
        time.sleep(0.2)  # 200ms delay for experience processing
        
        # CRITICAL: Reset Isaac Sim world (position, velocity, physics state) - TEMPORARY FOR TESTING
        if config.USE_SIMULATION and config.USE_ISAAC_SIM:
            # Reset the world - this resets robot position, velocities, and physics state
            config.ISAAC_WORLD.reset()

            # Give Isaac Sim a few steps to stabilize after world reset
            for _ in range(5):
                config.ISAAC_WORLD.step(render=True)

        # CRITICAL: Move robot to neutral position (joint angles)
        neutral_position_isaac()  # Move to neutral position in Isaac Sim

        # Give Isaac Sim more steps to stabilize after neutral position
        if config.USE_SIMULATION and config.USE_ISAAC_SIM:
            for _ in range(5):
                config.ISAAC_WORLD.step(render=True)

        # Reset Python tracking variables for this robot
        setattr(rewards, f'EPISODE_STEP_{robot_id}', 0)
        agent['episode_reward'] = 0.0
        agent['episode_states'] = []
        agent['episode_actions'] = []

        # Increment episode counter
        agent['episode_counter'] += 1

        logging.info(f"(training.py): Robot {robot_id} Episode {agent['episode_counter']} reset complete - World and robot state reset.\n")

        # Log learning progress every 10 episodes
        if agent['episode_counter'] % 10 == 0:
            logging.debug(f"🎯 Robot {robot_id} Training Progress: {agent['episode_counter']} episodes completed.\n")
            if agent['episode_states']: # Check if episode_states is not empty
                logging.debug(f"   📊 Episode data size: {len(agent['episode_states'])}\n")
            if ppo_policy is not None:
                logging.debug(f"   🧠 PPO policy ready for training.\n")

    except Exception as e:
        logging.error(f"(training.py): Failed to reset robot {robot_id} episode: {e}\n")
        # Don't crash - try to continue with next episode
        agent['episode_counter'] += 1
        setattr(rewards, f'EPISODE_STEP_{robot_id}', 0)
        agent['episode_reward'] = 0.0
