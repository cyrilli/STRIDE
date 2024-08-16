import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

from datetime import datetime
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from env import EDP
from program_agent import EDPAgent
from utils import Logger
from envs.env_helper import get_env_param


def play_through(game, agents, logger, exmps_file=None):
    # assert game.check_agents(agents) # check if all agents required by the game are specified
    logger.write(game.description_of_problem_class)
    # describe the decision making problem instace to each agent
    instance_description = game.get_description()
    logger.write(instance_description)
    with open(exmps_file, "w") as f:
        f.write("==== ASSISTANT ====\n")
        f.write(instance_description+"\n")

    action = agents.move()
    env_action = game.step(action)
    logger.write("\nThe edit distance between A = {} to B = {} is {}.\n".format(game.A, game.B, env_action))
    logger.write("The game has ended!", color="red")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="edit_distance_problem")
    args = parser.parse_args()

    logger_output_path = "envs/edit_distance_problem/outputs/"
    os.makedirs(logger_output_path, exist_ok=True) 
    now = datetime.now()
    time_string = now.strftime('%Y%m%d%H%M%S')
    logger = Logger(logger_output_path + args.game + "-" + time_string + ".html", verbose=True, writeToFile=False)

    exmps_output_path = "envs/edit_distance_problem/prompts/"
    os.makedirs(exmps_output_path, exist_ok=True) 
    exmps_file = exmps_output_path + "edit_distance_problem_exmps_"+time_string+".txt"

    ### initialize game and agents ###
    env_param = get_env_param(env_name="edit_distance_problem")
    game = EDP(env_param=env_param)
    working_memory = {"A":game.A, "B":game.B, "m":game.m, "n":game.n, "dp":[]}
    agent = EDPAgent(working_memory=working_memory, exmps_file=exmps_file)
    agents = {"agent":agent}

    ### start play ###
    play_through(game=game, agents=agent, logger=logger, exmps_file=exmps_file)