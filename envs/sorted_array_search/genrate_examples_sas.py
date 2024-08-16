import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

from datetime import datetime
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from env import SAS
from program_agent import SASAgent
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

    # time_step = 0
    # while not game.is_done:
    action = agents.move()
        # logger.write("agent takes action {}".format(action), color = "green")
    env_action = game.step(action)
        # time_step += 1
    logger.write("\nThe position of the target value T = {} is {}.\n".format(game.T,env_action))
    logger.write("The game has ended!", color="red")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="sorted_array_search")
    args = parser.parse_args()

    logger_output_path = "envs/sorted_array_search/outputs/"
    os.makedirs(logger_output_path, exist_ok=True) 
    now = datetime.now()
    time_string = now.strftime('%Y%m%d%H%M%S')
    logger = Logger(logger_output_path + args.game + "-" + time_string + ".html", verbose=True, writeToFile=False)

    exmps_output_path = "envs/sorted_array_search/prompts/"
    os.makedirs(exmps_output_path, exist_ok=True) 
    exmps_file = exmps_output_path + "sorted_array_search_exmps_"+time_string+".txt"

    ### initialize game and agents ###
    env_param = get_env_param(env_name="sorted_array_search")
    game = SAS(env_param=env_param)
    working_memory = {"A":game.A, "T":game.T, "n":game.n, "left":0, "right":game.n-1, "mid":0}
    agent = SASAgent(working_memory=working_memory, exmps_file=exmps_file)
    agents = {"agent":agent}

    ### start play ###
    play_through(game=game, agents=agent, logger=logger, exmps_file=exmps_file)
