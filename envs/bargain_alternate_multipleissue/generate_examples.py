import sys
import os
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from env import BargainAlternateMultiIssue
from program_agent import BargainAgent
# from utils import Logger
from envs.env_helper import get_env_param
from agents.Random import RandomAgent
from test import *
from rich.console import Console

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

def play_through(game, agents, logger, n_episodes, exmps_files={}):
    # assert game.check_agents(agents)  # check if all agents required by the game are specified
    # describe the decision making problem instance to each agent
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    
    for role in agents:
        instance_description = game.get_description(agent_role=role)
        with open(os.path.join(current_dir, exmps_files[role]), "a") as f:
            f.write("==== ASSISTANT ====\n")
            f.write(instance_description + "\n")
        logger.write(instance_description)

    for ep in range(n_episodes):
        game.reset()
        logger.write("episode {}/{}...".format(ep, n_episodes - 1), color="green")

        agents['buyer'].compute_spe(role='buyer')
        agents['seller'].compute_spe(role='seller')
        # new_working_memory = agents['seller'].working_memory

        state = game.state

        while not game.is_done:
            logger.write(state.textual_descript, color="green")
            cur_agent = agents[state.cur_agent]
            # current agent choose an action from the set of legal actions
            action = cur_agent.move(state)
            if game.state.actions == [0.0, 1.0]:
                if cur_agent == "buyer":
                    discounts = env_param['buyerDiscount']
                    weights = env_param['buyerWeight']
                else:
                    discounts = env_param["sellerDiscount"]
                    weights = env_param["sellerWeight"]
                util, _ = game.calculate_utility(game.state.time_step, discounts, weights, action)
                util = np.round(util, 2)
                print("spe price {} and utility {}.".format(action, util))
            logger.write("{}: {}".format(state.cur_agent, action, 2), color="green")
            state, _ = game.step(action)

        logger.write("The game has ended!", color="red")

class Logger(object):
    def __init__(self, log_file, verbose=True):
        self.console = Console(record=True)
        self.log_file = log_file
        self.verbose = verbose

        self.write("All outputs written to %s" % log_file)
        print(self.is_initialized())
        
    def write(self, message, color=None):
        try:
            self.console.save_html(self.log_file, clear=False)
            if self.verbose: 
                if color is not None:
                    self.console.print("[{}]".format(color) + message + "[/{}]".format(color))
                else:
                    self.console.print(message)
        except Exception as e:
            if self.verbose:
                self.console.print(f"Error writing to log: {e}")

    def is_initialized(self):
        try:
            with open(self.log_file, 'a'):
                pass
            return f"Logger successfully initialized. Log file is writable: {self.log_file}"
        except IOError:
            return f"Logger initialization failed. Log file is not writable: {self.log_file}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="bargain_alternate_multipleissue")
    parser.add_argument('--random_param', type=bool, default=True)
    parser.add_argument('--n_episodes', type=int, default=5, help='number of episodes')
    parser.add_argument('--output_path', type=str, default="./outputs/", help='path to save the output')
    parser.add_argument('--verbose', type=int, default=1, help="0: not logger.write, 1: logger.write")
    args = parser.parse_args()

    current_file_path = os.path.abspath(__file__)

    current_folder_path = os.path.dirname(current_file_path)

    output_path = os.path.join(current_folder_path, "outputs")
    print(output_path)
    os.makedirs(output_path, exist_ok=True)
    now = datetime.now()
    time_string = now.strftime('%Y%m%d%H%M%S')
    logger = Logger(output_path + "/" + args.game + "-" + time_string + ".html", args.verbose)

    ### initialize game and agents ###
    env_param = get_env_param(env_name="bargain_alternate_multipleissue", random_param=args.random_param)
    # env_param = {
    #     "T": 4,
    #     "buyerDiscount": [0.83,0.78,0.48 ,0.01],
    #     "sellerDiscount": [0.87,0.05,0.16, 0.86],
    #     "buyerWeight": [0.45,0.03,0.87,0.39],
    #     "sellerWeight": [0.14,0.17,0.07,0.89]
    # }
    
    # env_param = {
    #     "T": 3,
    #     "buyerDiscount": [0.05, 0.42],
    #     "sellerDiscount": [0.77, 0.68],
    #     "buyerWeight": [0.13, 0.7],
    #     "sellerWeight": [0.54, 0.55]
    # }
    game = BargainAlternateMultiIssue(env_param=env_param)
    buyer_exmps_file = "prompts/buyer_exmps.txt"
    seller_exmps_file = "prompts/seller_exmps.txt"
    working_memory = {
        "T": game.T, 
        "buyerDiscounts": game.buyerDiscount, 
        "sellerDiscounts": game.sellerDiscount,
        "SPEPrices": {}, 
        "buyerWeights": game.buyerWeight, 
        "sellerWeights": game.sellerWeight,                                                                                                                                                                                                                                                                
    }

    if "SPEPrice" not in working_memory:
        working_memory["SPEPrice"] = {}
    buyer = BargainAgent(working_memory=deepcopy(working_memory), exmps_file=buyer_exmps_file)
    seller = BargainAgent(working_memory=deepcopy(working_memory), exmps_file=seller_exmps_file)
    agents = {"buyer": buyer, "seller": seller}

    ### start play ###
    play_through(game=game, agents=agents, logger=logger, n_episodes=args.n_episodes, exmps_files={"buyer": buyer_exmps_file, "seller": seller_exmps_file})
