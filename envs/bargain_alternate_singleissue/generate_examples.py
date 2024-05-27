import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

from datetime import datetime
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from env import BargainAlternateSingleIssue
from program_agent import BargainAgent
from agents.Random import RandomAgent
from utils import Logger
from envs.env_helper import get_env_param

def play_through(game, agents, logger, n_episodes, exmps_files={}):
    assert game.check_agents(agents) # check if all agents required by the game are specified
    # describe the decision making problem instace to each agent
    for role in agents:
        instance_description = game.get_description(agent_role=role)
        with open(exmps_files[role], "a") as f:
            f.write("==== ASSISTANT ====\n")
            f.write(instance_description+"\n")
        logger.write(instance_description)
    
    for ep in range(n_episodes):
        game.reset()
        logger.write("episode {}/{}...".format(ep, n_episodes-1), color = "green")

        agents['buyer'].compute_spe(role='buyer')
        agents['seller'].compute_spe(role='seller')

        state = game.state

        while not game.is_done:
            logger.write(state.textual_descript, color = "green")
            cur_agent = agents[state.cur_agent]
            # current agent choose an action from the set of legal actions
            action = cur_agent.move(state)
            if game.state.actions == [0.0, 1.0]:
                price, util = game.calculate_spe_price_utility(cur_time=game.state.time_step, cur_player=game.state.cur_agent, deadline=game.T, buyer_discount=game.buyerDiscount, seller_discount=game.sellerDiscount)
                print("spe price {} and utility {}.".format(price, util))
            logger.write("{}: {}".format(state.cur_agent, action), color = "green")
            state, _ = game.step(action)

        logger.write("The game has ended!", color="red")

    # plt.plot(cumulative_regret_ls)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="bargain_alternate_singleissue")
    parser.add_argument('--random_param', type=bool, default=True)
    parser.add_argument('--n_episodes', type=int, default=1, help='number of episodes')
    args = parser.parse_args()

    logger_output_path = "envs/bargain_alternate_singleissue/outputs/"
    os.makedirs(logger_output_path, exist_ok=True) 
    now = datetime.now()
    time_string = now.strftime('%Y%m%d%H%M%S')
    logger = Logger(logger_output_path + args.game + "-" + time_string + ".html", verbose=True, writeToFile=False)

    exmps_output_path = "envs/bargain_alternate_singleissue/prompts/"
    os.makedirs(exmps_output_path, exist_ok=True) 
    buyer_exmps_file = exmps_output_path + "buyer_exmps_"+time_string+".txt"
    seller_exmps_file = exmps_output_path + "seller_exmps_"+time_string+".txt"

    ### initialize game and agents ###
    env_param = get_env_param(env_name="bargain_alternate_singleissue", random_param=args.random_param)
    game = BargainAlternateSingleIssue(env_param=env_param)

    working_memory = {"T":game.T, "delta_b":game.buyerDiscount, "delta_s":game.sellerDiscount,
                      "SPEPrice": {},
                      }
    buyer = BargainAgent(working_memory=deepcopy(working_memory), exmps_file=buyer_exmps_file)
    # buyer = RandomAgent()
    seller = BargainAgent(working_memory=deepcopy(working_memory), exmps_file=seller_exmps_file)
    # seller = RandomAgent()
    agents = {"buyer":buyer, "seller":seller}

    ### start play ###
    play_through(game=game, agents=agents, logger=logger, n_episodes=args.n_episodes, exmps_files={"buyer":buyer_exmps_file, "seller":seller_exmps_file})
