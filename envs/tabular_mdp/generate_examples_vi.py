import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

from datetime import datetime
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from env import TabularMDP
from program_agent import VIAgent
from utils import Logger
from envs.env_helper import get_env_param

def play_through(game, agents, logger, exmps_file=None, mdp_known=True):
    assert game.check_agents(agents) # check if all agents required by the game are specified
    logger.write(game.description_of_problem_class)
    # describe the decision making problem instace to each agent
    for role in agents:
        instance_description = game.get_description(agent_role=role, mdp_known=mdp_known)
        logger.write(instance_description)
        with open(exmps_file, "a") as f:
            f.write("==== ASSISTANT ====\n")
            f.write(instance_description+"\n")

    q, v = game.compute_qVals()
    
    game.reset()

    agents['agent'].compute_policy()

    # evaluate policy
    v_policy = game.evaluate_policy(agents['agent'].working_memory["Q"])

    state = game.state
    epMaxVal = v[state.time_step][state.mathematical_descript]
    epPolicyVal = v_policy[state.time_step][state.mathematical_descript]
    logger.write("optimal value is {}, policy value is {}".format(epMaxVal, epPolicyVal))

    while not game.is_done:
        logger.write(state.textual_descript, color = "green")
        cur_agent = agents[state.cur_agent]
        # current agent choose an action from the set of legal actions
        action = cur_agent.move(state)
        logger.write("agent takes action {}".format(action), color = "green")
        state, reward = game.step(action)
        logger.write("agent gets reward {}".format(reward), color = "green")

        logger.write("The game has ended!", color="red")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="tabular_mdp")
    parser.add_argument('--mdp_known', type=bool, default=True)
    parser.add_argument('--random_param', type=bool, default=True)
    args = parser.parse_args()

    logger_output_path = "envs/tabular_mdp/outputs/"
    os.makedirs(logger_output_path, exist_ok=True) 
    now = datetime.now()
    time_string = now.strftime('%Y%m%d%H%M%S')
    logger = Logger(logger_output_path + args.game + "-" + time_string + ".html", verbose=True, writeToFile=False)

    exmps_output_path = "envs/tabular_mdp/prompts/"
    os.makedirs(exmps_output_path, exist_ok=True) 
    exmps_file = exmps_output_path + "tabular_mdp_vi_exmps_"+time_string+".txt"

    ### initialize game and agents ###
    env_param = get_env_param(env_name="tabular_mdp", random_param=args.random_param)
    game = TabularMDP(env_param=env_param)
    working_memory = {"P":game.P, "R":game.R, "nState":game.nState, "nAction":game.nAction, "epLen":game.epLen,
                      "V": np.zeros((game.epLen,game.nState)),
                      "Q": np.zeros((game.epLen,game.nState,game.nAction)),
                      }
    agent = VIAgent(working_memory=working_memory, exmps_file=exmps_file)
    agents = {"agent":agent}

    ### start play ###
    play_through(game=game, agents=agents, logger=logger, exmps_file=exmps_file, mdp_known=args.mdp_known)
