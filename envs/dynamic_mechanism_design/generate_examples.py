import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

from datetime import datetime
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from env import DynamicMechanismDesign
from program_agent import DynamicVCGAgent
from utils import Logger
from envs.env_helper import get_env_param

def play_through(game, agents, logger, n_episodes, exmps_file=None):
    assert game.check_agents(agents) # check if all agents required by the game are specified
    # describe the decision making problem instace to each agent

    instance_description = game.get_description(agent_role="designer")
    with open(exmps_file, "a") as f:
        f.write("==== ASSISTANT ====\n")
        # f.write(game.description_of_mdp+"\n")
        f.write(instance_description+"\n")
    logger.write(instance_description)
    q_max_all, v_max_all = game.compute_qVals(agent_to_exclude=None)
    
    for ep in range(n_episodes):
        game.reset()
        logger.write("episode {}/{}...".format(ep, n_episodes-1), color = "green")

        agents["designer"].compute_policy()

        # evaluate designer's chosen policy on MDP that considers all agents' reward
        v_policy = game.evaluate_policy(agents["designer"].working_memory["Q"])

        state = game.state
        epMaxVal = v_max_all[state.time_step][state.mathematical_descript]
        epPolicyVal = v_policy[state.time_step][state.mathematical_descript]
        logger.write("optimal value is {}, policy value is {}".format(epMaxVal, epPolicyVal))
        while not game.is_done:
            logger.write(state.textual_descript, color = "green")
            cur_agent = agents[state.cur_agent]
            # current agent choose an action from the set of legal actions
            action = cur_agent.move(state)
            logger.write("designer takes action {}".format(action), color = "green")
            state, reward = game.step(action)

        # compute the VCG prices
        for i in range(game.nAgent):
            charged_price = agents["designer"].compute_price(agent=i)

            q_max_exclude_i, v_max_exclude_i = game.compute_qVals(agent_to_exclude=i)
            v_policy = game.evaluate_policy(q_max_all, agent_to_exclude=i)
            # print("====")
            # print(v_max_exclude_i[0,0])
            # print(v_policy[0,0])
            vcg_price = v_max_exclude_i[0,0] - v_policy[0,0]
            logger.write("agent {}: charged price {} vcg price {}".format(i, charged_price, vcg_price))
        logger.write("The game has ended!", color="red")

    # plt.plot(cumulative_regret_ls)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="dynamic_mechanism_design", help="[dynamic_mechanism_design]")
    parser.add_argument('--random_param', type=bool, default=True)
    parser.add_argument('--n_episodes', type=int, default=1, help='number of episodes')
    args = parser.parse_args()

    logger_output_path = "envs/dynamic_mechanism_design/outputs/"
    os.makedirs(logger_output_path, exist_ok=True) 
    now = datetime.now()
    time_string = now.strftime('%Y%m%d%H%M%S')
    logger = Logger(logger_output_path + args.game + "-" + time_string + ".html", verbose=True, writeToFile=False)

    exmps_output_path = "envs/dynamic_mechanism_design/prompts/"
    os.makedirs(exmps_output_path, exist_ok=True) 
    exmps_file = exmps_output_path + "dynamic_vcg_exmps_"+time_string+".txt"

    ### initialize game and agents ###
    env_param = get_env_param(env_name="dynamic_mechanism_design", random_param=args.random_param)
    game = DynamicMechanismDesign(env_param=env_param)

    working_memory = {"P":game.P, "R":game.R, "nState":game.nState, "nAction":game.nAction, "epLen":game.epLen, 
                      "nAgent":game.nAgent,
                      "V": np.zeros((game.epLen,game.nState)),
                      "Q": np.zeros((game.epLen,game.nState,game.nAction)),
                      "VExcluding": np.zeros((game.nAgent,game.epLen,game.nState)),
                      "QExcluding": np.zeros((game.nAgent,game.epLen,game.nState,game.nAction)),
                      }
    agent = DynamicVCGAgent(working_memory=working_memory, exmps_file=exmps_file)
    agents = {"designer":agent}

    ### start play ###
    play_through(game=game, agents=agents, logger=logger, n_episodes=args.n_episodes, exmps_file=exmps_file)
