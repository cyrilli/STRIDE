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
from env import TabularMDP
from program_agent import UCBVIAgent
from utils import Logger
from envs.env_helper import get_env_param

def play_through(game, agents, logger, n_episodes, exmps_file=None, write_exmps=False, mdp_known=True):
    assert game.check_agents(agents) # check if all agents required by the game are specified
    # describe the decision making problem instace to each agent

    cumulative_regret = 0
    cumulative_regret_ls = []
    cumulative_reward_ls = []
    q, v = game.compute_qVals()
    
    for ep in range(n_episodes):

        for role in agents:
            instance_description = game.get_description(agent_role=role, episode_ind=ep, mdp_known=mdp_known, external_summarization=False)
            logger.write(instance_description)
            if write_exmps:
                with open(exmps_file, "a") as f:
                    f.write("==== ASSISTANT ====\n")
                    f.write(instance_description+"\n")

        cumulative_reward = 0
        game.reset()
        agents['agent'].reset()
        logger.write("episode {}/{}...".format(ep, n_episodes-1), color = "green")
        agents['agent'].compute_policy()

        # evaluate policy
        v_policy = game.evaluate_policy(agents['agent'].working_memory["Q"])

        state = game.state
        epMaxVal = v[state.time_step][state.mathematical_descript]
        epPolicyVal = v_policy[state.time_step][state.mathematical_descript]
        cumulative_regret += (epMaxVal - epPolicyVal)
        cumulative_regret_ls.append(cumulative_regret)
        while not game.is_done:
            # logger.write(state.textual_descript, color = "green")
            cur_agent = agents[state.cur_agent]
            old_state = deepcopy(state)
            # current agent choose an action from the set of legal actions
            action = cur_agent.move(state)
            # logger.write("agent takes action {}".format(action), color = "green")
            state, reward = game.step(action)
            cumulative_reward += reward
            if not mdp_known:
                # estimate transition and reward function
                cur_agent.update_obs(old_state, action, state, reward)
        cumulative_reward_ls.append(cumulative_reward)
        # logger.write("The game has ended!", color="red")
        logger.write("optimal value is {}, policy value is {}".format(epMaxVal, epPolicyVal))

        P_error = np.sum(np.square(game.P - agents["agent"].working_memory["P"]))/np.sum(game.P.shape)
        logger.write("P mean squared error {}".format(P_error))
        R_error = np.sum(np.square(game.R[:,:,0] - agents["agent"].working_memory["R"][:,:,0]))/np.sum(game.R[:,:,0].shape)
        logger.write("R mean squared error {}".format(R_error))

    return cumulative_reward_ls, cumulative_regret_ls

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="tabular_mdp")
    parser.add_argument('--mdp_known', type=bool, default=False)
    parser.add_argument('--random_param', type=bool, default=True)
    parser.add_argument('--write_exmps', type=bool, default=False)
    parser.add_argument('--n_episodes', type=int, default=100, help='number of episodes')
    args = parser.parse_args()

    logger_output_path = "envs/tabular_mdp/outputs/"
    os.makedirs(logger_output_path, exist_ok=True) 
    now = datetime.now()
    time_string = now.strftime('%Y%m%d%H%M%S')
    logger = Logger(logger_output_path + args.game + "-" + time_string + ".html", verbose=True, writeToFile=False)

    ### initialize game and agents ###
    exmps_output_path = "envs/tabular_mdp/prompts/"
    os.makedirs(exmps_output_path, exist_ok=True) 
    exmps_file = exmps_output_path + "tabular_mdp_ucbvi_exmps_"+time_string+".txt"

    ### initialize game and agents ###
    env_param = get_env_param(env_name="tabular_mdp", random_param=args.random_param)
    env_param["n_episodes"] = args.n_episodes
    game = TabularMDP(env_param=env_param, mdp_known=False)        

    # the agent does not know P and R, and thus here we initialize its estimation
    R_init = np.zeros((game.nState, game.nAction, 2))
    P_init = np.ones((game.nState, game.nAction, game.nState)) * 1.0 / game.nState
    assert R_init.shape == game.R.shape
    assert P_init.shape == game.P.shape

    bonus_scale_factor = 0.1
    working_memory = {"P":P_init, "R":R_init, 
                    "nState":game.nState, "nAction":game.nAction, "epLen":game.epLen,
                    "V": np.zeros((game.epLen,game.nState)),
                    "Q": np.zeros((game.epLen,game.nState,game.nAction)),
                    "Nsa": np.ones((game.nState, game.nAction)),
                    "bonus_scale_factor": bonus_scale_factor,
                    "epNum":args.n_episodes,
                    }
    agent = UCBVIAgent(working_memory=working_memory, exmps_file=exmps_file, write_exmps=args.write_exmps)
    agents = {"agent":agent}

    ### start play ###
    cumulative_reward_ls, cumulative_regret_ls = play_through(game=game, agents=agents, logger=logger, n_episodes=args.n_episodes, exmps_file=exmps_file, write_exmps=args.write_exmps, mdp_known=args.mdp_known)

    logger.write("cumulative rewards over episode: {}".format(cumulative_reward_ls))
    logger.write("cumulative regret over episode: {}".format(cumulative_regret_ls))