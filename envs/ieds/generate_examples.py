import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

from datetime import datetime
import numpy as np
import argparse

from env import IteratedEliminationDominatedStrategies
from program_agent import IEDSAgent
from utils import Logger
from envs.env_helper import get_env_param

def play_through(game, agents, logger, exmps_file=None):
    assert game.check_agents(agents)  # check if all agents required by the game are specified
    logger.write(game.description_of_problem_class)
    # describe the decision making problem instance to each agent
    for role in agents:
        instance_description = game.get_description(agent_role=role)
        logger.write(instance_description)
        with open(exmps_file, "a") as f:
            f.write("==== ASSISTANT ====\n")
            f.write(instance_description + "\n")

    game.reset()
    state = game.state

    while not game.is_done:
        logger.write(state.textual_descript, color="green")
        agents['player'].eliminate_dominated_strategies()
        state = game.step()

    logger.write("The IEDS game has ended!\nFinal strategies for player 1 are:\n{}\nFinal strategies for player 2 are:\n{}".format(np.array(state.remaining_strategies[0]), np.array(state.remaining_strategies[1])), color="red")
    logger.write("Final payoffs for player 1 are:\n{}\nFinal payoffs for player 2 are:\n{}".format(np.array(game.payoff_matrix[0]), np.array(game.payoff_matrix[1])), color="red")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="ieds")
    parser.add_argument('--random_param', action='store_true', default=False, help="Whether to use random parameters for the game")
    parser.add_argument('--num_strategies', nargs="+", type=int, help="Number of strategies for each player")
    parser.add_argument('--num_pne', type=int, default=1, help="Number of PNE in the game")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    args = parser.parse_args()

    logger_output_path = "envs/ieds/outputs/"
    os.makedirs(logger_output_path, exist_ok=True)
    now = datetime.now()
    time_string = now.strftime('%Y%m%d%H%M%S')
    logger = Logger(logger_output_path + args.game + "-" + time_string + ".html", verbose=True, writeToFile=False)

    exmps_output_path = "envs/ieds/prompts/"
    os.makedirs(exmps_output_path, exist_ok=True)
    exmps_file = exmps_output_path + "ieds_exmps_" + time_string + ".txt"

    ### initialize game and agents ###
    env_param = get_env_param(env_name="ieds", random_param=args.random_param, num_pne=args.num_pne, num_strategies=args.num_strategies)
    game = IteratedEliminationDominatedStrategies(env_param=env_param)
    working_memory = {
        "payoff_matrix": game.payoff_matrix,
        "current_player": game.cur_player,
        "remaining_strategies": game.state.remaining_strategies,
        "dominated_strategies": game.dominated_strategies
    }
    agent = IEDSAgent(working_memory=working_memory, exmps_file=exmps_file)
    agents = {"player": agent}

    ### start play ###
    play_through(game=game, agents=agents, logger=logger, exmps_file=exmps_file)
