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

from envs.sorted_array_search.env import SAS
from envs.edit_distance_problem.env import EDP
from envs.shortest_path_problem.env import SPP
from envs.knapsack_problem.env import KSP

from NPHardEval_data.data_miners.BSP_interface import BSP_data_reader
from NPHardEval_data.data_miners.EDP_interface import EDP_data_reader
from NPHardEval_data.data_miners.KSP_interface import KSP_data_reader
from NPHardEval_data.data_miners.SPP_interface import SPP_data_reader

from utils import Logger, load_initial_instructions
from envs.env_helper import get_env_param

from agents.StriDe import StriDeAgent, StriDeFlowAgent

def play_through(problem, agent, logger):
    # describe the decision making problem instace to each agent
    instance_description = problem.get_description()
    logger.write(instance_description)
    agent.get_instance_info(instance_description)
    if problem.name == "sorted_array_search":
        agent.reason("What is index of the target value?")
        computed_value = agent.working_memory["mid"]
        logger.write("agent computed index: {}".format(computed_value))
    if problem.name == "edit_distance_problem":
        agent.reason("What is the minimum number of operations needed for transforming the first string to the other?")
        computed_value = agent.working_memory["dp"][agent.working_memory["m"]][agent.working_memory["n"]]
        print(agent.working_memory["dp"][agent.working_memory["m"]][agent.working_memory["n"]])
        logger.write("agent computed number of operations: {}".format(computed_value))
    if problem.name == "shortest_path_problem":
        agent.reason("What is the distance of the shortest path between the the two vertices?")
        computed_value = agent.working_memory["dists"][agent.working_memory["end"]]
        logger.write("agent computed distance: {}".format(computed_value))
    if problem.name == "knapsack_problem":
        agent.reason("What is the value of the subset of items which has the maximum value without exceeding the weight capacity?")
        computed_value = agent.working_memory["dp"][agent.working_memory["n"]][agent.working_memory["capacity"]]
        logger.write("agent computed value: {}".format(computed_value))

    logger.write("End of the game.")

    return computed_value
    
    # while True:
    #     query = input("human ask question?")
    #     agent.reason(query)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default="knapsack_problem")
    parser.add_argument('--agent_engine', type=str, default="gpt-3.5-turbo", help="[gpt-3.5-turbo, gpt-4o, gpt-4-turbo]")
    parser.add_argument('--agent_type', type=str, default="strideflow", help="[stride, strideflow]")
    parser.add_argument('--random_param', type=bool, default=False)
    parser.add_argument('--n_exps', type=int, default=1, help='number of times to play in the environment')
    args = parser.parse_args()

    logger_output_path = "./outputs/" + args.problem + "/"
    os.makedirs(logger_output_path, exist_ok=True) 
    now = datetime.now()
    time_string = now.strftime('%Y%m%d%H%M%S')
    logger = Logger(logger_output_path + args.problem + "-" + time_string + ".html", verbose=True, writeToFile=True)

    if args.problem == "sorted_array_search":
        A, T, Answers, Levels = BSP_data_reader()

    if args.problem == "edit_distance_problem":
        A, B, Answers, Levels = EDP_data_reader()

    if args.problem == "shortest_path_problem":
        nodes, edges, Answers, Levels = SPP_data_reader()

    if args.problem == "knapsack_problem":
        items, capacity, Answers, Levels = KSP_data_reader()
    
    level_percentage = []
    mistakes = []
    Corrects = 0
    for exp in range(1, args.n_exps+1):
        logger.write("experiment {}/{} ...".format(exp, args.n_exps), color = "red")
        ## initialize game and agents ###
        # env_param = get_env_param(env_name=args.problem, random_param=args.random_param)

        exp = exp + 98

        if args.problem == "sorted_array_search":
            env_param = {'A':A[exp-1],'T':T[exp-1]}
            problem = SAS(env_param=env_param)
            demo = load_initial_instructions("envs/sorted_array_search/prompts/sorted_array_search_exmps_20240813130834.txt")
            from envs.sorted_array_search.tools import tool_names_sorted_array_search
            init_memory = {"A":problem.A,"T":problem.T,"n":problem.n,"left":0,"right":problem.n-1,"mid":0}
            agent = StriDeAgent(problem_description=problem.description_of_problem_class, demo=demo, tool_names=tool_names_sorted_array_search, init_memory=init_memory, logger=logger, engine=args.agent_engine)
        
        if args.problem == "edit_distance_problem":
            env_param = {'A':A[exp-1],'B':B[exp-1]}
            problem = EDP(env_param=env_param)
            demo = load_initial_instructions("envs/edit_distance_problem/prompts/edit_distance_problem_exmps_20240811155153.txt")
            from envs.edit_distance_problem.tools import tool_names_edit_distance_problem
            init_memory = {"A":problem.A, "B":problem.B, "m":problem.m, "n":problem.n, "dp":[]}
            agent = StriDeAgent(problem_description=problem.description_of_problem_class, demo=demo, tool_names=tool_names_edit_distance_problem, init_memory=init_memory, logger=logger, engine=args.agent_engine)
        
        if args.problem == "shortest_path_problem":
            env_param = {'nodes':nodes[exp-1], 'edges':edges[exp-1]}
            problem = SPP(env_param=env_param)
            demo = load_initial_instructions("envs/shortest_path_problem/prompts/shortest_path_problem_exmps_20240813115610.txt")
            from envs.shortest_path_problem.tools import tool_names_shortest_path_problem
            init_memory = {"nodes":problem.nodes, "edges":problem.edges, "start":problem.start, "end":problem.end, "Q":[], "dists":{}}
            agent = StriDeAgent(problem_description=problem.description_of_problem_class, demo=demo, tool_names=tool_names_shortest_path_problem, init_memory=init_memory, logger=logger, engine=args.agent_engine)
        
        if args.problem == "knapsack_problem":
            env_param = {'items': items[exp-1], 'capacity': capacity[exp-1]}
            problem = KSP(env_param=env_param)
            demo = load_initial_instructions("envs/knapsack_problem/prompts/knapsack_problem_flow_exmps_20240814110924.txt")
            from envs.knapsack_problem.tools import tool_names_knapsack_problem
            init_memory = {"items":problem.items, "capacity":problem.capacity, "n":problem.n, "dp":[]}
            if args.agent_type == "stride":
                agent = StriDeAgent(problem_description=problem.description_of_problem_class, demo=demo, tool_names=tool_names_knapsack_problem, init_memory=init_memory, logger=logger, engine=args.agent_engine)
            elif args.agent_type == "strideflow":
                agent = StriDeFlowAgent(problem_description=problem.description_of_problem_class, demo=demo, tool_names=tool_names_knapsack_problem, init_memory=init_memory, logger=logger, engine=args.agent_engine)



        ### start play ###
        computed_value = play_through(problem=problem, agent=agent, logger=logger)

        if(computed_value == Answers[exp-1]):
            Corrects += 1
        elif (computed_value == float('infinity')):
            if(Answers[exp-1] == None):
                Corrects += 1
        else:
            mistakes.append((exp,Answers[exp-1],computed_value))
        
        if(exp%10==0):
            level_percentage.append(Corrects/10)
            Corrects = 0
    print(level_percentage)
    print(mistakes)
