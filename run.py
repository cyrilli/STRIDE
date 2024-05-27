from datetime import datetime
import os
import argparse
import matplotlib.pyplot as plt
from rich.console import Console
from utils import Logger, create_env, create_agents
import numpy as np
from copy import deepcopy

class Logger(object):
    def __init__(self, log_file, verbose=True):
        self.console = Console(record=True)
        self.log_file = log_file
        self.verbose = verbose

        self.write("All outputs written to %s" % log_file)
        return 

    def write(self, message, color=None):
        self.console.save_html(self.log_file, clear=False)
        if(self.verbose): 
            if color is not None:
                self.console.print("[{}]".format(color)+message+"[/{}]".format(color))
            else:
                self.console.print(message)

def play_through(env, agents, logger, args):
    for ep in range(1, args.n_episodes+1):

        # repeatively play the same instance of the env (e.g., MDP with unknown env model)
        env.reset()
        logger.write("episode {}/{} ...".format(ep, args.n_episodes), color = "red")

        for role in agents:
            agents[role].reset()
            if env.name == "tabular_mdp":
                instance_description = env.get_description(agent_role=role, episode_ind=ep, mdp_known=args.mdp_known)
            else:
                instance_description = env.get_description(agent_role=role)
            agents[role].get_instance_info(instance_description)
            logger.write("To {}:".format(role))
            logger.write(instance_description)

        if env.name == "tabular_mdp":
            if args.mdp_known:
                agents["agent"].reason("Now compute the optimal policy, that is, the optimal action at each step and each state.")
            else:
                agents["agent"].reason("Now compute the optimistic policy based on your current estimation of transition function P and reward function R.")
        elif env.name == "dynamic_mechanism_design":
            agents["designer"].reason("Now compute the optimal policy that maximizes all agents' rewards.")
        elif env.name == "bargain_alternate_singleissue":
            agents["buyer"].reason("Now compute the subgame perfect equilibrium (SPE) step by step.")
            agents["seller"].reason("Now compute the subgame perfect equilibrium (SPE) step by step.")

        metric_ls = []
        state = env.state # initial state
        while not env.is_done: # in case game never ends due to failure in checking terminal condition
            logger.write(state.textual_descript, color = "red")
            cur_agent = agents[state.cur_agent]
            action = cur_agent.move(state)
            old_state = deepcopy(state)
            # compute some performance metric
            metric = get_result(env, agents, state, action, logger)
            metric_ls.append(metric)

            logger.write("{}: {}".format(state.cur_agent, action), color = "red")
            logger.write("metric: {}".format(metric), color = "red")
            state, reward = env.step(action)
            if env.name == "tabular_mdp" and not args.mdp_known:
                cur_agent.reason("After taking action {} at state {}, the state has transit to {} and the agent receives reward {}.\n".format(action, old_state.mathematical_descript, state.mathematical_descript, reward))

        if env.name == "dynamic_mechanism_design":
            q_max_all, v_max_all = env.compute_qVals(agent_to_exclude=None)
            # compute the VCG prices
            for i in range(env.nAgent):
                charged_price = agents["designer"].charge_price("Now compute the VCG price for agent {}.".format(i))

                q_max_exclude_i, v_max_exclude_i = env.compute_qVals(agent_to_exclude=i)
                v_policy = env.evaluate_policy(q_max_all, agent_to_exclude=i)
                vcg_price = v_max_exclude_i[0,0] - v_policy[0,0]
                logger.write("agent {}: charged price {} vcg price {}".format(i, charged_price, vcg_price))
                if abs(charged_price - vcg_price) <= 1e-2:
                    logger.write("metric: {}".format(True), color = "red")
                    metric_ls.append(True)
                else:
                    logger.write("metric: {}".format(False), color = "red")
                    metric_ls.append(False)

        logger.write("This episode has ended!", color="red")
        logger.write("Performance metric: {}".format(metric_ls))
    return metric_ls

def get_result(env, agents, state, action, logger):
    if env.name in ["tabular_mdp", "dynamic_mechanism_design"]:
        q_optimal, _ = env.compute_qVals()
        q = q_optimal[state.time_step, state.mathematical_descript]
        logger.write("q_optimal for current step and state {}".format(q))
        optimal_actions = np.where(q==np.max(q))
        if action in optimal_actions:
            success = True
        else:
            success = False
        return success
    if env.name == "bargain_alternate_singleissue":
        # the current agent is proposing a price
        # let's see if this price is spe price
        if state.actions == [0.0, 1.0]:
            price, util = env.calculate_spe_price_utility(cur_time=state.time_step, cur_player=state.cur_agent, deadline=env.T, buyer_discount=env.buyerDiscount, seller_discount=env.sellerDiscount)
            # print("spe price {} and utility {}.".format(price, util))
            logger.write("spe price {}, {} proposed price {}".format(price, state.cur_agent, action))
            if abs(price-action) <= 1e-2:
                success = True
            else:
                success = False
        else:
            # the current agent is deciding to acc or rej
            if state.cur_agent == "buyer":
                discount = env.env_param["buyerDiscount"]
                value = 1.0
            else:
                discount = env.env_param["sellerDiscount"]
                value = 0.0
            # utility of acc
            price = state.mathematical_descript[-1]
            util_acc = abs(price-value) * discount**(state.time_step-1)
            _, util_rej = env.calculate_spe_price_utility(cur_time=state.time_step+1, cur_player=state.cur_agent, deadline=env.T, buyer_discount=env.buyerDiscount, seller_discount=env.sellerDiscount)
            logger.write("utility accept {}, utility reject {}, {} action {}".format(util_acc, util_rej, state.cur_agent, action))
            if util_acc >= util_rej - 0.01:
                if action == "accept":
                    success = True
                else:
                    success = False
            else:
                if action == "accept":
                    success = False
                else:
                    success = True
        return success
    if env.name == "bargain_onesided_uncertainty":
        # the current agent, seller, is proposing a price
        # let's see if this price is spe price
        if state.actions == [0.0, 1.0]:
            se_prices = env.get_se_prices()
            price = se_prices[state.time_step]
            logger.write("spe price {}".format(price))
            if abs(price-action) <= 1e-2:
                success = True
            else:
                success = False
        else:
            # the current agent, buyer, is deciding to acc or rej
            # utility of acc
            discount = env.buyerDiscount
            price = state.mathematical_descript[-1]
            util_acc = (env.buyerVal-price) * discount**(state.time_step-1)
            if state.time_step == env.T:
                util_rej = 0.0
            else:
                se_prices = env.get_se_prices()
                se_price_next_time = se_prices[state.time_step+1]
                util_rej = (env.buyerVal-se_price_next_time) * discount**(state.time_step)
            logger.write("utility accept {}, utility reject {}, {} action {}".format(util_acc, util_rej, state.cur_agent, action))
            if util_acc >= util_rej - 0.01:
                if action == "accept":
                    success = True
                else:
                    success = False
            else:
                if action == "accept":
                    success = False
                else:
                    success = True
        return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="tabular_mdp", help="[tabular_mdp, dynamic_mechanism_design, bargain_alternate_singleissue, bargain_onesided_uncertainty]")
    parser.add_argument('--mdp_known', type=bool, default=True)
    parser.add_argument('--agent_type', type=str, default="stride", help="[stride]")
    parser.add_argument('--agent_engine', type=str, default="gpt-4o", help="[gpt-3.5-turbo, gpt-4o, gpt-4-turbo]")
    parser.add_argument('--random_param', type=bool, default=True)
    parser.add_argument('--n_exps', type=int, default=1, help='number of times to play in the environment')
    parser.add_argument('--n_episodes', type=int, default=1, help='number of episodes')
    parser.add_argument('--output_path', type=str, default="./outputs/", help='path to save the output')
    parser.add_argument('--verbose', type=int, default=1, help="0: not logger.write, 1: logger.write")
    args = parser.parse_args()

    output_path = "./outputs/" + args.env + "/"
    os.makedirs(output_path, exist_ok=True) 
    now = datetime.now()
    time_string = now.strftime('%Y%m%d%H%M%S')
    logger = Logger(output_path + args.env + "-" + time_string + ".html", args.verbose)

    result_list = []
    for exp in range(1, args.n_exps+1):
        logger.write("experiment {}/{} ...".format(exp, args.n_exps), color = "red")
        
        # initialize the environment and agents
        env = create_env(args.env, args.random_param, args.mdp_known)
        if args.env == "tabular_mdp" and not args.mdp_known:
            env.env_param["n_episodes"] = args.n_episodes
            env.n_episodes = args.n_episodes
        agents = create_agents(env, logger, args.agent_type, args.agent_engine, args.mdp_known)
        if not env.check_agents(agents): # check if all agents required by the env are specified
            raise ValueError("illegal agents for env {}".format(args.env))
        
        # start playing
        logger.write("Start to play {}".format(env.name), color = "red")
        result = play_through(env=env, agents=agents, logger=logger, args=args)
        result_list.append(result)
    
    if args.env == "tabular_mdp":
        total_success = 0.0
        for res in result_list:
            for r in res:
                if r:
                    total_success += 1.0
        logger.write("success rate is {}={}/{}".format(total_success/(args.n_exps*env.epLen), total_success, args.n_exps*env.epLen))
    elif args.env == "bargain_alternate_singleissue" or "bargain_onesided_uncertainty":
        total_success = 0.0
        total_num = 0.0
        for res in result_list:
            total_num += len(res)
            for r in res:
                if r:
                    total_success += 1.0
        logger.write("success rate is {}={}/{}".format(total_success/total_num, total_success, total_num))