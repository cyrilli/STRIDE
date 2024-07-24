from tools import *
import os
import re

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

import numpy as np
from copy import deepcopy

class BargainAgent:
    def __init__(self, working_memory, exmps_file) -> None:
        self.working_memory = working_memory
        self.exmps_file = exmps_file

    def compute_spe(self, role):
        with open(self.exmps_file, "a") as f:
            f.write("==== USER ====\n")
            f.write("Question: Now compute the subgame perfect equilibrium (SPE) for each time step.\n")     
        for t in reversed(range(1, self.working_memory["T"]+1)):
            if t == self.working_memory["T"]:
                # last time step of bargaining
                if self.working_memory["T"] % 2 == 1:
                    with open(self.exmps_file, "a") as f:
                        if role == "buyer":
                            f.write("Thought: We can compute the SPE via backward induction from the last time step {} to time step 1. Since t={} is odd, it will be my turn to make the final offer. Seller will get 0 utility if rejecting my offer. Therefore, we can compute my optimal prices at time step {} by calling BackwardOneStep, and then compute the corresponding utility by calling CalcUtil.\n".format(t,t,t))
                        else:
                            f.write("Thought: We can compute the SPE via backward induction from the last time step {} to time step 1. Since t={} is odd, it will be buyer's turn to make the final offer. I will get 0 utility if rejecting buyer's offer. Therefore, we can compute buyer's optimal prices at time step {} by calling BackwardOneStep, and then compute the corresponding utility by calling CalcUtil.\n".format(t,t,t))
                    # buyer is the last agent to propose prices
                    agent = "buyer"
                else:
                    if role == "buyer":
                        with open(self.exmps_file, "a") as f:
                            f.write("Thought: We can compute the SPE via backward induction from the last time step {} to time step 1. Since t={} is even, it will be seller's turn to make the final offer. I will get 0 utility if rejecting seller's offer. Therefore, we can compute seller's optimal prices at time step {} by calling BackwardOneStep, and then compute the corresponding utility by calling CalcUtil.\n".format(t,t,t))
                    else:
                        with open(self.exmps_file, "a") as f:
                            f.write("Thought: We can compute the SPE via backward induction from the last time step {} to time step 1. Since t={} is even, it will be my turn to make the final offer. Buyer will get 0 utility if rejecting my offer. Therefore, we can compute my optimal prices at time step {} by calling BackwardOneStep, and then compute the corresponding utility by calling CalcUtil.\n".format(t,t,t))
                    # seller is the last agent to propose prices
                    agent = "seller"
                
                op = BackwardOneStep(agent=agent, opponent_util_if_rej=0.0, time_step=t)
                prices = op.execute(working_memory=self.working_memory)
                with open(self.exmps_file, "a") as f:
                    inputs = {"agent": agent, "opponent_util_if_rej": 0.0, "time_step": t}
                    f.write("Operation: call function BackwardOneStep with inputs {}.\n".format(inputs))
                    f.write("Result: {}\n".format(prices))
                match = re.search(r'\[(.*?)\]', prices)
                prices = match.group(1)
                prices = list(map(float, prices.split()))
                for price in prices:
                    if agent == "buyer":
                        assert abs(price) <= 1e-2
                    else:
                        assert abs(1-price) <= 1e-2
                op = CalcUtil(agent=agent, prices=prices, time_step=t)
                util = op.execute(working_memory=self.working_memory)
                with open(self.exmps_file, "a") as f:
                    inputs = {"agent": agent, "prices": prices, "time_step": t}
                    f.write("Operation: call function CalcUtil with inputs {}.\n".format(inputs))
                    f.write("Result: {}\n".format(util))
                util = float(util.split()[-1])
            else:
                # not last time step
                # switch role
                if agent == "buyer":
                    agent = "seller"
                    opponent = "buyer"
                    if role == "buyer":
                        with open(self.exmps_file, "a") as f:
                            f.write("Thought: Now we move one time step back to t={}, which is the seller's turn to make an offer. Based on our reasoning for time step t={}, I would get a utility of {} if the game continues to time step t={}. Therefore, we can compute {}'s optimal prices at time step {} by calling BackwardOneStep, and then compute the corresponding utility by calling CalcUtil.\n".format(t, t+1, util, t+1, agent, t))

                    else:
                        with open(self.exmps_file, "a") as f:
                            f.write("Thought: Now we move one time step back to t={}, which is my turn to make an offer. Based on our reasoning for time step t={}, {} would get a utility of {} if the game continues to time step t={}. Therefore, we can compute my optimal prices at time step {} by calling BackwardOneStep, and then compute the corresponding utility by calling CalcUtil.\n".format(t, t+1, opponent, util, t+1, t))
                else:
                    agent = "buyer"
                    opponent = "seller"
                    if role == "buyer":
                        with open(self.exmps_file, "a") as f:
                            f.write("Thought: Now we move one time step back to t={}, which is my turn to make an offer. Based on our reasoning for time step t={}, {} would get a utility of {} if the game continues to time step t={}. Therefore, we can compute my optimal prices at time step {} by calling BackwardOneStep, and then compute the corresponding utility by calling CalcUtil.\n".format(t, t+1, opponent, util, t+1, t))

                    else:
                        with open(self.exmps_file, "a") as f:
                            f.write("Thought: Now we move one time step back to t={}, which is the {}'s turn to make an offer. Based on our reasoning for time step t={}, I would get a utility of {} if the game continues to time step t={}. Therefore, we can compute {}'s optimal prices at time step {} by calling BackwardOneStep, and then compute the corresponding utility by calling CalcUtil.\n".format(t, agent, t+1, util, t+1, agent, t))
                inputs = {"agent": agent, "opponent_util_if_rej": util, "time_step": t}
                op = BackwardOneStep(agent=agent, opponent_util_if_rej=util, time_step=t)
                prices = op.execute(working_memory=self.working_memory)
                match = re.search(r'\[(.*?)\]', prices)
                prices = match.group(1)
                prices = list(map(float, prices.split()))
                op = CalcUtil(agent=agent, prices=prices, time_step=t)
                util = op.execute(working_memory=self.working_memory)

                with open(self.exmps_file, "a") as f:
                    f.write("Operation: call function BackwardOneStep with inputs {}.\n".format(inputs))
                    f.write("Result: {}\n".format(prices))
                    inputs = {"agent": agent, "prices": prices, "time_step": t}
                    f.write("Operation: call function CalcUtil with inputs {}.\n".format(inputs))
                    f.write("Result: {}\n".format(util))
                    util = float(util.split()[-1])
            self.working_memory["SPEPrice"][t] = prices
        with open(self.exmps_file, "a") as f:
            f.write("Thought: I have finished my reasoning from the last time step t={} to time step t=1, so I can exit the reasoning process, and save the computed subgame perfect equilibrium strategy in my working memory.\n".format(self.working_memory["T"]))

    def move(self, state):
        # if this is time step 1
        # do backward induction to know what price the agent should offer at time step 1
        if state.actions == [0.0, 1.0]:
            with open(self.exmps_file, "a") as f:
                f.write("==== USER ====\n")
                f.write("Question: {}\n".format(state.textual_descript))

            assert self.working_memory["SPEPrice"] != {}
            op = GetSPEPrice(agent=state.cur_agent, time_step=state.time_step)
            spe_price = op.execute(working_memory=self.working_memory)

            with open(self.exmps_file, "a") as f:
                f.write("Thought: Assuming my opponent is rational, then to maximize my utility, I should adopt the SPE strategy. Therefore, I should retrive the computed SPE price for the current time step {} by calling GetSPEPrice.\n".format(state.time_step))
                inputs = {"agent":state.cur_agent, "time_step":state.time_step}
                f.write("Operation: call function GetSPEPrice with inputs {}.\n".format(inputs))
                f.write("Result: {}\n".format(spe_price))
                match = re.search(r'\[(.*?)\]', spe_price)
                spe_price = match.group(1)
                spe_price = list(map(float, spe_price.split()))
                f.write("Thought: I can exit the reasoning process and propose a price of {}, which is my SPE strategy at time step {}.\n".format(spe_price, state.time_step))
            return spe_price
        else:
            # first assess utility of accepting opponent's offer
            price = state.mathematical_descript[-1]
            op = CalcUtil(agent=state.cur_agent, prices=price, time_step=state.time_step)
            util_acc = op.execute(working_memory=self.working_memory)

            if state.time_step < self.working_memory["T"]:
                with open(self.exmps_file, "a") as f:
                    f.write("==== USER ====\n")
                    f.write("Question: {}\n".format(state.textual_descript))
                    f.write("Thought: I should compare the utility I can get by accepting this price vs the utility I can get by rejecting it and making a counteroffer in the next time step. First, let me compute my utility for accepting the price of {} at the current time step {} by calling CalcUtil.\n".format(price, state.time_step))
                    inputs = {"agent":state.cur_agent, "price": price, "time_step":state.time_step}
                    f.write("Operation: call function CalcUtil with inputs {}.\n".format(inputs))
                    f.write("Result: {}\n".format(util_acc))

                # then asses utility of rejecting the offer, and make counter offer at next time step 2 (via backward induction)
                assert self.working_memory["SPEPrice"] != {}
                op = GetSPEPrice(agent=state.cur_agent, time_step=state.time_step+1)
                spe_price = op.execute(working_memory=self.working_memory)
                match = re.search(r'\[(.*?)\]', spe_price)
                spe_price = match.group(1)
                spe_price = list(map(float, spe_price.split()))
                op = CalcUtil(agent=state.cur_agent, prices=spe_price, time_step=state.time_step+1)
                util_rej = op.execute(working_memory=self.working_memory)

                with open(self.exmps_file, "a") as f:
                    f.write("Thought: Then I should reason about the utility I can get by rejecting and making a counteroffer at the next time step {}. Since I have already computed the subgame perfect equilirbium strategy and saved it in the working memory, I can retrive the previously computed price for the next time step {} by calling GetSPEPrice. Then I can compute my utility by calling CalcUtil.\n".format(state.time_step+1, state.time_step+1))
                    inputs = {"agent":state.cur_agent, "time_step":state.time_step+1}
                    f.write("Operation: call function GetSPEPrice with inputs {}.\n".format(inputs))
                    f.write("Result: {}\n".format(spe_price))
                    inputs = {"agent":state.cur_agent, "price": spe_price, "time_step":state.time_step+1}
                    f.write("Operation: call function CalcUtil with inputs {}.\n".format(inputs))
                    f.write("Result: {}\n".format(util_rej))
                util_acc = float(util_acc.split()[-1])
                util_rej = float(util_rej.split()[-1])
                if util_acc >= util_rej:
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: Since the utility I get by accepting the offer is greater or equal to that of rejecting the offer, I can exit the reasoning process and choose to accept the offer.\n")
                    return "accept"
                else:
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: Since the utility I get by accepting the offer is less than that of rejecting the offer, I can exit the reasoning process and choose to reject the offer.\n")
                    return "reject"
            else:
                with open(self.exmps_file, "a") as f:
                    f.write("==== USER ====\n")
                    f.write("Question: {}\n".format(state.textual_descript))
                    f.write("Thought: Since this is the last time step, as long as my utility of accepting the price of {} at the current time step {} is positive, I will accept it. Therefore, I should call function CalcUtil to compute the utility.\n".format(price, state.time_step))
                    inputs = {"agent":state.cur_agent, "price": price, "time_step":state.time_step}
                    f.write("Operation: call function CalcUtil with inputs {}.\n".format(inputs))
                    f.write("Result: {}\n".format(util_acc))
                util_acc = float(util_acc.split()[-1])
                if util_acc >= 0:
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: Since the utility I get by accepting the offer is greater or equal to 0, I can exit the reasoning process and choose to accept the offer.\n")
                    return "accept"
                else:
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: Since the utility I get by accepting the offer is less than 0, I can exit the reasoning process and choose to reject the offer.\n")
                    return "reject"