import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

import numpy as np
from copy import deepcopy
from pydantic import BaseModel, Field, NonNegativeFloat
from typing import List, Optional

from envs.env_helper import BaseEnv

description_of_problem_class="""
The alternating offer bargaining game is a negotiation framework between two players, a buyer and a seller, aimed at determining the price of an item. This strategic game plays out over several rounds with a finite deadline, emphasizing the tactics of bargaining under time constraints.

Components:
Players: Two (Buyer and Seller).
Buyer's Value: 1 (the maximum price the buyer is willing to pay).
Seller's Value: 0 (the minimum price the seller is willing to accept).
Discount Factors ($\delta_b$ and $\delta_s$): Represents how much each player values immediate transactions over future possibilities, where $\delta_b,\delta_s \in (0, 1)$. Utility associated with future offers are discounted by $\delta_b^(t-1)$ and $\delta_s^(t-1)$ for the buyer and the seller, respectively, where t indicates the current round.
Buyer's Utility: If a price $p$ is agreed upon at time step $t<=T$, then buyer's utility is $u_b=(1-p)*\delta_b^(t-1)$.
Seller's Utility: If a price $p$ is agreed upon at time step $t<=T$, then seller's utility is $u_b=(p-0)*\delta_s^(t-1)$.
Deadline: If no sale is agreed upon by the end of time T, the negotiation fails, and no transaction occurs, in which case, both agents get 0 utility.
Complete Information: All details about the item's value range, the structure of the rounds, and the potential outcomes are common knowledge.

Interaction Protocol:
Decision Turns: Starting with the buyer, players alternate in making price offers. The player making an offer proposes a price within the range from the seller's value to the buyer's value.
Responses: The opponent can either accept the proposed price, resulting in a sale and the game ending, or reject the offer, in which case the negotiation advances to the next round.

Goal of the agents: 
The seller aims to maximize the sale price while the buyer seeks to minimize it. Each agent's goal is to negotiate a price as close as possible to their value (1 for the seller, 0 for the buyer) while considering the risk of no agreement by the deadline.
"""

class State(BaseModel):
    time_step: int
    cur_agent: str
    actions: List[str]|List[NonNegativeFloat]
    action_schema: List[tuple]
    textual_descript: str
    mathematical_descript: List[str|NonNegativeFloat]

    def is_valid_action(self, action):
        if type(action) == str:
            # print("========= str")
            # print(action)
            return action == "accept" or action == "reject"
        elif type(action) == float:
            # print("========= float")
            # print(action)
            return action >= 0 and action <= 1
        else:
            print("Illegal action {}.".format(action))
            return False

class BargainAlternateSingleIssue(BaseEnv):
    def __init__(self, env_param):
        self.name = "bargain_alternate_singleissue"
        self.required_agents = ["seller", "buyer"]
        self.env_param = env_param

        self.T = env_param["T"]
        self.buyerDiscount = env_param["buyerDiscount"]
        self.sellerDiscount = env_param["sellerDiscount"]

        self.proposal = [0.0, 1.0] # agents can make offer in the range of (0,1)
        self.response = ["accept", "reject"]

        self.description_of_problem_class = description_of_problem_class
        # set initial state, seller first proposes a price
        self.reset()

    def get_description(self, agent_role):
        # description = "This is an alternating offer bargaining game under complete information, where a seller and a buyer bargain over the price of a good for T time steps. Let t=1,2,3,...,T denote the time step. If the agents agree on a price of P at time step t<=T, then buyer's utility is (1-P)*E^(t-1) and the seller's utility is (P-0)*F^(t-1), where E denotes buyer's discount and F denotes seller's discount. When the time step t>T, both buyer and seller get utility 0. At time step 1, the buyer proposes an initial price to the seller. Then the seller decides whether to accept or reject it. When the seller rejects the offer, the game enters the next time step, where seller needs to make a counteroffer and buyer needs to decide whether to accept it. This process repeats until the deadline T is reached.\n"
        if agent_role == "buyer":
            description = "This is the beginning of a new game instance, where you will play as the buyer. Your discount factor delta_b={}, seller's discount factor delta_s={}, and the deadline T={}.".format(self.buyerDiscount, self.sellerDiscount, self.T)
        else:
            description = "This is the beginning of a new game instance, where you will play as the seller. Your discount factor delta_s={}, buyer's discount factor delta_b={}, and the deadline T={}.".format(self.sellerDiscount, self.buyerDiscount, self.T)
        return description
    
    def reset(self):
        '''Reset the environment'''
        self.state = State(time_step=1, cur_agent="buyer", actions=self.proposal, action_schema=[("action", "float", "The price that agent proposes, which should be in the range of [0, 1].")], textual_descript="This is time step {}. Now buyer needs to propose a price in the range of {} to the seller.".format(1, self.proposal), mathematical_descript=[])     
        self.is_done = False

    def switch_agent(self, agent):
        if agent == "seller":
            return "buyer"
        elif agent == "buyer":
            return "seller"
        else:
            raise ValueError("Unknown agent {}.".format(agent))

    def step(self, action):
        """
        if cur_agent is proposal a price, then action is a float in [0, 1]
        if cur_agent is responding to a proposal, then action is either "accept" or "reject
        """
        self.state.mathematical_descript.append(action) # keep track of actions taken by the two parties

        if self.state.actions == [0.0, 1.0]:
            action = float(action)
            assert type(action) == float and action >= 0.0 and action <= 1.0
            # action is a proposal of price, now the opponent needs to decide whether to accept it
            proposer = deepcopy(self.state.cur_agent)
            responder = self.switch_agent(proposer)
            self.state.cur_agent = responder
            self.state.actions = self.response
            self.state.action_schema = [("action", "string", "Agent's response to the proposed price, which is either accept or reject.")]
            if self.state.time_step > 1:
                self.state.textual_descript="{} has rejected your offer and the game enters time step {}. {} has proposed a price of {}. Now the {} needs to decide whether to accept it or reject it.".format(proposer, self.state.time_step, proposer, action, responder)
            else:
                self.state.textual_descript="This time step {}, and {} has proposed a price of {}. Now the {} needs to decide whether to accept it or reject it.".format(self.state.time_step, proposer, action, responder)
            self.final_price = None
        elif self.state.actions == ["accept", "reject"]:
            assert action in self.state.actions
            # action is a response to the proposed price
            responder = deepcopy(self.state.cur_agent)
            proposer = self.switch_agent(responder)
            if action == "accept":
                self.is_done = True
                self.final_price = self.state.mathematical_descript[-2]
            else:
                # proposal is rejected
                self.final_price = None
                if self.state.time_step >= self.T:
                    self.is_done = True
                else:
                    # we enter next time step
                    # where responder need to propose a new price
                    self.state.cur_agent = responder
                    self.state.time_step += 1
                    self.state.actions = self.proposal
                    self.state.action_schema = [("action", "float", "The price that agent proposes, which should be in the range of [0, 1].")]
                    self.state.textual_descript="You have rejected {}'s proposed price, and thus the game enters time step {}. Now you need to propose a new price in the range of {} to the {}.".format(proposer, self.state.time_step, self.proposal, proposer)
        else:
            raise ValueError("Something wrong with the action set definition {}.".format(self.state.actions))
        return self.state, None

    def calculate_spe_price_utility(self, cur_time: int, deadline: int, cur_player: str, buyer_discount: float, seller_discount: float, buyer_value: float=1.0, seller_value: float=0.0):
        # decide who the last player that proposes price would be, based on who the current player is
        if (deadline-cur_time) % 2 == 0:
            last_player = deepcopy(cur_player)
        else:
            last_player = self.switch_agent(cur_player)

        if last_player == "buyer":
            # deadline must be odd number, if last player to offer is buyer (since we assume buyer always act first)
            assert deadline % 2 == 1
        elif last_player == "seller":
            assert deadline % 2 == 0
        else:
            raise ValueError("Unknown player type %s" % last_player)

        iter_time = deadline
        if last_player == "buyer":
            iter_player = "buyer"
            # at last time step, the proposing agent can take all share
            buyer_share = 1
            seller_share = 0
        else:
            iter_player = "seller"
            # at last time step, the proposing agent can take all share
            buyer_share = 0
            seller_share = 1

        iter_time -= 1
        while iter_time >= cur_time:
            iter_player = self.switch_agent(iter_player)
            if iter_player == "seller":
                buyer_share *= buyer_discount
                seller_share = 1 - buyer_share
            else:
                seller_share *= seller_discount
                buyer_share = 1 - seller_share
            iter_time -= 1

        # calculate the spe price for the proposer, and the correponding utility
        if cur_player == "buyer":
            utility = (buyer_value-seller_value)*buyer_share*buyer_discount**(cur_time-1)
            price = buyer_value-buyer_share*(buyer_value-seller_value)
        else:
            utility = (buyer_value-seller_value)*seller_share*seller_discount**(cur_time-1)
            price = seller_value+seller_share*(buyer_value-seller_value)

        # return round(price, 2), round(utility, 2)
        return price, utility
    
if __name__ == "__main__":
    env_param = {"T":3, "buyerDiscount":0.7, "sellerDiscount":0.7}
    env = BargainAlternateSingleIssue(env_param=env_param)

    state = env.state
    print(state.textual_descript)
    while not env.is_done: # in case game never ends due to failure in checking terminal condition
        action = input("What is your decision?\n")
        if env.state.actions == [0.0, 1.0]:
            price, util = env.calculate_spe_price_utility(cur_time=env.state.time_step, cur_player=env.state.cur_agent, deadline=env.T, buyer_discount=env.buyerDiscount, seller_discount=env.sellerDiscount)
            print("spe price {} and utility {}.".format(price, util))
        state, _ = env.step(action)
        print(state.textual_descript)

    print("final price {}".format(env.final_price))
    print("The game has ended!")
