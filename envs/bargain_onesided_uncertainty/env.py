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
This is a finite horizon bargaining game with one-sided uncertainty, in which the uninformed bargainer, the seller, makes all the offers and the informed bargainer, the buyer, can only decides to accept or reject the offer.

Components:
Players: Buyer (informed) and Seller (uninformed).
Buyer's Value: b (the maximum price the buyer is willing to pay).
Seller's Value: 0 (the minimum price the seller is willing to accept).
Discount Factors ($\delta_b$ and $\delta_s$): Represents how much each player values immediate transactions over future possibilities, where $\delta_b,\delta_s \in (0, 1)$. Utility associated with future offers are discounted by $\delta_b^(t-1)$ and $\delta_s^(t-1)$ for the buyer and the seller, respectively, where t indicates the current time step.
Buyer's Utility: If a price $p$ is agreed upon at time step $t<=T$, then buyer's utility is $u_b=(b-p)*\delta_b^(t-1)$.
Seller's Utility: If a price $p$ is agreed upon at time step $t<=T$, then seller's utility is $u_b=(p-0)*\delta_s^(t-1)$.
Deadline: If no sale is agreed upon by the end of time T, the negotiation fails, and no transaction occurs, in which case, both agents get 0 utility.
Information Asymmetry: Buyer himself knows the true value of b, which is drawn from a known distribution $F(v)$ supported on $[0, 1]$. We assume $F(v)=v$, i.e., Buyer's value $b$ is sampled from a uniform distribution. The seller does not know $b$ but knows the distribution $F(v)$.

Interaction Protocol:
Decision Turns: In each time step $t=1,2,\dots,T$, it is always Seller who makes an offer $p_t$ within the range of [0,1].
Responses: Buyer can either accept the proposed price, resulting in a sale and the game ending, or reject the offer, in which case the negotiation advances to the next time step.

Goal of the agents: 
Seller's Objective: Maximize their expected payoff over the horizon of the game without knowing the true value of $b$. The seller must strategically decide on the prices $p_t$ to offer in each time step, considering the declining number of opportunities to make a sale and the distribution of $b$ inferred from the buyer's responses.
Buyer's Objective: Maximize their surplus, which is the difference between the true value $b$ and the price paid $p$, if a transaction occurs. The buyer needs to decide whether to accept or reject the seller's offers based on the value $b$ and the likelihood of a more favorable price in subsequent time steps, considering the finite number of time steps.
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

class BargainOneSidedUncertainty(BaseEnv):
    def __init__(self, env_param):
        self.name = "bargain_onesided_uncertainty"
        self.required_agents = ["seller", "buyer"]
        self.env_param = env_param
        self.description_of_problem_class = description_of_problem_class
        self.T = env_param["T"]
        self.buyerVal = env_param["buyerVal"] # a number in [0, 1]
        self.buyerDiscount = env_param["buyerDiscount"]
        self.sellerDiscount = env_param["sellerDiscount"]

        self.proposal = [0.0, 1.0] # seller can make offer in the range of (0,1)
        self.response = ["accept", "reject"]

        # set initial state, seller first proposes a price
        self.reset()

    def get_description(self, agent_role):
        if agent_role == "buyer":
            description = "This is the beginning of a new game instance, where you will play as the buyer. Your discount factor delta_b={}, seller's discount factor delta_s={}, and the deadline T={}. Your value b={}, which is uniformly sampled from [0,1]".format(self.buyerDiscount, self.sellerDiscount, self.T, self.buyerVal)
        else:
            description = "This is the beginning of a new game instance, where you will play as the seller. Your discount factor delta_s={}, buyer's discount factor delta_b={}, and the deadline T={}. The buyer's value b is unknown to you, but you know it is uniformly sampled from [0,1]".format(self.sellerDiscount, self.buyerDiscount, self.T)
        return description

    def get_se_prices(self):
        # compute the constant c for all time steps
        self.c = np.zeros(self.T+1)
        for i in reversed(range(1, self.T+1)):
            if i == self.T:
                self.c[i] = 0.5
            else:
                self.c[i] = (1-self.buyerDiscount+self.buyerDiscount*self.c[i+1])**2 / (2*(1-self.buyerDiscount+self.buyerDiscount*self.c[i+1])-self.sellerDiscount*self.c[i+1])
        # compute b for all time steps
        self.b = np.zeros(self.T)
        self.b[0] = 1.0 # initial belief
        for t in range(1, self.T):
            self.b[t] = (1-self.buyerDiscount+self.buyerDiscount*self.c[t+1]) / (2*(1-self.buyerDiscount+self.buyerDiscount*self.c[t+1])-self.sellerDiscount*self.c[t+1]) * self.b[t-1]
        # compute prices
        self.p = np.zeros(self.T+1)
        for t in range(1,self.T+1):
            self.p[t] = self.c[t] * self.b[t-1]
        return self.p

    def reset(self):
        '''Reset the environment'''
        self.state = State(time_step=1, cur_agent="seller", actions=self.proposal, action_schema=[("action", "float", "The price that agent proposes, which should be in the range of [0, 1].")], textual_descript="This is time step {}. Now seller needs to propose a price in the range of {} to the buyer.".format(1, self.proposal), mathematical_descript=[])     
        self.is_done = False

    def step(self, action):
        """
        if cur_agent is seller, then action is a float in [0, 1]
        if cur_agent is buyer, then action is either "accept" or "reject
        """
        self.state.mathematical_descript.append(action) # keep track of actions taken by the two parties

        if self.state.cur_agent == "buyer":
            # let's see if buyer has accepted
            if action == "accept":
                # buyer has accepted seller's offer or deadline is reached
                self.is_done = True
                self.final_price = self.state.mathematical_descript[-2]
            else:
                # buyer has rejected seller's offer
                self.final_price = None
                if self.state.time_step >= self.T:
                    self.is_done = True
                else:
                    # we enter next time step
                    # where seller need to make a new offer
                    self.state.time_step += 1
                    self.state.cur_agent = "seller"
                    self.state.actions = self.proposal
                    self.state.action_schema = [("action", "float", "The price that agent proposes, which should be in the range of [0, 1].")]
                    self.state.textual_descript="The buyer has rejected seller's offer at time step {}, and thus the game enters time step {}. Seller needs to propose a price in the range of {} to the buyer.".format(self.state.time_step-1, self.state.time_step, self.proposal)
        else:
            # seller has offered a price
            self.state.cur_agent = "buyer"
            self.final_price = None
            self.state.actions = self.response
            self.state.action_schema = [("action", "string", "Agent's response to the proposed price, which is either accept or reject.")]
            self.state.textual_descript="This is time step {}. Seller has offered a price of {} to the buyer. Now the buyer needs to decide whether to accept it or reject it".format(self.state.time_step, self.state.mathematical_descript[-1])
        return self.state, None

if __name__ == "__main__":
    env_param = {"T":3, "buyerVal": 0.4, "buyerDiscount":1.0, "sellerDiscount":1.0}
    env = BargainOneSidedUncertainty(env_param=env_param)

    print("sequential equilibrium price is {}".format(env.get_se_prices()))
    state = env.state
    print(state.textual_descript)
    while not env.is_done: # in case game never ends due to failure in checking terminal condition
        if state.cur_agent == "seller":
            action = input("As seller, what's your offer to buyer?\n")
        else:
            action = input("As buyer, do you accept seller's offer?\n")
        state, _ = env.step(action)
        print(state.textual_descript)

    print("final price {}".format(env.final_price))
    print("The game has ended!")
