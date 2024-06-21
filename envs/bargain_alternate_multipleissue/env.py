import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

import numpy as np
from copy import deepcopy

class State():
    def __init__(self, time_step, cur_agent, actions, action_schema, textual_descript, mathematical_descript):
        self.time_step = time_step
        self.cur_agent = cur_agent
        self.actions = actions
        self.textual_descript = textual_descript
        self.mathematical_descript = mathematical_descript
        self.action_schema = action_schema

    def is_valid_action(self, action):
        if isinstance(action, list) and all(isinstance(a, float) for a in action):
            return all(0 <= a <= 1 for a in action)
        elif isinstance(action, str):
            return action == "accept" or action == "reject"
        else:
            print(f"Illegal action {action}.")
            return False


class BargainAlternateMultiIssue():
    def __init__(self, env_param):
        self.name = "bargain_alternate_multiissue"
        self.required_agents = ["seller", "buyer"]
        self.env_param = env_param

        self.T = env_param["T"]
        self.buyerDiscount = env_param["buyerDiscount"]
        self.sellerDiscount = env_param["sellerDiscount"]
        self.buyerWeight = env_param["buyerWeight"]
        self.sellerWeight = env_param["sellerWeight"]

        self.proposal = [0.0, 1.0]
        self.response = ["accept", "reject"]

        self.reset()

    def get_description(self, agent_role):
        if (agent_role == "buyer"):
            description = (
                "This is the beginning of a new game instance, where you will play as the buyer. "
                f"Your discount factors delta_b={self.buyerDiscount}, seller's discount factors delta_s={self.sellerDiscount}, "
                f"weights={self.buyerWeight}, seller's weights={self.sellerWeight}, and the deadline T={self.T}."
            )
        else:
            description = (
                "This is the beginning of a new game instance, where you will play as the seller. "
                f"Your discount factors delta_s={self.sellerDiscount}, buyer's discount factors delta_b={self.buyerDiscount}, "
                f"weights={self.sellerWeight}, buyer's weights={self.buyerWeight}, and the deadline T={self.T}."
            )
        return description
    
    def reset(self):
        self.state = State(
            time_step=1,
            cur_agent="buyer",
            actions=self.proposal,
            action_schema=[("action", "list of floats", "The prices that agent proposes for each issue, which should be in the range of [0, 1].")],
            textual_descript="This is time step {}. Now buyer needs to propose prices for each issue in the range of [0, 1] to the seller.".format(1),
            mathematical_descript=[]
        )
        self.is_done = False

    def switch_agent(self, agent):
        return "buyer" if agent == "seller" else "seller"

    def step(self, action):
        self.state.mathematical_descript.append(action)  # Keep track of actions taken by the two parties
        # propose price
        if any(isinstance(a, str) for a in action):
            pass
        else:
            action = action.astype(float)
        if all(isinstance(a, float) for a in action):
            action = [float(a) for a in action]
            assert all(0.0 <= a <= 1.0 for a in action)
            proposer = deepcopy(self.state.cur_agent)
            responder = self.switch_agent(proposer)
            self.state.cur_agent = responder
            self.state.actions = self.response
            self.state.action_schema = [("action", "string", "Agent's response to the proposed prices, which is either accept or reject.")]
            if self.state.time_step > 1:
                self.state.textual_descript = "{} has rejected your offer and the game enters time step {}. {} has proposed prices of {}. Now the {} needs to decide whether to accept them or reject them.".format(proposer, self.state.time_step, proposer, action, responder)
            else:
                self.state.textual_descript = "This time step {}, and {} has proposed prices of {}. Now the {} needs to decide whether to accept them or reject them.".format(self.state.time_step, proposer, action, responder)
            self.final_price = None
        # accept or reject
        elif action in self.state.actions:
            responder = deepcopy(self.state.cur_agent)
            proposer = self.switch_agent(responder)
            if action == "accept":
                self.is_done = True
                self.final_price = self.state.mathematical_descript[-2]
            else:
                self.final_price = None
                if self.state.time_step >= self.T:
                    self.is_done = True
                else:
                    self.state.cur_agent = responder
                    self.state.time_step += 1
                    self.state.actions = self.proposal
                    self.state.action_schema = [("action", "list of floats", "The prices that agent proposes for each issue, which should be in the range of [0, 1].")]
                    self.state.textual_descript = "You have rejected {}'s proposed prices, and thus the game enters time step {}. Now you need to propose new prices in the range of [0, 1] to the {}.".format(proposer, self.state.time_step, proposer)
        else:
            raise ValueError(f"Something wrong with the action set definition {self.state.actions}.")
        return self.state, None

    def calculate_utility(self, cur_time, discounts, weights, shares):
        utility = sum(weights[i] * shares[i] * discounts[i]**(cur_time - 1) for i in range(len(weights)))
        return utility, None

    def solve_for_share(self, cur_player: str, cur_time: int, oppo_util_if_rej: float, buyer_weight: np.array, seller_weight: np.array, buyer_discount: np.array, seller_discount: np.array):
        assert len(buyer_weight) == len(buyer_discount)
        assert len(seller_weight) == len(seller_discount)
        assert len(buyer_weight) == len(seller_weight)
        assert type(cur_time) == int

        d = len(buyer_weight)
        oppo_util_if_rej_temp = deepcopy(oppo_util_if_rej)
        
        if cur_player == "buyer":
            share_to_opponent = np.zeros(d).astype(np.float32)
            ratio = np.divide(seller_weight, buyer_weight)
            sorted_descending = np.argsort(-ratio)
            for i in sorted_descending:
                if seller_weight[i] * seller_discount[i]**(cur_time-1) < oppo_util_if_rej_temp:
                    share_to_opponent[i] = 1.0
                    oppo_util_if_rej_temp -= seller_weight[i] * seller_discount[i]**(cur_time-1)
                else:
                    share_to_opponent[i] = oppo_util_if_rej_temp / (seller_weight[i] * seller_discount[i]**(cur_time-1))
                    assert share_to_opponent[i] <= 1
                    assert share_to_opponent[i] >= 0
                    oppo_util_if_rej_temp = 0

            # print(f"current player: {cur_player}")
            # print(f"current share_to_opponent: {share_to_opponent}")
            # print(f"current oppo_util_if_rej_temp: {oppo_util_if_rej_temp}")

            seller_util, _ = self.calculate_utility(cur_time, seller_discount, seller_weight, share_to_opponent)
            buyer_util, _ = self.calculate_utility(cur_time, buyer_discount, buyer_weight, 1.0 - share_to_opponent)
            assert abs(seller_util - oppo_util_if_rej) <= 1e-3
            return 1.0 - share_to_opponent, buyer_util, share_to_opponent, oppo_util_if_rej
        else:
            share_to_opponent = np.zeros(d).astype(np.float32)
            ratio = np.divide(buyer_weight, seller_weight)
            sorted_descending = np.argsort(-ratio)
            for i in sorted_descending:
                # print(f"current utils:{buyer_weight[i] * buyer_discount[i]**(cur_time-1)}, oppo_util_if_rej_temp:{oppo_util_if_rej_temp}")
                if buyer_weight[i] * buyer_discount[i]**(cur_time-1) < oppo_util_if_rej_temp:
                    share_to_opponent[i] = 1.0
                    oppo_util_if_rej_temp -= buyer_weight[i] * buyer_discount[i]**(cur_time-1)
                else:
                    # print(f"oppo_util_if_rej_temp left: {oppo_util_if_rej_temp}")
                    share_to_opponent[i] = oppo_util_if_rej_temp / (buyer_weight[i] * buyer_discount[i]**(cur_time-1))
                    assert share_to_opponent[i] <= 1
                    assert share_to_opponent[i] >= 0
                    oppo_util_if_rej_temp = 0
            # print(f"current player: {cur_player}")
            # print(f"current share_to_opponent: {share_to_opponent}")
            # print(f"current oppo_util_if_rej_temp: {oppo_util_if_rej_temp}")
            
            seller_util, _ = self.calculate_utility(cur_time, seller_discount, seller_weight, 1.0 - share_to_opponent)
            buyer_util, _ = self.calculate_utility(cur_time, buyer_discount, buyer_weight, share_to_opponent)
            # print(f"abs value: {abs(buyer_util - oppo_util_if_rej)}")
            # assert abs(buyer_util - oppo_util_if_rej) <= 1e-3
            return share_to_opponent, oppo_util_if_rej, 1.0 - share_to_opponent, seller_util

    def calculate_spe_price_utility(self, cur_time, oppo_util_if_rej, cur_player, buyer_discounts, seller_discounts, buyer_weights, seller_weights):
        assert len(buyer_weights) == len(buyer_discounts)
        assert len(seller_weights) == len(seller_discounts)
        assert len(buyer_weights) == len(seller_weights)
        assert type(cur_time) == int

        d = len(buyer_weights)

        for t in range(cur_time, env.T+1): 
            t = env.T + 1 - t
            # print(f"time_step: {t}")
            if cur_player == "buyer":
                next_player = "seller"
            else:
                next_player = "buyer"

            if next_player == "buyer":
                next_share, next_util, _, _ = self.solve_for_share("buyer", t, oppo_util_if_rej, buyer_weights, seller_weights, buyer_discounts, seller_discounts)
            else:
                _, _, next_share, next_util = self.solve_for_share("seller", t, oppo_util_if_rej, buyer_weights, seller_weights, buyer_discounts, seller_discounts)

            oppo_util_if_rej = next_util
            # print(f"oppo_util_if_rej: {oppo_util_if_rej}")

        if cur_player == "buyer":
            price_list, utility_list, _, _ = self.solve_for_share("buyer", cur_time, oppo_util_if_rej, buyer_weights, seller_weights, buyer_discounts, seller_discounts)
        else:
            _, _, price_list, utility_list = self.solve_for_share("seller", cur_time, oppo_util_if_rej, buyer_weights, seller_weights, buyer_discounts, seller_discounts)

        total_buyer_weight = sum(buyer_weights)
        buyer_share = [w / total_buyer_weight for w in buyer_weights]

        total_price = sum(price_list[i] * buyer_share[i] for i in range(d))
        total_utility = sum(utility_list * buyer_share[i] for i in range(d))
        total_price_opponent = 1 - total_utility
        
        return total_price, total_utility, price_list, total_price_opponent 
   

if __name__ == "__main__":
    env_param = {
        "T": 5,
        "buyerDiscount": [0.76, 0.56, 0.87, 0.13, 0.47],
        "sellerDiscount": [1.,   0.28, 0.78, 0.47, 0.05],
        "buyerWeight": [0.81, 0.81, 0.17, 0.5, 0.59],
        "sellerWeight": [0.7, 0.92, 0.44, 0.4,  0.3 ]
    }
    env = BargainAlternateMultiIssue(env_param=env_param)

    state = env.state
    print(state.textual_descript)

    price, util, price_list, _ = env.calculate_spe_price_utility(
                cur_time=env.state.time_step,
                oppo_util_if_rej=0,  
                cur_player=env.state.cur_agent,
                buyer_discounts=env.buyerDiscount,
                seller_discounts=env.sellerDiscount,
                buyer_weights=env.buyerWeight,
                seller_weights=env.sellerWeight
            )
    print("SPE price: {} and utility: {} with share: {}.".format(price, util, price_list))
