from pydantic import BaseModel, Field, field_validator
import numpy as np
from copy import deepcopy

tool_names = ["CalcUtil", "BackwardOneStep", "GetSPEPrice"]

class CalcUtil(BaseModel):
    """
    Calculate utility for all buyers or all sellers
    """
    agent_type: str = Field(
        ...,
        description="""The type of agent: 'buyer' or 'seller'""",
    )
    price: list = Field(
        ...,
        description="""The price""",
    )
    time_step: int = Field(
        ...,
        description="""The time step""",
    )

# class CalcUtil():
#     def __init__(self, agent_type, price, time_step):
#         self.agent_type = agent_type
#         self.price = price
#         self.time_step = time_step

    def execute(self, working_memory):
        required_params = ["buyerDiscounts", "sellerDiscounts", "buyerWeights", "sellerWeights"]
        missing_params = [param for param in required_params if param not in working_memory]
        if missing_params:
            return "Parameters {} missing in the working memory.".format(missing_params)

        if self.agent_type == "buyer":
            values = np.ones(len(working_memory["buyerDiscounts"]))
            discounts = np.array(working_memory["buyerDiscounts"])
            weights = np.array(working_memory["buyerWeights"])
        elif self.agent_type == "seller":
            values = np.zeros(len(working_memory["sellerDiscounts"]))
            discounts = np.array(working_memory["sellerDiscounts"])
            weights = np.array(working_memory["sellerWeights"])

        values_arr = np.array(values)
        self_price_arr = np.array(self.price)
        discounts_arr = np.array(discounts)
        weights_arr = np.array(weights)

        utils = np.abs(values_arr - self_price_arr) * np.power(discounts_arr, self.time_step - 1) * weights_arr

        result = "The utilities that all {}s get for agreeing on price {} at time step {} are: {}".format(
            self.agent_type, self.price, self.time_step, np.round(utils, 4).tolist()
        )
        return result

class BackwardOneStep(BaseModel):
    """
    Compute SPE price using one step of backward induction reasoning for all agents
    """
    agent_type: str = Field(
        ...,
        description="""The current agent type: 'buyer' or 'seller'""",
    )
    opponent_util_if_rej: list = Field(
        ...,
        description="""The utilities that the opponents can get if the game continues to the next time step""",
    )
    time_step: int = Field(
        ...,
        description="""The current time step""",
    )

# class BackwardOneStep():
#     def __init__(self, agent_type, opponent_util_if_rej, time_step):
#         self.agent_type = agent_type
#         self.opponent_util_if_rej = opponent_util_if_rej
#         self.time_step = time_step

    def execute(self, working_memory):
        required_params = ["buyerDiscounts", "sellerDiscounts", "buyerWeights", "sellerWeights", "SPEPrices", "T"]
        missing_params = [param for param in required_params if param not in working_memory]
        if missing_params:
            return "Parameters {} missing in the working memory.".format(missing_params)
        buyer_discounts = working_memory["buyerDiscounts"]
        seller_discounts = working_memory["sellerDiscounts"]
        buyer_weights = working_memory["buyerWeights"]
        seller_weights = working_memory["sellerWeights"]
        T = working_memory["T"]

        buyer_share, buyer_utility, seller_share, seller_utility = self.calculate_spe_price_utility(self.time_step, self.opponent_util_if_rej, self.agent_type, buyer_discounts, seller_discounts, buyer_weights, seller_weights, T)

        if self.agent_type == "buyer":
            util = buyer_utility
            share = buyer_share
        else:
            util = seller_utility
            share = seller_share
        print(f"current agent: {self.agent_type}, buyer_share: {buyer_share}, buyer_utility: {buyer_utility}, seller_share: {seller_share}, seller_utility: {seller_utility}")
        if "SPEPrices" not in working_memory:
            working_memory["SPEPrice"] = {}
        
        working_memory["SPEPrices"][self.time_step] = share
        result = "The SPE prices of {} at time step {} are {} with utility {}".format(
            self.agent_type, self.time_step, np.round(share, 2).tolist(), util
        )
        return result, np.round(share, 2).tolist(), util
    
    def calculate_spe_price_utility(self, cur_time, oppo_util_if_rej, cur_player, buyer_discounts, seller_discounts, buyer_weights, seller_weights, T):
        assert len(buyer_weights) == len(buyer_discounts)
        assert len(seller_weights) == len(seller_discounts)
        assert len(buyer_weights) == len(seller_weights)
        assert type(cur_time) == int

        d = len(buyer_weights)
        for t in range(1, cur_time + 1):
            t = T - t + 1
            # print(f"Time Step: {t}")
            if cur_player == "buyer":
                next_player = "seller"
            else:
                next_player = "buyer"

            # The first round
            if t == T:
                oppo_util_if_rej = 0
                if next_player == "buyer":
                    buyer_next_share = [0 for _ in range(d)]
                    seller_next_share = [1 for _ in range(d)]
                else:
                    buyer_next_share = [1 for _ in range(d)]
                    seller_next_share = [0 for _ in range(d)]
                buyer_next_util, _ = self.calculate_utility(t, buyer_discounts, buyer_weights, buyer_next_share)
                seller_next_util, _ = self.calculate_utility(t, seller_discounts, seller_weights, seller_next_share)

                oppo_util_if_rej = buyer_next_util if (buyer_next_util > 0) else seller_next_util
                
            else:
                if cur_player == "buyer":
                    buyer_next_share, buyer_next_util, seller_next_share, seller_next_util = self.solve_for_share(cur_player, t, oppo_util_if_rej, buyer_weights, seller_weights, buyer_discounts, seller_discounts)
                    oppo_util_if_rej = buyer_next_util

                else:
                    buyer_next_share, buyer_next_util, seller_next_share, seller_next_util = self.solve_for_share(cur_player, t, oppo_util_if_rej, buyer_weights, seller_weights, buyer_discounts, seller_discounts)
                    oppo_util_if_rej = seller_next_util
                    
            # print(f"next player:{next_player}, buyer_next_share: {buyer_next_share}, buyer_next_util: {buyer_next_util}, seller_next_share: {seller_next_share}, seller_next_util: {seller_next_util}, oppo_util_if_rej: {oppo_util_if_rej}")
            print(f"current player:{cur_player}, buyer_next_util: {buyer_next_util}, seller_next_util: {seller_next_util}, oppo_util_if_rej: {oppo_util_if_rej}")
            cur_player = next_player

        # if cur_player == "buyer":
        #     buyer_share, buyer_utility, seller_share, seller_utility = self.solve_for_share("buyer", cur_time, oppo_util_if_rej, buyer_weights, seller_weights, buyer_discounts, seller_discounts)
        # else:
        #     buyer_share, buyer_utility, seller_share, seller_utility = self.solve_for_share("seller", cur_time, oppo_util_if_rej, buyer_weights, seller_weights, buyer_discounts, seller_discounts)

        #    print(f"buyer_share: {buyer_next_share}, buyer_util: {buyer_next_util}, seller_share: {seller_next_share}, seller_util: {seller_next_util}")
        return buyer_next_share, buyer_next_util, seller_next_share, seller_next_util
    
    def solve_for_share(self, cur_player, cur_time, oppo_util_if_rej, buyer_weight, seller_weight, buyer_discount, seller_discount):
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
                    oppo_util_if_rej_temp = 0

            seller_util, _ = self.calculate_utility(cur_time, seller_discount, seller_weight, share_to_opponent)
            buyer_util, _ = self.calculate_utility(cur_time, buyer_discount, buyer_weight, 1.0 - share_to_opponent)
            # assert abs(seller_util - oppo_util_if_rej) <= 1e-3
            buyer_share = 1.0 - share_to_opponent
            seller_share = share_to_opponent

        else:
            share_to_opponent = np.zeros(d).astype(np.float32)
            ratio = np.divide(buyer_weight, seller_weight)
            sorted_descending = np.argsort(-ratio)
            for i in sorted_descending:
                if buyer_weight[i] * buyer_discount[i]**(cur_time-1) < oppo_util_if_rej_temp:
                    share_to_opponent[i] = 1.0
                    oppo_util_if_rej_temp -= buyer_weight[i] * buyer_discount[i]**(cur_time-1)
                else:
                    share_to_opponent[i] = oppo_util_if_rej_temp / (buyer_weight[i] * buyer_discount[i]**(cur_time-1))
                    oppo_util_if_rej_temp = 0

            seller_util, _ = self.calculate_utility(cur_time, seller_discount, seller_weight, 1.0 - share_to_opponent)
            buyer_util, _ = self.calculate_utility(cur_time, buyer_discount, buyer_weight, share_to_opponent)
            # assert abs(buyer_util - oppo_util_if_rej) <= 1e-3
            buyer_share = share_to_opponent
            seller_share = 1.0 - share_to_opponent
        return np.round(buyer_share, 2), buyer_util, np.round(seller_share, 2), seller_util
    
    def calculate_utility(sefl, cur_time, discounts, weights, shares):
        utility = sum(weights[i] * shares[i] * discounts[i]**(cur_time - 1) for i in range(len(weights)))
        utility = round(utility, 2)
        return utility, shares

class GetSPEPrice(BaseModel):
    """
    When making an offer, use this operation to retrieve the SPE prices computed before
    """
    agent_type: str = Field(
        ...,
        description="""The agent type: 'buyer' or 'seller'""",
    )
    time_step: int = Field(
        ...,
        description="""The time step""",
    )

# class GetSPEPrice():
    
#     def __init__(self, agent_type,  time_step):
#         self.agent_type = agent_type
#         self.time_step = time_step

    def execute(self, working_memory):
        required_params = ["SPEPrices"]
        missing_params = [param for param in required_params if param not in working_memory]
        if missing_params:
            return "Parameters {} missing in the working memory.".format(missing_params)
        time = working_memory["T"] + 1 - self.time_step
        if time in working_memory["SPEPrices"]:
            spe_prices = working_memory["SPEPrices"][time]
            if self.time_step % 2 == 0:
                if self.agent_type != "seller":
                    return "Buyer is not the agent that offers a price at time step {}".format(self.time_step)
            else:
                if self.agent_type != "buyer":
                    return "Seller is not the agent that offers a price at time step {}".format(self.time_step)

            return "The SPE prices of all {}s at time step {} are: {}".format(
                self.agent_type, self.time_step, np.round(spe_prices, 4).tolist()
            )
        else:
            return "The SPE prices for time step {} haven't been computed yet.".format(self.time_step)