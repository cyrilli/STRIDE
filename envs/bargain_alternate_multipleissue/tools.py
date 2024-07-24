from pydantic import BaseModel, Field, field_validator
from typing import List
import numpy as np
from copy import deepcopy

tool_names_bargain_complete_info_multi = ["CalcUtil", "BackwardOneStep", "GetSPEPrice"]

class CalcUtil(BaseModel):
    """
    Calculate utility for buyer or all sellers.
    """
    agent: str = Field(
        ...,
        description="""The type of agent: 'buyer' or 'seller'""",
    )
    prices: List[float] = Field(
        ...,
        description="""The prices""",
    )
    time_step: int = Field(
        ...,
        description="""The time step""",
    )

    def execute(self, working_memory):
        required_params = ["buyerDiscounts", "sellerDiscounts", "buyerWeights", "sellerWeights"]
        missing_params = [param for param in required_params if param not in working_memory]
        if missing_params:
            return "Parameters {} missing in the working memory.".format(missing_params)

        if self.agent == "buyer":
            values = np.ones(len(working_memory["buyerDiscounts"]))
            discounts = np.array(working_memory["buyerDiscounts"])
            weights = np.array(working_memory["buyerWeights"])
        elif self.agent == "seller":
            values = np.zeros(len(working_memory["sellerDiscounts"]))
            discounts = np.array(working_memory["sellerDiscounts"])
            weights = np.array(working_memory["sellerWeights"])
        else:
            return "Invalid agent type."

        values_arr = np.array(values)
        self_price_arr = np.array(self.prices)
        discounts_arr = np.array(discounts)
        weights_arr = np.array(weights)

        # Ensure all arrays are the same length
        min_length = min(len(values_arr), len(self_price_arr), len(discounts_arr), len(weights_arr))
        values_arr = values_arr[:min_length]
        self_price_arr = self_price_arr[:min_length]
        discounts_arr = discounts_arr[:min_length]
        weights_arr = weights_arr[:min_length]

        # Calculate utilities
        utils = np.abs(values_arr - self_price_arr) * np.power(discounts_arr, self.time_step - 1) * weights_arr
        total_util = np.round(np.sum(utils), 2)

        result = "The total utility that {} gets for agreeing on prices {} at time step {} is: {}".format(
            self.agent, self.prices, self.time_step, total_util
        )
        return result


class BackwardOneStep(BaseModel):
    """
    compute SPE price using one step of backward induction reasoning based on opponent's utility if rejecting
    """
    agent: str = Field(
        ...,
        description="""the current agent""",
    )
    opponent_util_if_rej: float = Field(
        ...,
        description="""the utility that the opponent can get if the game continues to the next time step""",
    )
    time_step: int = Field(
        ...,
        description="""the current time step""",
    )

    def solve_for_share(self, cur_player, cur_time, oppo_util_if_rej, buyer_weight, seller_weight, buyer_discount, seller_discount):
        assert len(buyer_weight) == len(buyer_discount)
        assert len(seller_weight) == len(seller_discount)
        assert len(buyer_weight) == len(seller_weight)
        assert isinstance(cur_time, int)

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

            buyer_share = 1.0 - share_to_opponent
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

            buyer_share = share_to_opponent

        seller_share = 1.0 - buyer_share
        return np.round(buyer_share, 2), np.round(seller_share, 2)

    def execute(self, working_memory):
        required_params = ["buyerDiscounts", "sellerDiscounts", "buyerWeights", "sellerWeights"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params:
            return f"Parameters {missing_params} missing in the working memory."

        buyer_discount = working_memory["buyerDiscounts"]
        seller_discount = working_memory["sellerDiscounts"]
        buyer_weight = working_memory["buyerWeights"]
        seller_weight = working_memory["sellerWeights"]

        buyer_share, seller_share = self.solve_for_share(
            self.agent, self.time_step, self.opponent_util_if_rej,
            buyer_weight, seller_weight, buyer_discount, seller_discount
        )

        if self.agent == "buyer":
            price = 1.0 - buyer_share
        else:
            price = seller_share

        working_memory["SPEPrice"][self.time_step] = price
        return f"The SPE price of {self.agent} at time step {self.time_step} is {np.round(price, 4)}"


class GetSPEPrice(BaseModel):
    """
    when making an offer, use this operation to retrieve the SPE price computed before
    """
    agent: str = Field(
        ...,
        description="""the agent""",
    )
    time_step: int = Field(
        ...,
        description="""the time step""",
    )

    def execute(self, working_memory):
        required_params = ["SPEPrice"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params:
            return f"Parameters {missing_params} missing in the working memory."
        if self.time_step in working_memory["SPEPrice"]:
            spe_price = working_memory["SPEPrice"][self.time_step]        
            if self.time_step % 2 == 0:
                # even, seller's turn
                if self.agent != "seller":
                    return f"buyer is not the agent that offers a price at time step {self.time_step}"
            else:
                if self.agent != "buyer":
                    return f"seller is not the agent that offers a price at time step {self.time_step}"

            return f"The SPE price of {self.agent} at time step {self.time_step} is {np.round(spe_price, 4)}"
        else:
            return f"The SPE price for time step {self.time_step} hasn't been computed yet."
