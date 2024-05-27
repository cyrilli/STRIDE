from pydantic import BaseModel, Field, field_validator
import numpy as np

tool_names_bargain_complete_info_single = ["CalcUtil", "BackwardOneStep", "GetSPEPrice"]

class CalcUtil(BaseModel):
    """
    caculate buyer or seller's utility
    """
    agent: str = Field(
        ...,
        description="""the agent that we are computing utility for""",
    )
    price: float = Field(
        ...,
        description="""the price""",
    )
    time_step: int = Field(
        ...,
        description="""the time step""",
    )

    def execute(self, working_memory):
        required_params = ["delta_b", "delta_s"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        if self.agent == "buyer":
            if "b_value" in working_memory:
                # for incomplete information setting
                value = working_memory["b_value"]
            else:
                value = 1.0
            discount = working_memory["delta_b"]
            util = (value-self.price) * discount**(self.time_step-1)
        elif self.agent == "seller":
            value = 0.0
            discount = working_memory["delta_s"]
            util = (self.price-value) * discount**(self.time_step-1)

        return "The utility that {} gets for agreeing on price {} at time step {} is {}".format(self.agent, self.price, self.time_step, round(util, 4))
        # return round(util, 4)
        # return util

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

    def execute(self, working_memory):
        required_params = ["delta_b", "delta_s"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        if self.agent == "buyer":
            # opponent is seller
            opponent = "seller"
            oppo_discount = working_memory["delta_s"]
            oppo_share_if_rej = self.opponent_util_if_rej / (oppo_discount**self.time_step)
        else:
            # opponent is buyer
            opponent = "buyer"
            oppo_discount = working_memory["delta_b"]
            oppo_share_if_rej = self.opponent_util_if_rej / (oppo_discount**self.time_step)

        # solve for opponent's share at cur_time to incentive him to accept
        my_share_now = 1.0 - oppo_share_if_rej * oppo_discount
        # calculate the price corresponding to my_share_now
        if self.agent == "buyer":
            price = 1.0 - my_share_now
        else:
            price = my_share_now
        working_memory["SPEPrice"][self.time_step] = price
        # return "To maximize {}'s utility at time step {}, while ensuring a utility no less than {} for {}, {} should offer price {}. This is {}'s subgame perfect equilibrium strategy at time step {}, and is stored in working memory for future use.".format(self.agent, self.time_step, self.opponent_util_if_rej, opponent, self.agent, price, self.agent, self.time_step)
        return "The SPE price of {} at time step {} is {}".format(self.agent, self.time_step, round(price, 4))
        # return round(price, 4)
        # return price

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
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        if self.time_step in working_memory["SPEPrice"]:
            spe_price = working_memory["SPEPrice"][self.time_step]        
            if self.time_step % 2 == 0:
                # even, seller's turn
                if self.agent != "seller":
                    return "buyer is not the agent that offers a price at time step {}".format(self.time_step)
            else:
                if self.agent != "buyer":
                    return "seller is not the agent that offers a price at time step {}".format(self.time_step)

            # return "At time step {}, agent {} will propose a price of {} to maximize his own utility.".format(self.time_step, self.agent, spe_price)
            # return spe_price
            return "The SPE price of {} at time step {} is {}".format(self.agent, self.time_step, round(spe_price, 4))
        else:
            return "The SPE price for time step {} hasn't been computed yet.".format(self.time_step)