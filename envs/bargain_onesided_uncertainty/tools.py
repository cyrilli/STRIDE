from pydantic import BaseModel, Field, field_validator
import numpy as np
from copy import deepcopy
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)


tool_names_bargain_incomplete_info_onsided = ["CalcUtil", "ComputeBt", "SolveLast", "Solve", "GetSEPrice"]

class ComputeBt(BaseModel):
    """
    this operation is used in bisection search, to compute what seller's belief about buyer's value would be at the current time step, given seller's belief at time step T-1
    """
    time_step : int = Field(
        ...,
        description="""the current time step""",
    )
    b_last: float = Field(
        ...,
        description = """seller's belief about buyer's value after observing buyer's response at time step T-1"""
    )

    def execute(self, working_memory):
        required_params = ["delta_b", "delta_s", "c", "T"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)

        index_cur_b = self.time_step-1
        t = working_memory["T"]-1
        b = self.b_last
        working_memory["b_list"] = {}
        working_memory["b_list"][t] = deepcopy(b)
        while t >= (index_cur_b+1):
            b = (2*(1-working_memory["delta_b"]+working_memory["delta_b"]*working_memory["c"][t+1])-working_memory["delta_s"]*working_memory["c"][t+1]) / (1-working_memory["delta_b"]+working_memory["delta_b"]*working_memory["c"][t+1]) * b
            t = t-1
            working_memory["b_list"][t] = deepcopy(b)
        return "If seller's belief at time step T-1 is {}, then the current belief should be {}".format(self.b_last, b)
    
class SolveLast(BaseModel):
    """
    compute seller's expected utility and the corresponding price at the last time step
    """
    b_last: float = Field(
        ...,
        description = """seller's belief about buyer's value after observing buyer's response at time step T-1"""
    )

    def execute(self, working_memory):
        required_params = ["delta_s", "T"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)

        # see page 11 of Cramton (1984)        
        p_t = 0.5 * self.b_last
        u_t = 0.25 * self.b_last**2 / self.b_last # undiscounted utility
        
        working_memory["b_t_minus_1"] = deepcopy(self.b_last)
        working_memory["SEPrice"][working_memory["T"]] = p_t
        return "At the last time step {}, seller's expected utility is {} and the corresponding price is {}".format(working_memory["T"], u_t*working_memory["delta_s"]**(working_memory["T"]-1), p_t)

class Solve(BaseModel):
    """
    compute the expected utility and the corresponding price at the current time step, based on the results computed for the next time step.
    """
    u_t : float = Field(
        ...,
        description = """seller's expected utility at the next time step"""
    )
    p_t : float = Field(
        ...,
        description="""seller's price at the next time step""",
    )
    t : int = Field(
        ...,
        description="""index for the time step we are reasoning on""",
    )

    def execute(self, working_memory):
        required_params = ["delta_b", "delta_s", "c", "b_t_minus_1"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        p_t = deepcopy(self.p_t)
        u_t = deepcopy(self.u_t)
        t = deepcopy(self.t)

        b_t = working_memory["b_t_minus_1"]
        # working_memory["b_t_minus_1_to_be"] = deepcopy(b_t)
        p_t = (1-working_memory["delta_b"])*b_t + working_memory["delta_b"] * p_t
        b_t_minus_1 = (2*(1-working_memory["delta_b"]+working_memory["delta_b"]*working_memory["c"][t+1])-working_memory["delta_s"]*working_memory["c"][t+1]) / (1-working_memory["delta_b"]+working_memory["delta_b"]*working_memory["c"][t+1]) * b_t
        u_t = (b_t_minus_1 - b_t)/(b_t_minus_1) * p_t + b_t/(b_t_minus_1) * working_memory["delta_s"] * u_t

        working_memory["b_t_minus_1"] = b_t_minus_1
        working_memory["SEPrice"][self.t] = p_t
        return "At time step {}, seller's expected utility is {} and the corresponding price is {}".format(t, u_t*working_memory["delta_s"]**(t-1), p_t)

class GetSEPrice(BaseModel):
    """
    when making an offer, use this operation to retrieve the SE price computed before
    """
    time_step: int = Field(
        ...,
        description="""the time step""",
    )

    def execute(self, working_memory):
        required_params = ["SEPrice"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        if self.time_step in working_memory["SEPrice"]:
            se_price = working_memory["SEPrice"][self.time_step]        
            return "The SE price at time step {} is {}".format(self.time_step, round(se_price, 4))
        else:
            return "The SE price for time step {} hasn't been computed yet.".format(self.time_step)