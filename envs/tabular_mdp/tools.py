from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import numpy as np

tool_names_mdp_known = ["UpdateQbyR", "UpdateQbyPV", "UpdateV", "GetQ", "GetArgMax"]
tool_names_mdp_unknown = ["UpdateQbyR", "UpdateQbyPV", "UpdateV", "GetQ", "GetArgMax", "UpdateQbyBonus", "UpdateMDPModel"]

class UpdateQbyR(BaseModel):
    """
    add immediate rewards to the Q values for all state-action pairs at current time step
    """
    time_step: int = Field(
        ...,
        description="""current time step during value iteration""",
    )

    def execute(self, working_memory):
        required_params = ["Q", "R", "nState", "nAction"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        for s in range(working_memory["nState"]):
            for a in range(working_memory["nAction"]):
                working_memory["Q"][self.time_step, s, a] += working_memory["R"][s, a][0]

        return "Q values for time step {} are updated with the immediate rewards and stored in the working memory.".format(self.time_step)

class UpdateQbyPV(BaseModel):
    """
    add the expected long-term rewards starting from the next time step to the Q values for all state-action pairs at current time step
    """
    time_step: int = Field(
        ...,
        description="""current time step during value iteration""",
    )

    def execute(self, working_memory):
        required_params = ["Q", "V", "P", "nState", "nAction"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        for s in range(working_memory["nState"]):
            for a in range(working_memory["nAction"]):
                temp = 0.0
                for s_prime in range(working_memory["nState"]):
                    temp += working_memory["P"][s, a, s_prime] * working_memory["V"][self.time_step+1, s_prime]
                working_memory["Q"][self.time_step, s, a] += temp

        return "Q values for time step {} are updated with the one-step look ahead and stored in the working memory.".format(self.time_step)

class UpdateV(BaseModel):
    """
    update the V values based on the computed Q values for the current time step
    """
    time_step: int = Field(
        ...,
        description="""current time step during value iteration""",
    )

    def execute(self, working_memory):
        required_params = ["Q", "V", "nState"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        for s in range(working_memory["nState"]):
            working_memory["V"][self.time_step, s] = np.max(working_memory["Q"][self.time_step, s])

        return "V values for time step {} are updated based on the computed Q values and stored in the working memory.".format(self.time_step)

class GetQ(BaseModel):
    """
    retrieve Q values for all actions at the current state and time step
    """
    time_step: int = Field(
        ...,
        description="""current time step""",
    )
    state: int = Field(
        ...,
        description="""current state""",
    )
    
    def execute(self, working_memory):
        required_params = ["Q"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        q_vals = working_memory["Q"][self.time_step, self.state]

        return np.round(q_vals, decimals=4)

class GetArgMax(BaseModel):
    """
    return the indices corresponding to the maximal value in the given list of numbers
    """
    number_list: List[float] = Field(
        ...,
        description="""the list of numbers""",
    )
    
    def execute(self, working_memory):
        required_params = []
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        number_list = np.array(self.number_list)        
        return np.where(number_list == np.max(number_list))[0]

class UpdateQbyBonus(BaseModel):
    """
    add exploration bonus to the Q values for all state-action pairs at current time step
    """
    time_step: int = Field(
        ...,
        description="""current time step during value iteration""",
    )

    def execute(self, working_memory):
        required_params = ["Nsa", "Q", "nState", "nAction", "bonus_scale_factor", "epLen", "epNum"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        # c = np.sqrt(2)

        for s in range(working_memory["nState"]):
            for a in range(working_memory["nAction"]):
                working_memory["Q"][self.time_step, s, a] += working_memory["bonus_scale_factor"] * working_memory["epLen"] * np.sqrt(np.log(working_memory["nState"]*working_memory["nAction"]*working_memory["epLen"]*working_memory["epNum"]) / working_memory["Nsa"][s, a])

        return "Q values for time step {} are updated with the exploration bonus and stored in the working memory.".format(self.time_step)

class UpdateMDPModel(BaseModel):
    """
    update the estimation of the reward and transition function of MDP based on the observed quadruple (old state, action, new state, reward)
    """
    s: int = Field(
        ...,
        description="""the old state""",
    )
    a: int = Field(
        ...,
        description="""the action taken at the old state""",
    )
    s_prime: int = Field(
        ...,
        description="""the new state environment has transit to""",
    )
    r: float = Field(
        ...,
        description="""the reward received by the agent""",
    )


    def execute(self, working_memory):
        required_params = ["Nsa", "P", "R"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)

        working_memory["Nsa"][self.s, self.a] += 1

        nn = working_memory["Nsa"][self.s, self.a]
        prev_r = working_memory["R"][self.s, self.a, 0]
        prev_p = working_memory["P"][self.s, self.a, :]

        working_memory["R"][self.s, self.a] = (1.0-1.0/nn)*prev_r + self.r*1.0/nn

        working_memory["P"][self.s, self.a, :] = (1.0-1.0/nn)*prev_p
        working_memory["P"][self.s, self.a, self.s_prime] += 1.0/nn

        return "Estimation of the transition function P and reward function R have been updated and stored in working memory."