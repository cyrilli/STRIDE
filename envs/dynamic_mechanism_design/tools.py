from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import numpy as np

tool_names_dynamic_vcg = ["UpdateQbyRExcluding", "UpdateQbyPVExcluding", "UpdateVExcluding", "GetQExcluding", "GetArgMax", "EvaluatePolicyExcluding", "GetMax"]

class UpdateQbyRExcluding(BaseModel):
    """
    add immediate rewards, excluding the reward of agent_to_exclude, to the Q values for all state-action pairs at current time step. If agent_to_exclude is set to None, all agents's rewards are considered.
    """
    time_step: int = Field(
        ...,
        description="""current time step during value iteration""",
    )
    agent_to_exclude: Optional[int] = Field(
        ...,
        description="""the agent whose reward will not be considered during value iteration""",
    )

    def execute(self, working_memory):
        
        if self.agent_to_exclude is not None:
            required_params = ["QExcluding", "R", "nState", "nAction"]
            missing_params = []
            for required_param in required_params:
                if required_param not in working_memory:
                    missing_params.append(required_param)
            if missing_params != []:
                return "Parameters {} missing in the working memory.".format(missing_params)

            for s in range(working_memory["nState"]):
                for a in range(working_memory["nAction"]):
                    working_memory["QExcluding"][self.agent_to_exclude, self.time_step, s, a] += np.sum(working_memory["R"][:self.agent_to_exclude, s, a, 0])
                    working_memory["QExcluding"][self.agent_to_exclude, self.time_step, s, a] += np.sum(working_memory["R"][self.agent_to_exclude+1:, s, a, 0])

            return "Q values for time step {} are updated with the immediate rewards excluding agent {} and stored in the working memory.".format(self.time_step, self.agent_to_exclude)

        else:
            required_params = ["Q", "R", "nState", "nAction"]
            missing_params = []
            for required_param in required_params:
                if required_param not in working_memory:
                    missing_params.append(required_param)
            if missing_params != []:
                return "Parameters {} missing in the working memory.".format(missing_params)
            
            for s in range(working_memory["nState"]):
                for a in range(working_memory["nAction"]):
                    working_memory["Q"][self.time_step, s, a] += np.sum(working_memory["R"][:, s, a, 0])

            return "Q values for time step {} are updated using all agents' immediate rewards and stored in the working memory.".format(self.time_step)

class UpdateQbyPVExcluding(BaseModel):
    """
    add the expected long-term rewards, excluding the reward of agent_to_exclude, to the Q values for all state-action pairs at current time step. If agent_to_exclude is set to None, all agents's rewards are considered.
    """
    time_step: int = Field(
        ...,
        description="""current time step during value iteration""",
    )
    agent_to_exclude: Optional[int] = Field(
        ...,
        description="""the agent whose reward will not be considered during value iteration""",
    )

    def execute(self, working_memory):
        if self.agent_to_exclude is None:
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

        else:
            required_params = ["QExcluding", "VExcluding", "P", "nState", "nAction"]
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
                        temp += working_memory["P"][s, a, s_prime] * working_memory["VExcluding"][self.agent_to_exclude, self.time_step+1, s_prime]
                    working_memory["QExcluding"][self.agent_to_exclude, self.time_step, s, a] += temp

        return "Q values for time step {} are updated with the one-step look ahead and stored in the working memory.".format(self.time_step)

class UpdateVExcluding(BaseModel):
    """
    update the V values, excluding the reward of agent_to_exclude, based on the computed Q values for the current time step. If agent_to_exclude is set to None, all agents's rewards are considered.
    """
    time_step: int = Field(
        ...,
        description="""current time step during value iteration""",
    )
    agent_to_exclude: Optional[int] = Field(
        ...,
        description="""the agent whose reward will not be considered during value iteration""",
    )

    def execute(self, working_memory):
        if self.agent_to_exclude is None:
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

        else:
            required_params = ["QExcluding", "VExcluding", "nState"]
            missing_params = []
            for required_param in required_params:
                if required_param not in working_memory:
                    missing_params.append(required_param)
            if missing_params != []:
                return "Parameters {} missing in the working memory.".format(missing_params)
            
            for s in range(working_memory["nState"]):
                working_memory["VExcluding"][self.agent_to_exclude, self.time_step, s] = np.max(working_memory["QExcluding"][self.agent_to_exclude, self.time_step, s])

            return "V values for time step {} are updated based on the computed Q values and stored in the working memory.".format(self.time_step)

class GetQExcluding(BaseModel):
    """
    retrieve Q values, that excludes the rewards of agent_to_exclude, for all actions at the current state and time step. If agent_to_exclude is set to None, the Q values computed using all agents's rewards will be returned.
    """
    time_step: int = Field(
        ...,
        description="""current time step""",
    )
    state: int = Field(
        ...,
        description="""current state""",
    )
    agent_to_exclude: Optional[int] = Field(
        ...,
        description="""the agent whose reward is not considered""",
    )
    def execute(self, working_memory):
        if self.agent_to_exclude is not None:
            required_params = ["QExcluding"]
            missing_params = []
            for required_param in required_params:
                if required_param not in working_memory:
                    missing_params.append(required_param)
            if missing_params != []:
                return "Parameters {} missing in the working memory.".format(missing_params)
            
            q_vals = working_memory["QExcluding"][self.agent_to_exclude, self.time_step, self.state]

            return np.round(q_vals, decimals=4)

        else:
            required_params = ["Q"]
            missing_params = []
            for required_param in required_params:
                if required_param not in working_memory:
                    missing_params.append(required_param)
            if missing_params != []:
                return "Parameters {} missing in the working memory.".format(missing_params)
            
            q_vals = working_memory["Q"][self.time_step, self.state]

            return np.round(q_vals, decimals=4)

class GetMax(BaseModel):
    """
    return the maximal value in the given list of numbers
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
        return np.max(number_list)

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

class EvaluatePolicyExcluding(BaseModel):
    """
    evaluate the optimal policy on an imagined MDP that excludes the reward function of agent_to_exclude.
    """
    agent_to_exclude: int = Field(
        ...,
        description="""the agent whose reward will not be considered during value iteration""",
    )

    def execute(self, working_memory):
        required_params = ["Q", "P", "R", "nState", "nAction", "epLen"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)

        p_q = working_memory["Q"]
        v = np.zeros((working_memory["epLen"]+1,working_memory["nState"]))

        for time in reversed(range(working_memory["epLen"])):
            for s in range(working_memory["nState"]):
                # action taken by the current policy
                a = np.random.choice(np.flatnonzero(p_q[time, s] == p_q[time, s].max()))
                
                temp = 0.0
                for s_prime in range(working_memory["nState"]):
                    temp += working_memory["P"][s, a, s_prime] * v[time+1, s_prime]
                v[time, s] = np.sum(working_memory["R"][:self.agent_to_exclude, s, a, 0]) + np.sum(working_memory["R"][self.agent_to_exclude+1:, s, a, 0]) + temp
        return "The value of the optimal policy on the MDP that excludes agent {}'s reward is {}".format(self.agent_to_exclude, v[0,0])