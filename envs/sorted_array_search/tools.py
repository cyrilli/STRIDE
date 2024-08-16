from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import numpy as np


tool_names_sorted_array_search = ["sort_array","init_leftright_index","findmidindex","find_leftright_index","CheckIndexVal","Check_left_right"]

class sort_array(BaseModel):
    """
    initialize the left and right index 
    """

    def execute(self, working_memory):
        required_params = ["A"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        working_memory["A"] = sorted(working_memory["A"])

        return "The array is now sorted!"

class init_leftright_index(BaseModel):
    """
    initialize the left and right index 
    """

    def execute(self, working_memory):
        required_params = ["left", "right", "n"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        working_memory["left"] = 0
        working_memory["right"] = working_memory["n"] - 1

        return "The left index is {} and the right index is {}.".format( working_memory["left"],working_memory["right"])


class findmidindex(BaseModel):
    """
    find the middle index between the left and right index
    """

    def execute(self, working_memory):
        required_params = ["left", "right"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        working_memory["mid"] = working_memory["left"] + round((working_memory["right"] - working_memory["left"])/2)

        return "The middle index for the given period is {}.".format(working_memory["mid"] )
    

class find_leftright_index(BaseModel):
    """
    find the left and right index of the array
    """

    def execute(self, working_memory):
        required_params = ["A", "T", "left", "right", "mid"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)

        if(working_memory["A"][working_memory["mid"]] < working_memory["T"]):
            working_memory["left"] = working_memory["mid"] + 1
        if(working_memory["A"][working_memory["mid"]] > working_memory["T"]):
            working_memory["right"] = working_memory["mid"] - 1
            
        return "The left index is {} and the right index is {}.".format( working_memory["left"],working_memory["right"])
    


    
class CheckIndexVal(BaseModel):
    """
    find the value of the index in the array
    """

    def execute(self, working_memory):
        required_params = ["A", "mid"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        return working_memory["A"][working_memory["mid"]] == working_memory["T"]
    

class Check_left_right(BaseModel):
    """
    check if the left index is more than the right index
    """
    def execute(self, working_memory):
        required_params = ["left", "right"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        return working_memory["left"]>working_memory["right"]