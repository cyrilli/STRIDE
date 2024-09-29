from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import numpy as np

tool_names_knapsack_problem = ["Initial_Knapsack_DPTable","Update_Knapsack_DPTable","Check_No_Items"]

class Initial_Knapsack_DPTable(BaseModel):
    """
    initialize the dynamic programming table. It should be a 2D-array with dimensions [n,capacity]
    """

    def execute(self, working_memory):
        required_params = ["dp","capacity","n"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        working_memory["dp"] = [[0 for _ in range(working_memory["capacity"] + 1)] for _ in range(working_memory["n"] + 1)]

        return "The dynamic programming table is initialized with dimensions [{},{}]".format(working_memory["n"],working_memory["capacity"])

        
class Update_Knapsack_DPTable(BaseModel):
    """
    Update the dynamic programming table by adding the new item's value, conditioned on the total capacity
    """
    No_items : int = Field(
        ...,
        description = """the number of items to be considered for calculating the value""",
    )

    def execute(self, working_memory):
        required_params = ["dp", "items"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        for w in range(working_memory["capacity"] + 1):
            if self.No_items == 0 or w == 0:
                working_memory["dp"][self.No_items][w] = 0
            elif working_memory["items"][self.No_items-1][0] <= w:
                working_memory["dp"][self.No_items][w] = max(working_memory["items"][self.No_items-1][1] + working_memory["dp"][self.No_items-1][w-working_memory["items"][self.No_items-1][0]], working_memory["dp"][self.No_items-1][w])
            else:
                working_memory["dp"][self.No_items][w] = working_memory["dp"][self.No_items-1][w]

        
        return "The dynamic programming table is now updated."
    

class Check_No_Items(BaseModel):
    """
    check whether you have considered all of the first n items or not.
    """
    Items_Checked : int = Field(
        ...,
        description = """the number of items to be considered for calculating the value""",
    )

    def execute(self, working_memory):
        required_params = ["dp","capacity","n"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        return working_memory["n"]==self.Items_Checked