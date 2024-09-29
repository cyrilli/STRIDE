from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import numpy as np

tool_names_edit_distance_problem = ["InitialDPTable","UpdateDPTable"]

class InitialDPTable(BaseModel):
    """
    initialize the dynamic programming table
    """

    def execute(self, working_memory):
        required_params = ["m", "n", "dp"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        working_memory["dp"] = [[0 for _ in range(working_memory["n"] + 1)] for _ in range(working_memory["m"] + 1)]

        for i in range(working_memory["m"] + 1):
            working_memory["dp"][i][0] = i  # It takes i deletions to transform A[0..i] to an empty string
        for j in range(working_memory["n"] + 1):
            working_memory["dp"][0][j] = j  # It takes j insertions to transform an empty string to B[0..j]
        
        return "The dynamic programming table has been initialized!"

class UpdateDPTable(BaseModel):
    """
    update the dynamic programming table
    """

    character_index : int = Field(
        ...,
        description = """ The last character of string A""",
    )

    # last_character_B : int = Field(
    #     ...,
    #     description = """ The last character of string B""",
    # )


    def execute(self, working_memory):
        required_params = ["A", "B", "dp"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        for j in range (1,working_memory["n"]+1):
            if working_memory["A"][self.character_index - 1] == working_memory["B"][j - 1]:
                working_memory["dp"][self.character_index][j] = working_memory["dp"][self.character_index - 1][j - 1]

            else:
                working_memory["dp"][self.character_index][j] = 1 + min(working_memory["dp"][self.character_index - 1][j],  # Deletion
                        working_memory["dp"][self.character_index][j - 1],  # Insertion
                        working_memory["dp"][self.character_index - 1][j - 1])  # Substitution
            
        return "The dynamic programming table is updated after finding the edit distance between {} and all the substrings of B.!".format(working_memory["A"][:self.character_index])
    
