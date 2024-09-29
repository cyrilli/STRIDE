import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

import random
import numpy as np
from copy import deepcopy
from pydantic import BaseModel, NonNegativeInt
from typing import List

from envs.env_helper import BaseEnv

description_of_problem_class_EDP = """
The edit distance problem involves determining the minimum number of operations required to transform one string into another. This distance represents the minimum cost associated with converting one string into the other.

Components:
String A: The initial string that needs to be transformed.
String B: The target string that we want to obtain after transformation.

Operations: The allowed transformations which typically include:

    Insertion: Adding a single character to the string.
    Deletion: Removing a single character from the string.
    Substitution: Replacing one character in the string with another.

Goal:
Identify the minimum number of operations needed to transform string A into string B. This number, denoted as d, represents the edit distance between the two strings.
"""


class EDP(BaseEnv):

    def __init__(self,env_param):

        '''
        Initialize a knapsack problem

        env_param:
            A  - str - string A
            B - str - String B
        '''

        self.A = env_param["A"]
        self.B = env_param["B"]

        self.m = len(self.A)
        self.n = len(self.B)
        self.description_of_problem_class = description_of_problem_class_EDP
        self.name = "edit_distance_problem"

    def step(self,action):
        return action
    

    def get_description(self):
        """
        description of the current instance of edit distance problem
        """
        description = "Now you are going to find the minimum number of operations required for transforming string A = '{}' to string B = '{}'.\n".format(self.A,self.B)
        
        return description
    
    def edp_algorithm(self):

        m = len(self.A)
        n = len(self.B)
        
        # Create a DP table to memoize results of subproblems
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize the DP table
        for i in range(m + 1):
            dp[i][0] = i  # It takes i deletions to transform A[0..i] to an empty string
        for j in range(n + 1):
            dp[0][j] = j  # It takes j insertions to transform an empty string to B[0..j]
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if self.A[i - 1] == self.B[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # No operation needed
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j],  # Deletion
                                    dp[i][j - 1],  # Insertion
                                    dp[i - 1][j - 1])  # Substitution
        
        return dp[m][n]
    
    