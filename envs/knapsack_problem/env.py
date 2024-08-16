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


description_of_problem_class_KSP = """ 
The goal of the knapsack problem is to select a subset of items that maximizes the total value without exceeding a given weight capacity.

Components:
items $items$: A list of tuples which in each tuple the weight and the value of each item is stored (weight,value).
Weight capacity $capacity$: The weight capacity.


Goal:
Select a subset of items that maximizes the total value without exceeding a given weight capacity.
"""

class KSP(BaseEnv):
    
    def __init__(self,env_param):

        '''
        Initialize a knapsack problem

        env_param:
            items  - list - a list of tuples (weight,value)
            capacity - int - the weight capacity
        '''

        self.items = env_param["items"]
        self.capacity = env_param["capacity"]
        self.n = len(self.items)

        self.is_done = False
        self.description_of_problem_class = description_of_problem_class_KSP
        self.name = "knapsack_problem"

    def get_description(self):
            """
            description of the current instance of the shortest path problem
            """
            description = "Now you are going to find the subset of items of {} that maximizes the total value without exceeding a given weight capacity {}.".format(self.items,self.capacity)
            return description
    
    def step(self,action):
         return action
        
    def knapsack_problem(self):
            """
            This function finds the maximum value of the chosen subset which their weight doesn't exceed the capacity.

            Args:
                items: A list of tuples of the items which the weight and the value of each item is stored like (weight,value).
                capacity: The weight capacity.
            Returns:
                Maximum value.
            """

            n = len(self.items)
            dp = [[0 for _ in range(self.capacity + 1)] for _ in range(n + 1)]

            for i in range(n + 1):
                for w in range(self.capacity + 1):
                    if i == 0 or w == 0:
                        dp[i][w] = 0
                    elif self.items[i-1][0] <= w:
                        dp[i][w] = max(self.items[i-1][1] + dp[i-1][w-self.items[i-1][0]], dp[i-1][w])
                    else:
                        dp[i][w] = dp[i-1][w]

            max_value = dp[n][self.capacity]
            return max_value
