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

description_of_problem_class_SAS = """ 
The sorted array search problem, also known as the binary search problem, involves finding the position of a target value within a sorted array. This problem leverages the property of the array being sorted to perform the search more efficiently than a linear search.

Components:
Array $A$: an array A
Target value $T$: an integer which is the value we want to find in $A$

Goal of the agent:
Find the position of the target value $T$ within the sorted array $A$
"""

class SAS(BaseEnv):


    def __init__(self, env_param):
        '''
        Initialize a sorted array search problem

        env_param:
            A  - list - number array
            T - int - target value
        '''
        self.name = "sorted_array_search"
        self.required_agents = ["agent"]
        
        self.A = env_param["A"]
        self.T = env_param["T"]
        self.n = len(self.A)
        self.description_of_problem_class = description_of_problem_class_SAS
        self.is_done = False

    def get_description(self):
        """
        description of the current instance of MDP
        """
        description = "Now you are going to find the position of the target value T = {} within the array A = {} after it gets sorted, which will be stored in the working memory.".format(self.T,self.A)
        return description


    def step(self,action):
        if action == self.SAS_algorithm():
            self.is_done = True
        return action

    
    def SAS_algorithm(self):
        self.A = sorted(self.A)
        left = 0
        right = self.n - 1
        while(left <= right):
            mid = left + round((right-left)/2)
            if(self.A[mid] == self.T):
                return mid
            elif(self.A[mid] <= self.T):
                left = mid + 1
            elif(self.A[mid] >= self.T):
                right = mid - 1
        return -1 