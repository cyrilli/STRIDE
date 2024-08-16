import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

import random
import numpy as np
from copy import deepcopy
from pydantic import BaseModel, NonNegativeInt
from typing import List
from envs.env_helper import get_env_param


from envs.env_helper import BaseEnv


description_of_problem_class_SPP = """ 
The shortest path problem involves finding the most efficient route between two points (vertices) within a graph. This path minimizes the total cost associated with traversing the edges (connections) between those points.

Components:
Graph G: A mathematical structure representing a network of interconnected points (vertices) and the relationships (edges) between them.
Vertex u: The starting point within the graph G.
Vertex v: The destination point within the graph G.
Edge weights: Values assigned to each edge in the graph representing the cost of traveling along that connection.

Goal:
Identify the shortest path, considering edge weights (if provided), that connects vertex u to vertex v within graph G. This path represents the most efficient route between the two points.
"""


class SPP(BaseEnv):
    
    def __init__(self,env_param):

        '''
        Initialize a shortest path problem problem

        env_param:
            nodes  - dict - nodes of a graph
            edges - dict - edges of a graph
        '''

        self.nodes = env_param["nodes"]
        self.edges = env_param["edges"]
        self.start = env_param["nodes"][0]
        self.end = env_param["nodes"][-1]

        self.is_done = False
        self.description_of_problem_class = description_of_problem_class_SPP
        self.name = "shortest_path_problem"

    def get_description(self):
            """
            description of the current instance of the shortest path problem
            """
            description = "Now you are going to find the most efficient path between vertice start = {} and vertice end = {} in a weighted graph".format(self.start,self.end)
            return description
    
    def step(self,action):
         return action
        
    def shortest_path(self):
            """
            This function finds the shortest distance between two vertices in a graph using Dijkstra's algorithm.

            Args:
                graph: A dictionary representing the graph. Keys are vertices, and values are dictionaries 
                        mapping neighbor vertices to their edge weights.
                source: The starting vertex.
                destination: The destination vertex.

            Returns:
                The distance of the shortest path from source to destination, or None if no path exists.
            """

            # Initialize distances with infinity and paths with None
            distances = {node: float('inf') for node in self.nodes}
            distances[self.start] = 0

            # Priority queue to hold vertices to explore
            pq = PriorityQueue()
            pq.put(0, self.start)

            bidirectional_edges = self.edges + [{"from": edge["to"], "to": edge["from"], "weight": edge["weight"]} for edge in self.edges]

            while not pq.is_empty():
                current_distance, current_node = pq.get()

                if current_node == self.end:
                    break

                if current_distance > distances[current_node]:
                    continue

                for edge in bidirectional_edges:

                    if edge['from'] ==current_node:
                        neighbor = edge['to']
                        weight = edge['weight']
                        distance = current_distance + weight

                        if distance < distances[neighbor]:
                            distances[neighbor] = distance
                            pq.put(neighbor, distance)

            return distances[self.end]


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def is_empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        self.elements.append((priority, item))
        self.elements.sort(reverse=True)  # Sort elements by priority

    def get(self):
        return self.elements.pop()  # Pop the element with the lowest priority
    

if __name__ == "__main__":
    env_param = get_env_param(env_name="shortest_path_problem")
    d = SPP(env_param).shortest_path()
    print(d)
