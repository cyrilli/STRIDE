from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import numpy as np


tool_names_shortest_path_problem = ["PoPfromQue","PushinQue","isQueempty","InitDistTable","CheckCurrentVertex","Update_Dist_Que"]

class PoPfromQue(BaseModel):
    """
    Pop a tuple from the que
    """

    def execute(self,working_memory):
        required_params = ["Q"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        node, distance = working_memory["Q"].pop()  # Pop the element with the lowest priority

        return node, distance
    


class PushinQue(BaseModel):
    """
    Push a tuple in the que
    """

    node : int = Field(
        ...,
        description = """The number of the node""",
    )
    
    
    distance : int = Field(
        ...,
        description = """The distance of the node from the start node"""
    )
    




    def execute(self,working_memory):
        required_params = ["Q"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        

        working_memory["Q"].append((self.distance, self.node))
        working_memory["Q"].sort(reverse=True)

        return "The tuple ({},{}) has been pushed in the que.".format(self.distance, self.node)

class isQueempty(BaseModel):
    """
    Check whether the que is empty or not
    """

    def execute(self,working_memory):
        required_params = ["Q"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        return len(working_memory["Q"]) == 0



class InitDistTable(BaseModel):
    """
    Initialize the dynamic programming table. It should be list of tuples (vertex,inf).
    """

    def execute(self,working_memory):
        required_params = ["dists","nodes","start"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        working_memory["dists"] = {vertex: float('infinity') for vertex in working_memory["nodes"]}
        working_memory["dists"][working_memory["start"]] = 0

        return "The dynamic programming table is now initialized."

class CheckCurrentVertex(BaseModel):
    """
    Check whether the current node is end node or not
    """

    current_vertex : int = Field(
        ...,
        description = """The number of the current node""",
    )

    def execute(self,working_memory):
        required_params = ["dists","nodes","end"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        return self.current_vertex == working_memory["end"]
    


# class FindNeighborsDist(BaseModel):
#     """
#     Find the distances of the current vertex's neighbors from the current vertex
#     """

#     current_vertex : int = Field(
#         ...,
#         description = """The number of the current vertex""",
#     )

#     def execute(self,working_memory):
#         required_params = ["dists","G"]
#         missing_params = []
#         for required_param in required_params:
#             if required_param not in working_memory:
#                 missing_params.append(required_param)
#         if missing_params != []:
#             return "Parameters {} missing in the working memory.".format(missing_params)
        
#         distances = []
#         for (neighbor, weight) in working_memory["G"][self.current_vertex].items():
#             distances.append((neighbor,working_memory["dists"][self.current_vertex] + weight))
#         return distances
    


class Update_Dist_Que(BaseModel):
    """
    Update the distances of the dynamic programming table and the que
    """

    current_distance : int = Field(
        ...,
        description = """The distance of the node from the start node""",
    )

    current_node : int = Field(
        ...,
        description = """The current node""",
    )

    def execute(self,working_memory):
        required_params = ["dists","nodes","edges","Q"]
        missing_params = []
        for required_param in required_params:
            if required_param not in working_memory:
                missing_params.append(required_param)
        if missing_params != []:
            return "Parameters {} missing in the working memory.".format(missing_params)
        
        bidirectional_edges = working_memory["edges"] + [{"from": edge["to"], "to": edge["from"], "weight": edge["weight"]} for edge in working_memory["edges"]]

        for edge in bidirectional_edges:

            if edge['from'] == self.current_node:
                neighbor = edge['to']
                weight = edge['weight']
                distance = self.current_distance + weight

                if distance < working_memory["dists"][neighbor]:
                    working_memory["dists"][neighbor] = distance
                    working_memory["Q"].append((neighbor, distance ))
                    working_memory["Q"].sort(reverse=True)
        return "The table and the que are updated!"

    
