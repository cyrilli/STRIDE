from tools import *


class SPPAgent():
    def __init__(self, working_memory, exmps_file):
        '''
        Args:
            working_memory
            exmps_file
        '''
        self.working_memory = working_memory
        self.exmps_file = exmps_file


    def move(self):
    
        with open(self.exmps_file, "a") as f:
            f.write("==== USER ====\n")
            f.write("\nQuestion: What is the shortest distance between vertex 'start' = {} and 'end' = {} in graph G?".format(self.working_memory["start"], self.working_memory["end"]))
            f.write("\nThought: I will use Dijkstra's algorithm for solving this problem. I will use a dictionary called 'dists' for storing the distances from the starting point in the working memory. I need to initialize 'dists'.".format())
            f.write("\nOperation: call function InitDistTable with inputs {}.")
        Dist_init = InitDistTable()
        Dist_init.execute(self.working_memory)
        inputs = {"node":self.working_memory["start"], "distance": 0}
        with open(self.exmps_file, "a") as f:
            f.write("\nThought: I will use a priority que for checking the nodes. I have to push the start node = {} with distance {} in the que".format(self.working_memory["start"],0))
            f.write("\nOperation: call function PushinQue with inputs {}".format(inputs))
        pusher = PushinQue(node=inputs["node"],distance=inputs["distance"])
        pusher.execute(self.working_memory)
        with open(self.exmps_file, "a") as f:
            f.write("\nResults: The start node is pushed!")

        with open(self.exmps_file, "a") as f:
            f.write("\nThought: Now lets check whether the que is empty or not.")
            f.write("\nOperation: call function isQueempty with inputs {}")
        empty = isQueempty()
        if empty.execute(self.working_memory):
            with open(self.exmps_file, "a") as f:
                f.write("\nResult: The que is empty and the algorithm is finished.")
                return None
        else:
            with open(self.exmps_file, "a") as f:
                f.write("\nResult: The que is not empty")
        while True:
            with open(self.exmps_file, "a") as f:
                f.write("\nThought: I have to pop the node from the que.")
                f.write("\nOperation: call function PoPfromQue with inputs {}")
            popper = PoPfromQue()
            current_node, current_distance = popper.execute(self.working_memory)
            with open(self.exmps_file, "a") as f:
                f.write("\nResult: the 'current_node' is {} and its 'current_distance' from the start node is {}.".format(current_node,current_distance))
            inputs = {'current_node':current_node, 'current_distance':current_distance}
            with open(self.exmps_file, "a") as f:
                f.write("\nThought: Now I have to update the 'dists' table and the que with the 'current_node' = {} and 'current_distance' = {}.".format(current_node,current_distance))
                f.write("\nOperation: call function Update_Dist_Que with inputs {}".format(inputs))
            updater = Update_Dist_Que(current_distance=inputs["current_distance"], current_node=inputs["current_node"])
            updater.execute(self.working_memory)
            with open(self.exmps_file, "a") as f:
                f.write("\nResult: The table and the que are updated.")
            with open(self.exmps_file, "a") as f:
                f.write("\nThought: Now lets check whether the que is empty or not.")
                f.write("\nOperation: call function isQueempty with inputs {}")
            empty = isQueempty()
            if empty.execute(self.working_memory):
                with open(self.exmps_file, "a") as f:
                    f.write("\nResult: The que is empty and the algorithm is finished.")
                break
            else:
                with open(self.exmps_file, "a") as f:
                    f.write("\nResult: The que is not empty")
        
        return self.working_memory["dists"][self.working_memory["end"]]

