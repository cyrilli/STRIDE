from tools import *

class EDPAgent():
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
            f.write("==== USER ====")
            f.write("\nQuestion: What is the minimum number of operations needed for transforming string A = '{}' to string B = '{}'?".format(self.working_memory["A"], self.working_memory["B"]))
            f.write("\nThought: I will use dynamic programming for solving this problem. My strategy will be to find the edit distance between each substring of A and all the subtrings of B. I need to initialize the dp table which it's dimensions are [(length_of_A+1,length_of_B+1) that is [{},{}]].".format(self.working_memory["m"]+1,self.working_memory["n"]+1))
            f.write("\nOperation: call function InitialDPTable with inputs {}.")
        DPTable_init = InitialDPTable()
        DPTable_init.execute(self.working_memory)
        with open(self.exmps_file, "a") as f:
            f.write("\nResult: So 'dp' is now initialized.")
        
        
        for i in range(1, self.working_memory["m"] + 1):
            inputs = {"character_index":i}
            with open(self.exmps_file, "a") as f:
                f.write("\nThought: I will find the edit distance between the first {} characters of A  = '{}' and all the substrings of B = '{}'.".format(i,self.working_memory["A"],self.working_memory["B"]))
                f.write("\nOperation: call function UpdateDPTable with inputs {}.".format(inputs))
            TableUpdater = UpdateDPTable(character_index=inputs["character_index"])       
            TableUpdater.execute(self.working_memory)
            with open(self.exmps_file, "a") as f:
                f.write("\nResult: dp table is updated after finding the edit distance between '{}' and all the substrings of B = '{}'.".format(self.working_memory["A"][:i],self.working_memory["B"]))

        with open(self.exmps_file, "a") as f:
            f.write("\nThought: The edit distance is {}.".format(self.working_memory["dp"][self.working_memory["m"]][self.working_memory["n"]]))

        return self.working_memory["dp"][self.working_memory["m"]][self.working_memory["n"]]