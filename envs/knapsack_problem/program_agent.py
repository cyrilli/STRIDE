from tools import *

class KSPAgent():
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
            f.write("\nQuestion: Which subset of items {} has the maximum value without exceeding the weight capacity {}?".format(self.working_memory["items"], self.working_memory["capacity"]))
            f.write("\nThought: I will solve this problem with dynamic programming. My strategy will be to iteratively find the maximum value that can be obtained using the first i items. I will iterate for {} times. I will do this by updating a dynamic programming table called 'dp'. I need to initialize the 'dp' table.".format(self.working_memory["n"]+1,self.working_memory["n"],self.working_memory["capacity"]))
            f.write("\nOperation: call function Initial_Knapsack_DPTable with inputs {}.")
        DPTable_init = Initial_Knapsack_DPTable()
        DPTable_init.execute(self.working_memory)
        with open(self.exmps_file, "a") as f:
            f.write("\nResult: The 'dp' table is initialized.")
        
        
        for i in range(1,self.working_memory["n"] + 1):
                inputs = {"No_items":i}
                with open(self.exmps_file, "a") as f:
                    f.write("\nThought: I have to calculate the maximum value for the first 'No_items' =  {} items. For this I have to update the 'dp' table.".format(i,self.working_memory["capacity"]))
                    f.write("\nOperation: call function Update_Knapsack_DPTable with inputs {}.".format(inputs))
                TableUpdater = Update_Knapsack_DPTable(No_items = inputs["No_items"])       
                TableUpdater.execute(self.working_memory)
                with open(self.exmps_file, "a") as f:
                    f.write("\nResult: The 'dp' table is updated.")
                inputs = {"Items_Checked":i}
                with open(self.exmps_file, "a") as f:
                    f.write("\nThought: So far I have checked the first 'Items_Checked' = {} items. I have to see whether the algorithm is finished or not".format(i))
                    f.write("\nOperation: call function Check_No_Items with inputs {}.".format(inputs))
                checker = Check_No_Items(Items_Checked = inputs["Items_Checked"])
                if(checker.execute(self.working_memory)):
                    with open(self.exmps_file, "a") as f:
                        f.write("\nResult: I have considered all of the first {} items. So the maximum value is {}".format(self.working_memory["n"],self.working_memory["dp"][self.working_memory["n"]][self.working_memory["capacity"]]))
                else:
                    with open(self.exmps_file, "a") as f:
                        f.write("\nResult:I have to continue the algorithm.".format(i,self.working_memory["n"]))

        # with open(self.exmps_file, "a") as f:
        #     f.write("\nThought: The maximum value is {}.".format(self.working_memory["dp"][self.working_memory["n"]][self.working_memory["capacity"]]))

        return self.working_memory["dp"][self.working_memory["n"]][self.working_memory["capacity"]]
    




class KSPflowAgent():
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
                op_branch = {'condition': None, 'operations':[Initial_Knapsack_DPTable]}
                f.write("==== USER ====")
                f.write("\nQuestion: Which subset of items {} has the maximum value without exceeding the weight capacity {}?".format(self.working_memory["items"], self.working_memory["capacity"]))
                f.write("\nThought: I will solve this problem with dynamic programming. My strategy will be to iteratively find the maximum value that can be obtained using the first i items. I will iterate for {} times. I will do this by updating a dynamic programming table called 'dp'. I need to initialize the 'dp' table.".format(self.working_memory["n"]+1,self.working_memory["n"],self.working_memory["capacity"]))
                f.write("\nOperation Branch:{}".format(op_branch))
                f.write("\nOperation: call function Initial_Knapsack_DPTable with inputs {}.")
                f.write("\nResult: The 'dp' table is initialized.")
            DPTable_init = Initial_Knapsack_DPTable()
            DPTable_init.execute(self.working_memory)
                
            
            for items in range(1,self.working_memory["n"] + 1):
                    inputs = {"No_items":items}
                    op_branch = {'condition':f"items={items}", 'operations':[Update_Knapsack_DPTable, Check_No_Items]}
                    with open(self.exmps_file, "a") as f:
                        f.write("\nOperation Branch:{}".format(op_branch))
                        f.write("\nThought: I have to calculate the maximum value for the first 'No_items' =  {} items. For this I have to update the 'dp' table.".format(items,self.working_memory["capacity"]))
                        f.write("\nOperation: call function Update_Knapsack_DPTable with inputs {}.".format(inputs))
                    TableUpdater = Update_Knapsack_DPTable(No_items = inputs["No_items"])       
                    TableUpdater.execute(self.working_memory)
                    with open(self.exmps_file, "a") as f:
                        f.write("\nResult: The 'dp' table is updated.")
                    inputs = {"Items_Checked":items}
                    with open(self.exmps_file, "a") as f:
                        f.write("\nThought: So far I have checked the first 'Items_Checked' = {} items. I have to see whether the algorithm is finished or not".format(items))
                        f.write("\nOperation: call function Check_No_Items with inputs {}.".format(inputs))
                    checker = Check_No_Items(Items_Checked = inputs["Items_Checked"])
                    if(checker.execute(self.working_memory)):
                        with open(self.exmps_file, "a") as f:
                            f.write("\nResult: I have considered all of the first {} items. So the maximum value is {}".format(self.working_memory["n"],self.working_memory["dp"][self.working_memory["n"]][self.working_memory["capacity"]]))
                    else:
                        with open(self.exmps_file, "a") as f:
                            f.write("\nResult:I have to continue the algorithm.".format(items,self.working_memory["n"]))

            # with open(self.exmps_file, "a") as f:
            #     f.write("\nThought: The maximum value is {}.".format(self.working_memory["dp"][self.working_memory["n"]][self.working_memory["capacity"]]))

            return self.working_memory["dp"][self.working_memory["n"]][self.working_memory["capacity"]]