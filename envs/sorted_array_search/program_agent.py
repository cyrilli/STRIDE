from tools import *

class SASAgent:
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
            f.write("\nQuestion: In which index is T = {} located when the array  A = {} is sorted?".format(self.working_memory["T"], self.working_memory["A"]))
            f.write("\nThought: First, I have to sort array A.")
            f.write("\nOperation: call function sort_array with inputs {}.")
        array_sorter = sort_array()
        array_sorter.execute(self.working_memory)
        with open(self.exmps_file, "a") as f:
            f.write("\nThought: Now, I should find the middle index of the array A = {} in order to compare the target value T = {} with it. For this I should initialize the left and right index. Since we are in time_step = 0 and the length of the array is {}, I will set the left index 0 and the right index {}.".format(self.working_memory["A"], self.working_memory["T"],self.working_memory["n"],self.working_memory["n"]-1,self.working_memory["left"],self.working_memory["right"]))
            f.write("\nOperation: call function init_leftright_index with inputs {}.")
        init_lr_i = init_leftright_index()
        init_lr_i.execute(self.working_memory)
        with open(self.exmps_file, "a") as f:
            f.write("\nResult: The left index is {} and the right index is {}.".format(self.working_memory["left"],self.working_memory["right"]))


        with open(self.exmps_file, "a") as f:
            f.write("\nThought: My strategy is to check the middle index between the left and right index until I either find the target value or the left index becomes bigger than the right index. Now lets compute the middle index with respect to the left index {} and the right index {}.".format(self.working_memory["left"],self.working_memory["right"]))
            f.write("\nOperation: call function findmidindex with inputs {}.")
        mid_index_finder = findmidindex()
        mid_index_finder.execute(self.working_memory)

        with open(self.exmps_file, "a") as f:
            f.write("\nResult: The middle index is {}.".format(self.working_memory["mid"]))

        with open(self.exmps_file, "a") as f:
            f.write("\nThought: Now lets see that if the middle index value of the array equals the target value. For this I have to see the value of 'mid' index = {} in the array and compare it to the target value {}.".format(self.working_memory["mid"],self.working_memory["T"]))
            f.write("\nOperation: call function CheckIndexVal with inputs {}.")
        index_val_finder = CheckIndexVal()
        

        if(index_val_finder.execute(self.working_memory)):
            with open(self.exmps_file, "a") as f:
                f.write("\nResult: The value of middle index {} equals the target value.".format(self.working_memory["mid"]))
            return self.working_memory["mid"]
        else:
            with open(self.exmps_file, "a") as f:
                    f.write("\nResult: Since the value of the middle index {} was not equal to the target value {}, I will continue to find the target value.".format(self.working_memory["A"][self.working_memory["mid"]],self.working_memory["T"]))


        while(True):

            with open(self.exmps_file, "a") as f:
                f.write("\nThought: Now lets update the 'left' and 'right' index based on the new 'mid' = {} index.".format(self.working_memory["mid"]))
                f.write("\nOperation: call function find_leftright_index with inputs {}")
            leftright_index_finder = find_leftright_index()
            leftright_index_finder.execute(self.working_memory)
            with open(self.exmps_file, "a") as f:
                f.write("\nResult: The left index is {} and the right index is {}.".format(self.working_memory["left"],self.working_memory["right"]))

            with open(self.exmps_file, "a") as f:
                f.write("\nThought: Now lets check whether the left index is bigger than the right index or not.")
                f.write("\nOperation: call function Check_left_right with inputs {}")
            compare_left_right = Check_left_right()
            if(compare_left_right.execute(self.working_memory)):
                with open(self.exmps_file, "a") as f:
                    f.write("\nResult: The left index is bigger than the right index therefore the target value is not in the array.")
                    break
            else:
                with open(self.exmps_file, "a") as f:
                    f.write("\nResult: The left index is not bigger than the right index therefore we can continue.")

            with open(self.exmps_file, "a") as f:
                f.write("\nThought: Now lets compute the middle index with respect to the left index {} and the right index {}.".format(self.working_memory["left"],self.working_memory["right"]))
                f.write("\nOperation: call function findmidindex with inputs {}.")
            mid_index_finder = findmidindex()
            mid_index_finder.execute(self.working_memory)
            with open(self.exmps_file, "a") as f:
                f.write("\nResult: The middle index is {}.".format(self.working_memory["mid"]))

            with open(self.exmps_file, "a") as f:
                f.write("\nThought: Now lets see that if the middle index value of the array equals the target value. For this I have to see the value of 'mid' index = {} in the array and then compare it with the target value {}.".format(self.working_memory["mid"],self.working_memory["T"]))
                f.write("\nOperation: call function CheckIndexVal with inputs {}.")
            index_val_finder = CheckIndexVal()
            

            if(index_val_finder.execute(self.working_memory)):
                with open(self.exmps_file, "a") as f:
                    f.write("\nResult: The value of middle index {} equals the target value.".format(self.working_memory["mid"]))
                    f.write("\nThought: The index is {}.".format(self.working_memory["mid"]))
                return self.working_memory["mid"]
            else:
                with open(self.exmps_file, "a") as f:
                    f.write("\nResult: Since the value of the middle index {} was not equal to the target value {}, I will continue to find the target value.".format(self.working_memory["A"][self.working_memory["mid"]],self.working_memory["T"]))
        
        
        with open(self.exmps_file, "a") as f:
            f.write("\nThought: The target value T = {} is not in the array.".format(self.working_memory["T"]))
        return -1