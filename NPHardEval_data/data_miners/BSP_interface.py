import json
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

instances_file_path = "NPHardEval_data/Data_V2/BSP/bsp_instances.json"
answers_file_path = "NPHardEval_data/Data_V2/BSP/bsp_answers.json"


def BSP_data_reader():
    
    with open(instances_file_path, 'r') as file:
        input_data = json.load(file)
    with open(answers_file_path, 'r') as file:
        answer_data = json.load(file)
    

    A = []
    T = []
    Answers = []
    Levels = [0,0,0,0,0,0,0,0,0,0]

    for item in input_data:
        A.append(item["array"])
        T.append(item["target"])

    for item in answer_data:
        Answers.append(item["Position"])
        Levels[item["level"]-1] += 1

    return A, T, Answers, Levels

if __name__ == "__main__":
    BSP_data_reader()