import json
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

instances_file_path = "NPHardEval_data/Data_V2/KSP/ksp_instances.json"
answers_file_path = "NPHardEval_data/Data_V2/KSP/ksp_answers.json"


def KSP_data_reader():
    
    with open(instances_file_path, 'r') as file:
        input_data = json.load(file)
    with open(answers_file_path, 'r') as file:
        answer_data = json.load(file)
    

    items = []
    capacity = []
    Answers = []
    Levels = [0,0,0,0,0,0,0,0,0,0]

    for item in input_data:
        items.append([(i["weight"], i["value"]) for i in item["items"]])
        capacity.append(item["knapsack_capacity"])

    for item in answer_data:
        Answers.append(item["TotalValue"])
        Levels[item["level"]-1] += 1

    return items, capacity, Answers, Levels

if __name__ == "__main__":
    KSP_data_reader()
