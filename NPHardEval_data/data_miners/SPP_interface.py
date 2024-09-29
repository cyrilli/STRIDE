import json
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

instances_file_path = "NPHardEval_data/Data_V2/SPP/spp_instances.json"
answers_file_path = "NPHardEval_data/Data_V2/SPP/spp_answers.json"


def SPP_data_reader():
    
    with open(instances_file_path, 'r') as file:
        input_data = json.load(file)
    with open(answers_file_path, 'r') as file:
        answer_data = json.load(file)
    

    nodes = []
    edges = []
    Answers = []
    Levels = [0,0,0,0,0,0,0,0,0,0]

    for item in input_data:
        nodes.append(item["nodes"])
        edges.append(item["edges"])

    for item in answer_data:
        Answers.append(item["TotalDistance"])
        Levels[item["level"]-1] += 1

    return nodes, edges, Answers, Levels

if __name__ == "__main__":
    nodes, edges, Answers, _ = SPP_data_reader()
    print(nodes[4])
    print(edges[4])
    print(Answers[4])