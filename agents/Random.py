import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

import random

class RandomAgent:
    def __init__(self) -> None:
        pass

    def reset(self):
        pass
    def get_instance_info(self, instance_descript):
        pass
    def update(self, state, reward):
        pass

    def move(self, state):
        if state.actions == [0.0, 1.0]:
            print("offer a random price in [0,1]")
            # offer space of bargaining
            p = random.uniform(0, 1)
            return round(p, 2)
        else:
            print("randomly pick accept or reject")
            return random.choice(state.actions)