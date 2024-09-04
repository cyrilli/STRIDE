import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

import numpy as np
from itertools import product
from copy import deepcopy
from pydantic import BaseModel, NonNegativeInt
from typing import List, Optional

from envs.env_helper import BaseEnv, get_env_param

description_of_problem_class_ieds = """
Iterated Elimination of Dominated Strategies (IEDS) is a process used in game theory to simplify the analysis of strategic interactions by iteratively removing strategies that are strictly dominated.

Components:
Players: A finite set of players, each with a finite set of strategies.
Payoff Matrix: A matrix that represents the payoffs for each player given the strategies chosen by all players.

Interaction protocol:
1. Identify strictly dominated strategies for each player and remove them.
2. Update the game by considering only the remaining strategies.
3. Repeat steps 1 and 2 until no more strictly dominated strategies can be identified.

Goal of the players:
Each player aims to maximize their payoff, and the IEDS process helps in narrowing down the set of strategies to those that could be rationally chosen by eliminating those that are dominated.
"""

class State(BaseModel):
    iteration: int
    remaining_strategies: List[List[NonNegativeInt]]
    payoff_matrix: List[List[List[int]]]
    textual_descript: str
    mathematical_descript: str

    def is_valid_action(self, action):
        return True

    def generate_remaining_payoff_matrix(self):
        '''Generate the payoff matrix with only the remaining strategies'''
        cur_payoff_matrix = []
        for player in range(len(self.payoff_matrix)):
            player_matrix = []
            for i in self.remaining_strategies[0]:
                row = []
                for j in self.remaining_strategies[1]:
                    row.append(self.payoff_matrix[player][i][j])
                player_matrix.append(row)
            cur_payoff_matrix.append(player_matrix)
        print("Ending Payoff for Player 1:\n{}".format(cur_payoff_matrix[0]))
        print("Ending Payoff for Player 2:\n{}".format(cur_payoff_matrix[1]))
        print("==============================================================")
        return cur_payoff_matrix

    def update_textual_descript(self):
        '''Update the textual description to include remaining payoff matrices'''
        self.textual_descript = "ITERATION: {}\nStarting Strategies for Player {}:\n{}\nStarting Strategies for Player {}:\n{}\n".format(
            self.iteration, 1, self.remaining_strategies[0], 2, self.remaining_strategies[1]
        )

class IteratedEliminationDominatedStrategies(BaseEnv):
    def __init__(self, env_param):
        self.name = "ieds"
        self.required_agents = ["player"]
        self.env_param = env_param
        self.num_players = env_param["num_players"]
        self.cur_player = 0
        self.strategies_per_player = env_param["strategies_per_player"]
        self.payoff_matrix = env_param["payoff_matrix"]
        self.dominated_strategies = []

        self.reset()
        self.description_of_problem_class = description_of_problem_class_ieds

    def get_description(self, agent_role):
        description = "This is the beginning of a new game instance, where you will play as a player using the Iterated Elimination of Dominated Strategies (IEDS) method. The game has {} players, each with a set of strategies. The payoff matrix is provided. Your goal is to iteratively eliminate strictly dominated strategies and simplify the game.\n".format(self.num_players)
        return description

    def reset(self):
        '''Reset the environment'''
        self.state = State(iteration=1, remaining_strategies=[list(range(s)) for s in self.strategies_per_player], payoff_matrix=self.payoff_matrix, textual_descript="This is iteration {}. The remaining strategies are {}.".format(0, [list(range(s)) for s in self.strategies_per_player]), mathematical_descript="Initial state")
        self.is_done = False
        self.state.update_textual_descript()

    def step(self):
        '''
        Perform one step of the IEDS process

        Returns:
        newState - State - new state after elimination
        '''
        self.state.iteration += 1
        elimination_occured = False

        for player in range(self.num_players):
            self.cur_player = player
            # self.eliminate_dominated_strategies()
            dominated_strategy, dominating_strategy = self.find_strictly_dominated_strategy(self.cur_player)

            if dominated_strategy is not None and dominating_strategy is not None:
                # self.eliminate_dominated_strategies(dominated_strategies)
                print("Strategy {} strictly dominates strategy {} for Player {}.".format(
                dominating_strategy, dominated_strategy, player+1))
                self.eliminate_strictly_dominated_strategy(dominated_strategy)
                self.payoff_matrix = self.state.generate_remaining_payoff_matrix()
                elimination_occured = True
                if self.check_termination():
                    self.is_done = True
                    break
                
                break

        if not elimination_occured:
            self.check_termination()
            self.is_done = True

        self.state.update_textual_descript()

        return self.state

    def eliminate_strictly_dominated_strategy(self, strategy):
        '''Eliminate a single strictly dominated strategy for the current player'''
        self.state.remaining_strategies[self.cur_player].remove(strategy)
        # +1 to strategy and player to match 1-indexing for output clarity
        print("Eliminating strictly dominated strategy {} for Player {}...\n".format(strategy, self.cur_player + 1))
        print("Ending strategies for Player 1:\n{}".format(self.state.remaining_strategies[0]))
        print("Ending strategies for Player 2:\n{}\n".format(self.state.remaining_strategies[1]))


    def find_strictly_dominated_strategy(self, player):
        '''Find a strictly dominated strategy for a given player'''
        remaining_strategies = self.state.remaining_strategies[player]
        dominated_strategy = None
        dominating_strategy = None

        for i in range(len(remaining_strategies)):
            s1 = remaining_strategies[i]
            for j in range(i + 1, len(remaining_strategies)):
                s2 = remaining_strategies[j]
                dominated_result = self.check_dominance(player, s1, s2)

                if dominated_result == s1:
                    dominated_strategy = s1
                    dominating_strategy = s2

                elif dominated_result == s2:
                    dominated_strategy = s2
                    dominating_strategy = s1

            if dominated_strategy is not None and dominating_strategy is not None:
                break
                
        return dominated_strategy, dominating_strategy

    def check_dominance(self, player, s1, s2):
        '''Check if strategy s1 is strictly dominated by strategy s2 or vice versa for player'''
        opponent_indices = [i for i in range(self.num_players) if i != player]
        opp_remaining_strategies = [self.state.remaining_strategies[i] for i in opponent_indices]

        s1_dominates = True
        s2_dominates = True

        for opp_strategies in product(*opp_remaining_strategies):
            # Build the full index list for accessing the payoff matrix
            s1_index = [opp_remaining_strategies[i].index(opp_strategies[i]) for i in range(len(opponent_indices))]
            s2_index = [opp_remaining_strategies[i].index(opp_strategies[i]) for i in range(len(opponent_indices))]

            # Map s1 and s2 to their positions in the remaining strategies list
            mapped_s1 = self.state.remaining_strategies[player].index(s1)
            mapped_s2 = self.state.remaining_strategies[player].index(s2)

            s1_index.insert(player, mapped_s1)
            s2_index.insert(player, mapped_s2)

            payoff_s1 = self.payoff_matrix[player]
            payoff_s2 = self.payoff_matrix[player]

            for idx in s1_index:
                payoff_s1 = payoff_s1[idx]
            for idx in s2_index:
                payoff_s2 = payoff_s2[idx]

            if payoff_s1 >= payoff_s2:
                s2_dominates = False
            if payoff_s2 >= payoff_s1:
                s1_dominates = False

            # Early exit if neither strictly dominates
            if not s1_dominates and not s2_dominates:
                return None

        if s1_dominates:
            return s2
        elif s2_dominates:
            return s1  # s1 strictly dominates s2
        else:
            return None

    def is_dominated(self, player, s1, s2):
        '''Check if strategy s1 is strictly dominated by strategy s2 for player'''
        opponent_indices = [i for i in range(self.num_players) if i != player]

        opp_remaining_strategies = []
        for i in opponent_indices:
            opp_remaining_strategies.append(self.state.remaining_strategies[i])

        for opp_strategies in product(*opp_remaining_strategies):
            # Build the full index list for accessing the payoff matrix
            s1_index = [opp_remaining_strategies[i].index(opp_strategies[i]) for i in range(len(opponent_indices))]
            s2_index = [opp_remaining_strategies[i].index(opp_strategies[i]) for i in range(len(opponent_indices))]

            # Map s1 and s2 to their positions in the remaining strategies list
            mapped_s1 = self.state.remaining_strategies[player].index(s1)
            mapped_s2 = self.state.remaining_strategies[player].index(s2)

            s1_index.insert(player, mapped_s1)
            s2_index.insert(player, mapped_s2)

            payoff_s1 = self.payoff_matrix[player]
            payoff_s2 = self.payoff_matrix[player]

            for idx in s1_index:
                payoff_s1 = payoff_s1[idx]
            for idx in s2_index:
                payoff_s2 = payoff_s2[idx]

            if payoff_s1 >= payoff_s2:
                return False
        return True

    def generate_opponent_strategies(self, player):
        '''Generate all combinations of opponent strategies for a given player'''
        opponent_strategies = [self.state.remaining_strategies[i] for i in range(self.num_players) if i != player]
        return opponent_strategies[0]

    def check_termination(self):
        '''Check if no more strictly dominated strategies can be found'''
        for player in range(self.num_players):
            dominated_strategy, dominating_strategy = self.find_strictly_dominated_strategy(player) 
            if dominated_strategy is not None and dominating_strategy is not None:
                return False

        remaining_strategies_count = [len(strategies) for strategies in self.state.remaining_strategies]
        
        if all(count == 1 for count in remaining_strategies_count):
            print("Game ended: Unique PNE found.\n")
        elif self.strategies_per_player[0] == remaining_strategies_count[0] and self.strategies_per_player[1] == remaining_strategies_count[1]:
            print("Game ended early: No PNE found.\n")
        elif 1 < remaining_strategies_count[0] < self.strategies_per_player[0] and 1 < remaining_strategies_count[1] < self.strategies_per_player[1]:
            print("Game ended early: Multiple PNE found.\n")

        return True
    

if __name__ == "__main__":
    ## num_pne: 0 for no PNE, 1 for unique PNE, 2+ for multiple PNE
    env_param = get_env_param("ieds", random_param=False)
    env = IteratedEliminationDominatedStrategies(env_param=env_param)

    state = env.state
    print(state.textual_descript)
    while not env.is_done:
        state = env.step()
        print(state.textual_descript)

    print("\nThe game has ended! \nFinal remaining strategies are: {}. \nFinal remaining payoff is: {}".format(state.remaining_strategies, env.payoff_matrix))