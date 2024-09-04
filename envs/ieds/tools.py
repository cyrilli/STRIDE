from pydantic import BaseModel, Field
from typing import List
import numpy as np
from itertools import product

tool_names_ieds = [
    "EliminateDominatedStrategy", 
    "FindDominatedStrategy", 
    "CheckDominance", 
    "GenerateOpponentStrategies", 
    "CheckTermination"
]

class EliminateDominatedStrategy(BaseModel):
    """
    Eliminate a strictly dominated strategy for the current player
    """
    def execute(self, working_memory):
        required_params = ["remaining_strategies", "current_player", "payoff_matrix", "dominated_strategies", "num_players"]
        for param in required_params:
            if param not in working_memory:
                return f"Parameter {param} is missing in the working memory."

        player = working_memory["current_player"]
        dominated_strategies = FindDominatedStrategy(player=player).execute(working_memory)
        if dominated_strategies:
            # for strategy in dominated_strategies:
            strategy = dominated_strategies[0]
            working_memory["remaining_strategies"][player].remove(strategy)
            working_memory["dominated_strategies"][player].append(strategy)

        updated_payoff_matrix = self.generate_remaining_payoff_matrix(working_memory)
        working_memory["payoff_matrix"] = updated_payoff_matrix

        return "Dominated strategies eliminated and remaining strategies updated in working memory."

    def generate_remaining_payoff_matrix(self, working_memory):
            '''Generate the payoff matrix with only the remaining strategies'''
            cur_payoff_matrix = []
            remaining_strategies = working_memory["remaining_strategies"]
            payoff_matrix = working_memory["payoff_matrix"]
            for player in range(len(payoff_matrix)):
                player_matrix = []
                for i in remaining_strategies[0]:
                    row = []
                    for j in remaining_strategies[1]:
                        row.append(payoff_matrix[player][i][j])
                    player_matrix.append(row)
                cur_payoff_matrix.append(player_matrix)
            
            print("Remaining Payoff for Player 1:", cur_payoff_matrix[0])
            print("Remaining Payoff for Player 2:", cur_payoff_matrix[1])
            return cur_payoff_matrix

class FindDominatedStrategy(BaseModel):
    """
    Find a strictly dominated strategy for a given player.
    """
    player: int = Field(..., description="The player for whom to find dominated strategies")

    def execute(self, working_memory):
        required_params = ["remaining_strategies", "payoff_matrix"]
        for param in required_params:
            if param not in working_memory:
                return f"Parameter {param} is missing in the working memory."

        remaining_strategies = working_memory["remaining_strategies"][self.player]
        dominated_strategy = None
        for i in range(len(remaining_strategies)):
            s1 = remaining_strategies[i]
            for j in range(i + 1, len(remaining_strategies)):
                s2 = remaining_strategies[j]
                # if s1 != s2 and CheckDominance(player=self.player, s1=s1, s2=s2).execute(working_memory):
                #     dominated_strategies.append(s1)
                #     break
                dominated_result = CheckDominance(player=self.player, s1=s1, s2=s2).execute(working_memory)
                if dominated_result == s1:
                    dominated_strategy = s1
                elif dominated_result == s2:
                    dominated_strategy = s2

        return dominated_strategy

class CheckDominance(BaseModel):
    """
    Check if a strategy s1 is strictly dominated by strategy s2 or vice versa for a given player.
    """
    player: int = Field(..., description="The player for whom to check dominated strategies")
    s1: int = Field(..., description="Strategy s1 to check")
    s2: int = Field(..., description="Strategy s2 to compare against")

    def execute(self, working_memory):
        required_params = ["payoff_matrix", "remaining_strategies", "num_players"]
        missing_params = [param for param in required_params if param not in working_memory]
        if missing_params:
            return f"Parameters {missing_params} are missing in the working memory."

        opponent_strategies = GenerateOpponentStrategies(player=self.player).execute(working_memory)

        # Indices of other players
        opponent_indices = [i for i in range(working_memory["num_players"]) if i != self.player]

        # Initialize dominance checks
        s1_dominated = True
        s2_dominated = True

        for opp_strategies in opponent_strategies:
            # Build the full index list for accessing the payoff matrix
            s1_index = [opponent_strategies[i].index(opp_strategies[i]) for i in range(len(opponent_indices))]
            s2_index = [opponent_strategies[i].index(opp_strategies[i]) for i in range(len(opponent_indices))]

            # Map s1 and s2 to their positions in the remaining strategies list
            mapped_s1 = working_memory["remaining_strategies"][self.player].index(self.s1)
            mapped_s2 = working_memory["remaining_strategies"][self.player].index(self.s2)

            s1_index.insert(self.player, mapped_s1)
            s2_index.insert(self.player, mapped_s2)

            # Retrieve payoffs for the current strategy profiles
            payoff_s1 = working_memory["payoff_matrix"][self.player]
            payoff_s2 = working_memory["payoff_matrix"][self.player]

            for idx in s1_index:
                payoff_s1 = payoff_s1[idx]
            for idx in s2_index:
                payoff_s2 = payoff_s2[idx]

            # Check for dominance conditions
            if payoff_s1 >= payoff_s2:
                s2_dominated = False
            if payoff_s2 >= payoff_s1:
                s1_dominated = False

            # Early exit if neither strategy is strictly dominated
            if not s1_dominated and not s2_dominated:
                return None

        if s1_dominated:
            return self.s1
        elif s2_dominated:
            return self.s2
        else:
            return None

class GenerateOpponentStrategies(BaseModel):
    """
    Generate all combinations of opponent strategies for a given player.
    """
    player: int = Field(..., description="The player for whom to generate opponent strategies")

    def execute(self, working_memory):
        required_params = ["remaining_strategies", "current_player"]
        for param in required_params:
            if param not in working_memory:
                return f"Parameter {param} is missing in the working memory."

        opponent_strategies = [working_memory["remaining_strategies"][i] for i in range(len(working_memory["remaining_strategies"])) if i != self.player]
        return list(product(*opponent_strategies))

class CheckTermination(BaseModel):
    """
    Check if no more strictly dominated strategies can be found.
    """
    def execute(self, working_memory):
        required_params = ["remaining_strategies", "payoff_matrix", "current_player"]
        for param in required_params:
            if param not in working_memory:
                return f"Parameter {param} is missing in the working memory."

        for player in range(len(working_memory["remaining_strategies"])):
            if FindDominatedStrategy(player=player).execute(working_memory) is not None:
                return False
        return True
