from tools import *
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

class IEDSAgent:
    def __init__(self, working_memory, exmps_file):
        '''
        Agent for iterated elimination of dominated strategies.

        Args:
            working_memory
            exmps_file
        '''
        self.working_memory = working_memory
        self.exmps_file = exmps_file

    def eliminate_dominated_strategies(self):
        with open(self.exmps_file, "a") as f:
            f.write("==== USER ====\n")
            f.write("Question: How do I perform iterated elimination of dominated strategies?\n")
            f.write("Thought: To perform iterated elimination of dominated strategies, I need to find and eliminate strictly dominated strategies for each player until no more dominated strategies can be found.\n")

        while not self.check_termination():
            for player in range(len(self.working_memory["remaining_strategies"])):
                dominated_strategy = self.find_dominated_strategy(player)
                if dominated_strategy is not None:
                    self.update_strategies(player, dominated_strategy)

                    with open(self.exmps_file, "a") as f:
                        f.write("==== USER ====\n")
                        f.write("Player {} has the following strictly dominated strategy eliminated: {}\n".format(player, dominated_strategy))

        with open(self.exmps_file, "a") as f:
            f.write("Thought: No more strictly dominated strategies can be found. The iterated elimination process is complete.\n")

    def find_dominated_strategy(self, player):
        op = FindDominatedStrategy(player=player)
        return op.execute(self.working_memory)

    def update_strategies(self, player, dominated_strategy):
        self.working_memory["remaining_strategies"][player].remove(dominated_strategy)
        with open(self.exmps_file, "a") as f:
            f.write("Thought: Eliminated dominated strategy {} for player {}.\n".format(dominated_strategy, player))

    def check_termination(self):
        op = CheckTermination()
        return op.execute(self.working_memory)

    def move(self, state_struct):
        with open(self.exmps_file, "a") as f:
            f.write("==== USER ====\n")
            f.write(state_struct.textual_descript + "\n")
            f.write("Question: Which action should I choose?\n")
            f.write("Thought: I should choose an action that maximizes my payoff given the remaining strategies of the other players.\n")

        remaining_strategies = self.working_memory["remaining_strategies"][state_struct.cur_agent]
        payoffs = []

        for strategy in remaining_strategies:
            payoff = self.compute_payoff(state_struct.cur_agent, strategy)
            payoffs.append(payoff)

        with open(self.exmps_file, "a") as f:
            f.write("Operation: compute payoffs for remaining strategies.\n")
            f.write("Result: {}.\n".format(payoffs))

        best_strategy_index = np.argmax(payoffs)

        with open(self.exmps_file, "a") as f:
            f.write("Operation: choose the strategy with the highest payoff.\n")
            f.write("Result: {}.\n".format(best_strategy_index))
            f.write("Thought: I will choose the strategy {} which has the highest payoff.\n".format(remaining_strategies[best_strategy_index]))

        return remaining_strategies[best_strategy_index]

    def compute_payoff(self, player, strategy):
        """
        Compute the expected payoff for a given player and strategy given the remaining strategies of the other players.
        """
        total_payoff = 0
        opponent_strategies = self.generate_opponent_strategies(player)

        for opp_strategy in opponent_strategies:
            payoff = self.working_memory["payoff_matrix"][player][strategy][tuple(opp_strategy)]
            total_payoff += payoff

        return total_payoff / len(opponent_strategies)

    def generate_opponent_strategies(self, player):
        """
        Generate all combinations of opponent strategies for a given player.
        """
        opponent_strategies = [self.working_memory["remaining_strategies"][i] for i in range(len(self.working_memory["remaining_strategies"])) if i != player]
        return list(product(*[range(len(s)) for s in opponent_strategies]))
