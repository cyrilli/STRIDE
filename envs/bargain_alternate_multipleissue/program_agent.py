from tools import *
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

class BargainAgent:
    def __init__(self, working_memory, exmps_file) -> None:
        self.working_memory = working_memory
        self.exmps_file = exmps_file

    def compute_spe(self, role):
        with open(os.path.join(current_dir, self.exmps_file), "a") as f:
            f.write("==== USER ====\n")
            f.write("Question: Now compute the subgame perfect equilibrium (SPE) for each time step.\n")     
        for t in reversed(range(1, self.working_memory["T"]+1)):
            # first step
            if t == self.working_memory["T"]:
                if self.working_memory["T"] % 2 == 1:
                    agent = "buyer"
                else:
                    agent = "seller"

                op = BackwardOneStep(agent_type=agent, opponent_util_if_rej=0, time_step=t)
                result, price_list, util = op.execute(working_memory=self.working_memory)
                # price_list = eval(price_result.split(":")[-1])

                with open(os.path.join(current_dir, self.exmps_file), "a") as f:
                    inputs = {"agent": agent, "opponent_util_if_rej": 0, "time_step": t}
                    f.write(f"Operation: call function BackwardOneStep with inputs {inputs}.\n")
                    f.write(f"Result: {price_list}\n")

                op = CalcUtil(agent_type=agent, price=price_list, time_step=t)
                util_result = op.execute(working_memory=self.working_memory)
                util_list = eval(util_result.split(":")[-1])

                with open(os.path.join(current_dir, self.exmps_file), "a") as f:
                    inputs = {"agent": agent, "price": price_list, "time_step": t}
                    f.write(f"Operation: call function CalcUtil with inputs {inputs}.\n")
                    f.write(f"Result: {util_result}\n")
            else:
                if agent == "buyer":
                    agent = "seller"
                    opponent = "buyer"
                else:
                    agent = "buyer"
                    opponent = "seller"

                op = BackwardOneStep(agent_type=agent, opponent_util_if_rej=util_list, time_step=t)
                result, price_list, util = op.execute(working_memory=self.working_memory)
                # price_list = eval(price_result.split(":")[-1])

                op = CalcUtil(agent_type=agent, price=price_list, time_step=t)
                util_result = op.execute(working_memory=self.working_memory)
                util_list = eval(util_result.split(":")[-1])

                path = os.path.join(current_dir, self.exmps_file)

                with open(os.path.join(current_dir, self.exmps_file), "a") as f:
                    inputs = {"agent": agent, "opponent_util_if_rej": util_list, "time_step": t}
                    f.write(f"Operation: call function BackwardOneStep with inputs {inputs}.\n")
                    f.write(f"Result: {price_list}\n")
                    inputs = {"agent": agent, "price": price_list, "time_step": t}
                    f.write(f"Operation: call function CalcUtil with inputs {inputs}.\n")
                    f.write(f"Result: {util_result}\n")

            self.working_memory["SPEPrice"][t] = np.round(price_list, 2)

        with open(os.path.join(current_dir, self.exmps_file), "a") as f:
            f.write(f"Thought: I have finished my reasoning from the last time step t={self.working_memory['T']} to time step t=1, so I can exit the reasoning process, and save the computed subgame perfect equilibrium strategy in my working memory.\n")

        
    
    def move(self, state):
        if state.actions == [0.0, 1.0]:
            with open(os.path.join(current_dir, self.exmps_file), "a") as f:
                f.write("==== USER ====\n")
                f.write(f"Question: {state.textual_descript}\n")

            assert self.working_memory["SPEPrice"] != {}
            op = GetSPEPrice(agent_type=state.cur_agent, time_step=state.time_step)
            spe_price_result = op.execute(working_memory=self.working_memory)
            spe_price_list = eval(spe_price_result.split(":")[-1])
            spe_price_list = np.round(spe_price_list, 2)

            with open(os.path.join(current_dir, self.exmps_file), "a") as f:
                f.write(f"Thought: Assuming my opponent is rational, then to maximize my utility, I should adopt the SPE strategy. Therefore, I should retrieve the computed SPE price for the current time step {state.time_step} by calling GetSPEPrice.\n")
                inputs = {"agent": state.cur_agent, "time_step": state.time_step}
                f.write(f"Operation: call function GetSPEPrice with inputs {inputs}.\n")
                f.write(f"Result: {spe_price_result}\n")
                f.write(f"Thought: I can exit the reasoning process and propose a price of {spe_price_list}, which is my SPE strategy at time step {state.time_step}.\n")

            return np.round(spe_price_list, 2)
        
        else:
            price = state.mathematical_descript[-1]
            op = CalcUtil(agent_type=state.cur_agent, price=price, time_step=state.time_step)
            util_acc_result = op.execute(working_memory=self.working_memory)
            util_acc_list = eval(util_acc_result.split(":")[-1])

            if state.time_step < self.working_memory["T"]:
                with open(os.path.join(current_dir, self.exmps_file), "a") as f:
                    f.write("==== USER ====\n")
                    f.write(f"Question: {state.textual_descript}\n")
                    f.write(f"Thought: I should compare the utility I can get by accepting this price vs the utility I can get by rejecting it and making a counteroffer in the next time step. First, let me compute my utility for accepting the price of {price} at the current time step {state.time_step} by calling CalcUtil.\n")
                    inputs = {"agent": state.cur_agent, "price": [price] * len(self.working_memory["buyerWeights"]), "time_step": state.time_step}
                    f.write(f"Operation: call function CalcUtil with inputs {inputs}.\n")
                    f.write(f"Result: {util_acc_result}\n")

                # assert self.working_memory["SPEPrice"] != {}
                op = GetSPEPrice(agent_type=state.cur_agent, time_step=state.time_step + 1)
                spe_price_result = op.execute(working_memory=self.working_memory)
                spe_price_list = eval(spe_price_result.split(":")[-1])

                op = CalcUtil(agent_type=state.cur_agent, price=spe_price_list, time_step=state.time_step + 1)
                util_rej_result = op.execute(working_memory=self.working_memory)
                util_rej_list = eval(util_rej_result.split(":")[-1])

                with open(os.path.join(current_dir, self.exmps_file), "a") as f:
                    f.write(f"Thought: Then I should reason about the utility I can get by rejecting and making a counteroffer at the next time step {state.time_step + 1}. Since I have already computed the subgame perfect equilibrium strategy and saved it in the working memory, I can retrieve the previously computed price for the next time step {state.time_step + 1} by calling GetSPEPrice. Then I can compute my utility by calling CalcUtil.\n")
                    inputs = {"agent": state.cur_agent, "time_step": state.time_step + 1}
                    f.write(f"Operation: call function GetSPEPrice with inputs {inputs}.\n")
                    f.write(f"Result: {spe_price_result}\n")
                    inputs = {"agent": state.cur_agent, "price": spe_price_list, "time_step": state.time_step + 1}
                    f.write(f"Operation: call function CalcUtil with inputs {inputs}.\n")
                    f.write(f"Result: {util_rej_result}\n")

                util_acc = np.array(util_acc_list)
                util_rej = np.array(util_rej_list)
                if np.all(util_acc >= util_rej):
                    with open(os.path.join(current_dir, self.exmps_file), "a") as f:
                        f.write("Thought: Since the utility I get by accepting the offer is greater or equal to that of rejecting the offer, I can exit the reasoning process and choose to accept the offer.\n")
                    return "accept"
                else:
                    with open(os.path.join(current_dir, self.exmps_file), "a") as f:
                        f.write("Thought: Since the utility I get by accepting the offer is less than that of rejecting the offer, I can exit the reasoning process and choose to reject the offer.\n")
                    return "reject"
            else:
                with open(os.path.join(current_dir, self.exmps_file), "a") as f:
                    f.write("==== USER ====\n")
                    f.write(f"Question: {state.textual_descript}\n")
                    f.write(f"Thought: Since this is the last time step, as long as my utility of accepting the price of {price} at the current time step {state.time_step} is positive, I will accept it. Therefore, I should call function CalcUtil to compute the utility.\n")
                    inputs = {"agent": state.cur_agent, "price": [price] * len(self.working_memory["buyerWeights"]), "time_step": state.time_step}
                    f.write(f"Operation: call function CalcUtil with inputs {inputs}.\n")
                    f.write(f"Result: {util_acc_result}\n")

                util_acc = np.array(util_acc_list)
                if np.all(util_acc >= 0):
                    with open(os.path.join(current_dir, self.exmps_file), "a") as f:
                        f.write("Thought: Since the utility I get by accepting the offer is greater or equal to 0, I can exit the reasoning process and choose to accept the offer.\n")
                    return "accept"
                else:
                    with open(os.path.join(current_dir, self.exmps_file), "a") as f:
                        f.write("Thought: Since the utility I get by accepting the offer is less than 0, I can exit the reasoning process and choose to reject the offer.\n")
                    return "reject"