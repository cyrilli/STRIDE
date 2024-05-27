from tools import *
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

class DynamicVCGAgent():
    def __init__(self, working_memory, exmps_file):
        '''
        Tabular episodic learner for time-homoegenous MDP.
        Must be used together with true state feature extractor.

        Args:
            working_memory
            exmps_file
        '''
        self.working_memory = working_memory
        self.exmps_file = exmps_file

    def compute_policy(self):
        with open(self.exmps_file, "a") as f:
            f.write("==== USER ====\n")
            # f.write("Question: How do I compute the optimal Q values for this MDP instance using value iteration?\n")
            f.write("Question: Now compute the optimal policy that maximizes all agents' rewards.\n")
        for time in reversed(range(self.working_memory["epLen"])):
            if time == self.working_memory["epLen"]-1:
                with open(self.exmps_file, "a") as f:
                    f.write("Thought: We can compute the optimal policy by value iteration, which iterates from the last time step {} back to the first time step 0. At the last time step {}, since the episode terminate at this step, the one-step lookahead is zero. Thus the Q values simply equal the immediate reward of each state-action pair. To compute the Q values, we only need to call function UpdateQbyRExcluding and add the immediate rewards of all agents. Then the V values can be computed by calling UpdateVExcluding, which takes maximum of the Q values over actions.\n".format(self.working_memory["epLen"]-1,self.working_memory["epLen"]-1))
                    inputs = {"time_step":self.working_memory["epLen"]-1, "agent_to_exclude":None}
                    f.write("Operation: call function UpdateQbyRExcluding with inputs {}.\n".format(inputs))
                    f.write("Result: Q values for time step {} are updated with the immediate rewards and stored in the working memory.\n".format(time))
                    f.write("Operation: call function UpdateVExcluding with inputs {}.\n".format(inputs))
                    f.write("Result: V values for time step {} are updated based on the computed Q values and stored in the working memory.\n".format(time))
                op = UpdateQbyRExcluding(time_step=inputs["time_step"], agent_to_exclude=None)
                op.execute(working_memory=self.working_memory)
                op = UpdateVExcluding(time_step=inputs["time_step"], agent_to_exclude=None)
                op.execute(working_memory=self.working_memory)

            else:
                with open(self.exmps_file, "a") as f:
                    f.write("Thought: Now we need to compute the Q values for time step {} (we can only exit the reasoning process after computing the Q values for time step 0). Using the idea of dynamic programming, we compute the Q values in two steps. **First**, add the immediate rewards of all agents by calling function UpdateQbyRExcluding. **Second** add the one-step lookahead by calling function UpdateQbyPVExcluding. Then the V values can be computed by calling UpdateVExcluding, which takes maximum of the Q values over actions.\n".format(time))
                    inputs = {"time_step":time, "agent_to_exclude":None}
                    f.write("Operation: call function UpdateQbyRExcluding with inputs {}.\n".format(inputs))
                    f.write("Result: Q values for time step {} are updated with the immediate rewards and stored in the working memory.\n".format(time))
                    f.write("Operation: call function UpdateQbyPVExcluding with inputs {}.\n".format(inputs))
                    f.write("Result: Q values for time step {} are updated with the one-step look ahead and stored in the working memory.\n".format(time))
                    f.write("Operation: call function UpdateVExcluding with inputs {}.\n".format(inputs))
                    f.write("Result: V values for time step {} are updated based on the computed Q values and stored in the working memory.\n".format(time))

                op = UpdateQbyRExcluding(time_step=inputs["time_step"], agent_to_exclude=None)
                op.execute(working_memory=self.working_memory)
                op = UpdateQbyPVExcluding(time_step=inputs["time_step"], agent_to_exclude=None)
                op.execute(working_memory=self.working_memory)
                op = UpdateVExcluding(time_step=inputs["time_step"], agent_to_exclude=None)
                op.execute(working_memory=self.working_memory)

        with open(self.exmps_file, "a") as f:
            f.write("Thought: The Q values from the last time step {} to time step 0 have now been calculated. We should exit the reasoning process.\n".format(self.working_memory["epLen"]-1))

    def move(self, state_struct):
        with open(self.exmps_file, "a") as f:
            f.write("==== USER ====\n")
            f.write(state_struct.textual_descript+"\n")
            # f.write("\nQuestion: {} Which action I should choose?\n".format(state_struct.textual_descript))
            f.write("Thought: I should retrieve the Q values for current state, which is {}, and time step, which is {}.\n".format(state_struct.mathematical_descript, state_struct.time_step))
            inputs = {"time_step": state_struct.time_step, "state": state_struct.mathematical_descript, "agent_to_exclude":None}
            f.write("Operation: call function GetQExcluding with inputs {}.\n".format(inputs))

        op = GetQExcluding(time_step=inputs["time_step"], state=inputs["state"], agent_to_exclude=None)
        qvalues = op.execute(working_memory=self.working_memory)
        # print("====")
        # print(qvalues)
        with open(self.exmps_file, "a") as f:
            f.write("Result: {}.\n".format(qvalues))
        op = GetArgMax(number_list=qvalues)
        max_indices = op.execute(working_memory=self.working_memory)
        inputs = {"number_list":qvalues.tolist()}
        with open(self.exmps_file, "a") as f:
            f.write("Thought: I should call function GetArgMax to get the action indices corresponding to the maximal value in the list {}.\n".format(qvalues))
            f.write("Operation: call function GetArgMax with inputs {}.\n".format(inputs))
            f.write("Result: {}.\n".format(max_indices.tolist()))
        a = np.random.choice(np.flatnonzero(qvalues == qvalues.max()))
        with open(self.exmps_file, "a") as f:
            f.write("Thought: Now I can exit the reasoning process, and choose action {}, as it maximizes the Q value (break the tie randomly if there are multiple maximums).\n".format(a))
        return a

    def compute_price(self, agent):
        """
        compute the VCG price for the specified agent
        """
        with open(self.exmps_file, "a") as f:
            f.write("==== USER ====\n")
            # f.write("Question: How do I compute the optimal Q values for this MDP instance using value iteration?\n")
            f.write("Question: Now compute the VCG price for agent {}.\n".format(agent))

        for time in reversed(range(self.working_memory["epLen"])):
            if time == self.working_memory["epLen"]-1:
                with open(self.exmps_file, "a") as f:
                    # f.write("Thought: We can compute the VCG price for agent {} in two steps. Step 1: Let's first compute the policy that maximizes the rewards of all agents except agent {} by value iteration, which iterates from the last time step {} back to the first time step 0. At the last time step {}, since the episode terminate at this step, the one-step lookahead is zero. Thus the Q values simply equal the immediate reward of each state-action pair. To compute the Q values, we only need to call function UpdateQbyRExcluding and add the immediate rewards of all agents except agent {}. Then the V values can be computed by calling UpdateVExcluding, which takes maximum of the Q values over actions.\n".format(agent, agent, self.working_memory["epLen"]-1,self.working_memory["epLen"]-1, agent))
                    f.write("Thought: Step 1: Compute the policy that maximizes the rewards of all agents excluding agent {} by value iteration, which iterates from the last time step {} back to the first time step 0. At the last time step {} (we can only exit the reasoning process after computing the Q values for time step 0), since the episode terminate at this step, the one-step lookahead is zero. Thus the Q values simply equal the immediate reward of each state-action pair. To compute the Q values, we only need to add the immediate rewards of all agents excluding agent {} by calling function UpdateQbyRExcluding. Then the V values can be computed by calling UpdateVExcluding.\n".format(agent, self.working_memory["epLen"]-1,self.working_memory["epLen"]-1, agent))
                    inputs = {"time_step":self.working_memory["epLen"]-1, "agent_to_exclude":agent}
                    op = UpdateQbyRExcluding(time_step=inputs["time_step"], agent_to_exclude=agent)
                    res_txt = op.execute(working_memory=self.working_memory)
                    f.write("Operation: call function UpdateQbyRExcluding with inputs {}.\n".format(inputs))
                    f.write("Result: "+res_txt+"\n")
                    op = UpdateVExcluding(time_step=inputs["time_step"], agent_to_exclude=agent)
                    res_txt = op.execute(working_memory=self.working_memory)
                    f.write("Operation: call function UpdateVExcluding with inputs {}.\n".format(inputs))
                    f.write("Result: "+res_txt+"\n")

            else:
                with open(self.exmps_file, "a") as f:
                    # f.write("Thought: Now we need to compute the Q values for time step {} (we can only exit the reasoning process after computing the Q values for time step 0). Using the idea of dynamic programming, we compute the Q values in two steps. **First**, add the immediate rewards of all agents excluding agent {} by calling function UpdateQbyRExcluding. **Second** add the one-step lookahead by calling function UpdateQbyPVExcluding. Then the V values can be computed by calling UpdateVExcluding, which takes maximum of the Q values over actions.\n".format(time,agent))
                    f.write("Thought: Now we compute Q values for time step {} (we can only exit the reasoning process after computing the Q values for time step 0). Using the idea of dynamic programming, we compute the Q values in two steps. **First**, add the immediate rewards of all agents excluding agent {} by calling function UpdateQbyRExcluding. **Second** add the one-step lookahead by calling function UpdateQbyPVExcluding. Then the V values can be computed by calling UpdateVExcluding.\n".format(time,agent))
                    inputs = {"time_step":time, "agent_to_exclude":agent}
                    op = UpdateQbyRExcluding(time_step=inputs["time_step"], agent_to_exclude=agent)
                    res_txt = op.execute(working_memory=self.working_memory)
                    f.write("Operation: call function UpdateQbyRExcluding with inputs {}.\n".format(inputs))
                    # f.write("Result: Q values for time step {} are updated with the immediate rewards and stored in the working memory.\n".format(time))
                    f.write("Result: "+res_txt+"/n")
                    op = UpdateQbyPVExcluding(time_step=inputs["time_step"], agent_to_exclude=agent)
                    res_txt = op.execute(working_memory=self.working_memory)
                    f.write("Operation: call function UpdateQbyPVExcluding with inputs {}.\n".format(inputs))
                    # f.write("Result: Q values for time step {} are updated with the one-step look ahead and stored in the working memory.\n".format(time))
                    f.write("Result: "+res_txt+"/n")
                    op = UpdateVExcluding(time_step=inputs["time_step"], agent_to_exclude=agent)
                    res_txt = op.execute(working_memory=self.working_memory)
                    f.write("Operation: call function UpdateVExcluding with inputs {}.\n".format(inputs))
                    # f.write("Result: V values for time step {} are updated based on the computed Q values and stored in the working memory.\n".format(time))
                    f.write("Result: "+res_txt+"/n")

        with open(self.exmps_file, "a") as f:
            f.write("Thought: The Q values from the last time step {} to time step 0 have now been calculated. We need to know the value of initial state 0 at time step 0. Let's call GetQExcluding to retrieve all Q values and call GetMax to get the state value.\n".format(self.working_memory["epLen"]-1))
            inputs = {"time_step":0, "state":0}
            f.write("Operation: call function GetQExcluding with inputs {}.\n".format(inputs))
            op = GetQExcluding(time_step=0, state=0, agent_to_exclude=agent)
            qvalues = op.execute(working_memory=self.working_memory)
            f.write("Result: {}.\n".format(qvalues))
            f.write("Operation: call function GetMax with inputs {}.\n".format(qvalues))
            op = GetMax(number_list=qvalues)
            v_value_1 = op.execute(working_memory=self.working_memory)
            f.write("Result: {}.\n".format(v_value_1))
        
        with open(self.exmps_file, "a") as f:
            f.write("Thought: Step 2: we need to evaluate the policy that maximizes all agents' reward on the MDP that excludes agent {}'s reward by calling function EvaluatePolicyExcluding.\n".format(agent))
            op = EvaluatePolicyExcluding(agent_to_exclude=agent)
            v_value_2_text = op.execute(working_memory=self.working_memory)
            f.write("Operation: call function EvaluatePolicyExcluding with inputs {}.\n".format({"agent_to_exclude":agent}))
            f.write("Result: {}.\n".format(v_value_2_text))
            v_value_2 = float(v_value_2_text.split()[-1])
        
        with open(self.exmps_file, "a") as f:
            f.write("Thought: In Step 1, we have computed the value of the policy that maximizes all agents' rewards except agent {}, which is {}, and in Step 2 we have computed the value of the policy that maximizes all agents' rewards including agent {}, which is {}. The VCG price is simply their difference {}-{}={}.\n".format(agent, v_value_1, agent, v_value_2, v_value_1, v_value_2, v_value_1-v_value_2))
        
        return v_value_1-v_value_2
