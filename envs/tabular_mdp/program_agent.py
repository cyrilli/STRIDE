from tools import *

class VIAgent():
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
            f.write("Question: Now compute the optimal policy, that is, the optimal action at each step and each state.\n")
        for time in reversed(range(self.working_memory["epLen"])):
            if time == self.working_memory["epLen"]-1:
                with open(self.exmps_file, "a") as f:
                    f.write("Thought: We can compute the optimal policy by value iteration, which iterates from the last time step {} back to the first time step 0. At the last time step {}, since the episode terminate at this step, the one-step lookahead is zero. Thus the Q values simply equal the immediate reward of each state-action pair. To compute the Q values, we only need to call function UpdateQbyR to add the immediate rewards. Then the V values can be computed by calling UpdateV, which takes maximum of the Q values over actions.\n".format(self.working_memory["epLen"]-1,self.working_memory["epLen"]-1))
                    inputs = {"time_step":self.working_memory["epLen"]-1}
                    f.write("Operation: call function UpdateQbyR with inputs {}.\n".format(inputs))
                    f.write("Result: Q values for time step {} are updated with the immediate rewards and stored in the working memory.\n".format(time))
                    f.write("Operation: call function UpdateV with inputs {}.\n".format(inputs))
                    f.write("Result: V values for time step {} are updated based on the computed Q values and stored in the working memory.\n".format(time))
                op = UpdateQbyR(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)
                op = UpdateV(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)

            else:
                with open(self.exmps_file, "a") as f:
                    f.write("Thought: Now we need to compute the Q values for time step {} (we can only exit the reasoning process after computing the Q values for time step 0). Using the idea of dynamic programming, we compute the Q values in two steps. **First**, add the immediate rewards by calling function UpdateQbyR. **Second** add the one-step lookahead by calling function UpdateQbyPV. Then the V values can be computed by calling UpdateV, which takes maximum of the Q values over actions.\n".format(time))
                    inputs = {"time_step":time}
                    f.write("Operation: call function UpdateQbyR with inputs {}.\n".format(inputs))
                    f.write("Result: Q values for time step {} are updated with the immediate rewards and stored in the working memory.\n".format(time))
                    f.write("Operation: call function UpdateQbyPV with inputs {}.\n".format(inputs))
                    f.write("Result: Q values for time step {} are updated with the one-step look ahead and stored in the working memory.\n".format(time))
                    f.write("Operation: call function UpdateV with inputs {}.\n".format(inputs))
                    f.write("Result: V values for time step {} are updated based on the computed Q values and stored in the working memory.\n".format(time))

                op = UpdateQbyR(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)
                op = UpdateQbyPV(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)
                op = UpdateV(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)

        with open(self.exmps_file, "a") as f:
            f.write("Thought: The Q values from the last time step {} to time step 0 have now been calculated. We should exit the reasoning process.\n".format(self.working_memory["epLen"]-1))

    def move(self, state_struct):
        with open(self.exmps_file, "a") as f:
            f.write("==== USER ====\n")
            f.write(state_struct.textual_descript+"\n")
            # f.write("\nQuestion: {} Which action I should choose?\n".format(state_struct.textual_descript))
            f.write("Thought: I should retrieve the Q values for current state, which is {}, and time step, which is {}.\n".format(state_struct.mathematical_descript, state_struct.time_step))
            inputs = {"time_step": state_struct.time_step, "state": state_struct.mathematical_descript}
            f.write("Operation: call function GetQ with inputs {}.\n".format(inputs))

        op = GetQ(time_step=inputs["time_step"], state=inputs["state"])
        qvalues = op.execute(working_memory=self.working_memory)
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


class UCBVIAgent():
    def __init__(self, working_memory, exmps_file, write_exmps):
        '''
        Tabular episodic learner for MDP.

        Args:
            working_memory
            exmps_file
        '''
        self.working_memory = working_memory
        self.exmps_file = exmps_file
        self.write_exmps = write_exmps

    def reset(self):
        # reset V and Q
        self.working_memory["V"] = np.zeros((self.working_memory["epLen"],self.working_memory["nState"]))
        self.working_memory["Q"] = np.zeros((self.working_memory["epLen"],self.working_memory["nState"],self.working_memory["nAction"]))

    def compute_policy(self):
        if self.write_exmps:
            with open(self.exmps_file, "a") as f:
                f.write("==== USER ====\n")
                # f.write("Question: How do I compute the optimal Q values for this MDP instance using value iteration?\n")
                f.write("Question: Now compute the optimistic policy based on your current estimation of transition function P and reward function R.\n")
        for time in reversed(range(self.working_memory["epLen"])):
            if time == self.working_memory["epLen"]-1:
                inputs = {"time_step":self.working_memory["epLen"]-1}
                if self.write_exmps:
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: We can compute the optimistic policy by upper confidence bound value iteration, which iterates from the last time step {} back to the first time step 0. At the last time step {}, since the episode terminate at this step, the one-step lookahead is zero. Thus the Q values simply equal the immediate reward of each state-action pair plus the exploration bonus. We need to call function UpdateQbyR to add the immediate rewards and call function UpdateQbyBonus to add the exploration bonus. Then the V values can be computed by calling UpdateV, which takes maximum of the Q values over actions.\n".format(self.working_memory["epLen"]-1,self.working_memory["epLen"]-1))
                        f.write("Operation: call function UpdateQbyR with inputs {}.\n".format(inputs))
                        f.write("Result: Q values for time step {} are updated with the immediate rewards and stored in the working memory.\n".format(time))
                        f.write("Operation: call function UpdateQbyBonus with inputs {}.\n".format(inputs))
                        f.write("Result: Q values for time step {} are updated with the exploration bonus and stored in the working memory.\n".format(time))
                        f.write("Operation: call function UpdateV with inputs {}.\n".format(inputs))
                        f.write("Result: V values for time step {} are updated based on the computed Q values and stored in the working memory.\n".format(time))
                op = UpdateQbyR(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)
                op = UpdateQbyBonus(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)
                op = UpdateV(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)

            else:
                inputs = {"time_step":time}
                if self.write_exmps:
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: Now we need to compute the Q values for time step {} (we can only exit the reasoning process after computing the Q values for time step 0). Using the idea of dynamic programming, we compute the Q values in three steps. **First**, add the immediate rewards by calling function UpdateQbyR. **Second** add the one-step lookahead by calling function UpdateQbyPV. **Third**, add the exploration bonus by calling function UpdateQbyBonus. Then the V values can be computed by calling UpdateV, which takes maximum of the Q values over actions.\n".format(time))
                        f.write("Operation: call function UpdateQbyR with inputs {}.\n".format(inputs))
                        f.write("Result: Q values for time step {} are updated with the immediate rewards and stored in the working memory.\n".format(time))
                        f.write("Operation: call function UpdateQbyPV with inputs {}.\n".format(inputs))
                        f.write("Result: Q values for time step {} are updated with the one-step look ahead and stored in the working memory.\n".format(time))
                        f.write("Operation: call function UpdateQbyBonus with inputs {}.\n".format(inputs))
                        f.write("Result: Q values for time step {} are updated with the exploration bonus and stored in the working memory.\n".format(time))
                        f.write("Operation: call function UpdateV with inputs {}.\n".format(inputs))
                        f.write("Result: V values for time step {} are updated based on the computed Q values and stored in the working memory.\n".format(time))

                op = UpdateQbyR(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)
                op = UpdateQbyPV(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)
                op = UpdateQbyBonus(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)
                op = UpdateV(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)
        if self.write_exmps:
            with open(self.exmps_file, "a") as f:
                f.write("Thought: The Q values from the last time step {} to time step 0 have now been calculated. We should exit the reasoning process.\n".format(self.working_memory["epLen"]-1))

    def compute_greedy_policy(self):
        if self.write_exmps:
            with open(self.exmps_file, "a") as f:
                f.write("==== USER ====\n")
                # f.write("Question: How do I compute the optimal Q values for this MDP instance using value iteration?\n")
                f.write("Question: Now compute the optimistic policy based on your current estimation of transition function P and reward function R.\n")
        for time in reversed(range(self.working_memory["epLen"])):
            if time == self.working_memory["epLen"]-1:
                inputs = {"time_step":self.working_memory["epLen"]-1}
                if self.write_exmps:
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: We can compute the optimistic policy by upper confidence bound value iteration, which iterates from the last time step {} back to the first time step 0. At the last time step {}, since the episode terminate at this step, the one-step lookahead is zero. Thus the Q values simply equal the immediate reward of each state-action pair plus the exploration bonus. We need to call function UpdateQbyR to add the immediate rewards and call function UpdateQbyBonus to add the exploration bonus. Then the V values can be computed by calling UpdateV, which takes maximum of the Q values over actions.\n".format(self.working_memory["epLen"]-1,self.working_memory["epLen"]-1))
                        f.write("Operation: call function UpdateQbyR with inputs {}.\n".format(inputs))
                        f.write("Result: Q values for time step {} are updated with the immediate rewards and stored in the working memory.\n".format(time))
                        # f.write("Operation: call function UpdateQbyBonus with inputs {}.\n".format(inputs))
                        # f.write("Result: Q values for time step {} are updated with the exploration bonus and stored in the working memory.\n".format(time))
                        f.write("Operation: call function UpdateV with inputs {}.\n".format(inputs))
                        f.write("Result: V values for time step {} are updated based on the computed Q values and stored in the working memory.\n".format(time))
                op = UpdateQbyR(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)
                # op = UpdateQbyBonus(time_step=inputs["time_step"])
                # op.execute(working_memory=self.working_memory)
                op = UpdateV(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)

            else:
                inputs = {"time_step":time}
                if self.write_exmps:
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: Now we need to compute the Q values for time step {} (we can only exit the reasoning process after computing the Q values for time step 0). Using the idea of dynamic programming, we compute the Q values in three steps. **First**, add the immediate rewards by calling function UpdateQbyR. **Second** add the one-step lookahead by calling function UpdateQbyPV. **Third**, add the exploration bonus by calling function UpdateQbyBonus. Then the V values can be computed by calling UpdateV, which takes maximum of the Q values over actions.\n".format(time))
                        f.write("Operation: call function UpdateQbyR with inputs {}.\n".format(inputs))
                        f.write("Result: Q values for time step {} are updated with the immediate rewards and stored in the working memory.\n".format(time))
                        f.write("Operation: call function UpdateQbyPV with inputs {}.\n".format(inputs))
                        f.write("Result: Q values for time step {} are updated with the one-step look ahead and stored in the working memory.\n".format(time))
                        # f.write("Operation: call function UpdateQbyBonus with inputs {}.\n".format(inputs))
                        # f.write("Result: Q values for time step {} are updated with the exploration bonus and stored in the working memory.\n".format(time))
                        f.write("Operation: call function UpdateV with inputs {}.\n".format(inputs))
                        f.write("Result: V values for time step {} are updated based on the computed Q values and stored in the working memory.\n".format(time))

                op = UpdateQbyR(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)
                op = UpdateQbyPV(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)
                # op = UpdateQbyBonus(time_step=inputs["time_step"])
                # op.execute(working_memory=self.working_memory)
                op = UpdateV(time_step=inputs["time_step"])
                op.execute(working_memory=self.working_memory)
        if self.write_exmps:
            with open(self.exmps_file, "a") as f:
                f.write("Thought: The Q values from the last time step {} to time step 0 have now been calculated. We should exit the reasoning process.\n".format(self.working_memory["epLen"]-1))

    def move(self, state_struct):
        inputs = {"time_step": state_struct.time_step, "state": state_struct.mathematical_descript}
        if self.write_exmps:
            with open(self.exmps_file, "a") as f:
                f.write("==== USER ====\n")
                f.write(state_struct.textual_descript+"\n")
                # f.write("\nQuestion: {} Which action I should choose?\n".format(state_struct.textual_descript))
                f.write("Thought: I should retrieve the Q values for current state, which is {}, and time step, which is {}.\n".format(state_struct.mathematical_descript, state_struct.time_step))
                f.write("Operation: call function GetQ with inputs {}.\n".format(inputs))

        op = GetQ(time_step=inputs["time_step"], state=inputs["state"])
        qvalues = op.execute(working_memory=self.working_memory)
        if self.write_exmps:
            with open(self.exmps_file, "a") as f:
                f.write("Result: {}.\n".format(qvalues))
        op = GetArgMax(number_list=qvalues)
        max_indices = op.execute(working_memory=self.working_memory)
        inputs = {"number_list":qvalues.tolist()}
        if self.write_exmps:
            with open(self.exmps_file, "a") as f:
                f.write("Thought: I should call function GetArgMax to get the action indices corresponding to the maximal value in the list {}.\n".format(qvalues))
                f.write("Operation: call function GetArgMax with inputs {}.\n".format(inputs))
                f.write("Result: {}.\n".format(max_indices.tolist()))
        a = np.random.choice(np.flatnonzero(qvalues == qvalues.max()))
        if self.write_exmps:
            with open(self.exmps_file, "a") as f:
                f.write("Thought: Now I can exit the reasoning process, and choose action {}, as it maximizes the Q value (break the tie randomly if there are multiple maximums).\n".format(a))
        return a

    def update_obs(self, old_state, action, new_state, reward):
        if self.write_exmps:
            with open(self.exmps_file, "a") as f:
                f.write("==== USER ====\n")
                f.write("Question: After taking action {} at state {}, the state has transit to {} and the agent receives reward {}.\n".format(action, old_state.mathematical_descript, new_state.mathematical_descript, reward))
                f.write("Thought: I should update my estimation of the reward and transition function based on the quadruple (s={}, a={}, s_prime={}, r={}) by calling function UpdateMDPModel.\n".format(old_state.mathematical_descript, action, new_state.mathematical_descript, reward))
                inputs = {"s":old_state.mathematical_descript, "a":action, "s_prime":new_state.mathematical_descript, "r":reward}
                f.write("Action: call function UpdateMDPModel with inputs {}.\n".format(inputs))
                f.write("Result: Estimation of the transition function P and reward function R have been updated and stored in working memory.\n")
                f.write("Thought: My estimation has been updated. Now I can exit the reasoning process.\n")
        
        op = UpdateMDPModel(s=old_state.mathematical_descript, a=action, s_prime=new_state.mathematical_descript, r=reward)
        op.execute(working_memory=self.working_memory)