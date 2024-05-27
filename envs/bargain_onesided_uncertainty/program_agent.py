from envs.bargain_alternate_singleissue.tools import CalcUtil
from tools import *

import numpy as np
import random
from copy import deepcopy

class SellerAgent:
    def __init__(self, working_memory, exmps_file) -> None:
        self.working_memory = working_memory
        self.exmps_file = exmps_file
        self.b = 1.0 # initial belief about buyer value is [0,1]

    def move(self, state):
        t = state.time_step
        assert state.cur_agent == "seller"
        with open(self.exmps_file, "a") as f:
            f.write("==== USER ====\n")
            f.write("Question: {}\n".format(state.textual_descript))
        epsilon = 0.01

        if state.time_step == 1:
            low = 0
            high = 1
            
            b_Tminus1_guess = (high + low)/2.0

            inputs = {"time_step":t, "b_last": b_Tminus1_guess}
            print("Thought: To decide my price at the current time step {}, I need to first compute my belief about buyer's value at time step T-1 under sequential equilbrium. I can achieve this via bisection search that terminates when I get close enough (the difference is smaller than {}) to the actual value of my current belief {}. Buyer's value is in the range of [0,1], so I should start with (0+1)/2=0.5. Then I should call function ComputeBt to compute what my current belief about buyer's value would be, assuming that my belief at time step T-1 is 0.5.".format(t, epsilon, self.b))
            with open(self.exmps_file, "a") as f:
                f.write("Thought: To decide my price at the current time step {}, I need to first compute my belief about buyer's value at time step T-1 under sequential equilbrium. I can achieve this via bisection search that terminates when I get close enough (the difference is smaller than {}) to the actual value of my current belief {}. Buyer's value is in the range of [0,1], so I should start with (0+1)/2=0.5. Then I should call function ComputeBt to compute what my current belief about buyer's value would be, assuming that my belief at time step T-1 is 0.5.".format(t, epsilon, self.b))
                f.write("Operation: Call function ComputeBt with inputs {}.\n".format(inputs))

            op = ComputeBt(time_step=t, b_last=b_Tminus1_guess)
            b_0_text = op.execute(working_memory=self.working_memory)
            b_0 = self.get_b_0(index_cur_b=t-1, b_Tminus1=b_Tminus1_guess)
            assert abs(float(b_0_text.split()[-1])-b_0) <= 1e-2

            # print("Result: My current belief about buyer's value would be {}.".format(b_0))
            print("Result: {}".format(b_0_text))
            with open(self.exmps_file, "a") as f:
                f.write("Result: {}\n".format(b_0_text))

            iter = 1
            while abs(b_0-self.b)>=epsilon:
                if b_0<self.b:
                    low = b_Tminus1_guess
                    print("Thought: Since the computed value of my current belief {} is smaller than the actual value of my current belief {}. In next iteration, I should focus on the upper half of buyer's value range [{}, {}], and try ({}+{})/2.0={}.".format(b_0, self.b, low, high, low, high, (low+high)/2.0))
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: Since the computed value of my current belief {} is smaller than the actual value of my current belief {}. In next iteration, I should focus on the upper half of buyer's value range [{}, {}], and try ({}+{})/2.0={}.\n".format(b_0, self.b, low, high, low, high, (low+high)/2.0))
                else:
                    high = b_Tminus1_guess
                    print("Thought: Since the computed value of my current belief {} is larger than the actual value of my current belief {}. In next iteration, I should focus on the lower half of buyer's value range [{}, {}], and try ({}+{})/2.0={}.".format(b_0, self.b, low, high, low, high, (low+high)/2.0))
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: Since the computed value of my current belief {} is larger than the actual value of my current belief {}. In next iteration, I should focus on the lower half of buyer's value range [{}, {}], and try ({}+{})/2.0={}.\n".format(b_0, self.b, low, high, low, high, (low+high)/2.0))
                b_Tminus1_guess = (high + low)/2

                inputs = {"time_step":t, "b_last": b_Tminus1_guess}
                print("Operation: Call function ComputeBt with inputs {}.".format(inputs))
                with open(self.exmps_file, "a") as f:
                    f.write("Operation: Call function ComputeBt with inputs {}.\n".format(inputs))

                op = ComputeBt(time_step=t, b_last=b_Tminus1_guess)
                b_0_text = op.execute(working_memory=self.working_memory)
                b_0 = self.get_b_0(index_cur_b=t-1, b_Tminus1=b_Tminus1_guess)
                
                print("Result: {}".format(b_0_text))
                with open(self.exmps_file, "a") as f:
                    f.write("Result: {}\n".format(b_0_text))
                assert abs(float(b_0_text.split()[-1])-b_0) <= 1e-2
                iter += 1

            print("Thought: Since |{}-{}| < {}, the computed value of my current belief is close enough to the actual value now. Therefore, {} is an accurate approximation of my belief about buyer's value at time step T-1.".format(b_0, self.b, epsilon, b_Tminus1_guess))
            with open(self.exmps_file, "a") as f:
                f.write("Thought: Since |{}-{}| < {}, the computed value of my current belief is close enough to the actual value now. Therefore, {} is an accurate approximation of my belief about buyer's value at time step T-1.\n".format(b_0, self.b, epsilon, b_Tminus1_guess))
            # print("Number of iterations needed is {}".format(iter))

            # now we have a good approximation of b_Tminus1
            # we can compute seller's price at current time step t
            print("Thought: Now I can start reasoning backward from the last time step {} to the current time step {} to compute the price I should offer at current time step {}.".format(self.working_memory["T"], t, t))
            with open(self.exmps_file, "a") as f:
                f.write("Thought: Now I can start reasoning backward from the last time step {} to the current time step {} to compute the price I should offer at current time step {}.\n".format(self.working_memory["T"], t, t))


            t_iter = self.working_memory["T"]
            while t_iter >= t:
                if t_iter == self.working_memory["T"]:
                    # b_t = None
                    # b_t_minus_1 = deepcopy(b_Tminus1_guess)
                    # p_t = 0.5 * b_t_minus_1
                    # u_t = 0.25 * b_t_minus_1**2 / b_t_minus_1

                    op = SolveLast(b_last=b_Tminus1_guess)
                    ut_pt_text = op.execute(working_memory=self.working_memory)
                    # assert abs(float(ut_pt_text.split()[-7])-u_t) <= 1e-2
                    # assert abs(float(ut_pt_text.split()[-1])-p_t) <= 1e-2
                    u_t = float(ut_pt_text.split()[-7])
                    p_t = float(ut_pt_text.split()[-1])
                    print("Operation: Call function SOLVELAST with inputs {}.".format({"b_last": b_Tminus1_guess}))
                    print("Result: {}".format(ut_pt_text))
                    with open(self.exmps_file, "a") as f:
                        f.write("Operation: Call function SOLVELAST with inputs {}.\n".format({"b_last": b_Tminus1_guess}))
                        f.write("Result: {}".format(ut_pt_text))
                else:
                    inputs = {"u_t":u_t, "p_t":p_t, "t":t_iter}
                    op = Solve(u_t=u_t, p_t=p_t, t=t_iter)
                    ut_pt_text = op.execute(working_memory=self.working_memory)
                    u_t = float(ut_pt_text.split()[-7])
                    p_t = float(ut_pt_text.split()[-1])

                    print("Thought: Now I need to continue to time step {}.".format(t_iter))
                    print("Operation: Call function SOLVE with inputs {}.".format(inputs))
                    print("Result: {}".format(ut_pt_text))
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: Now I need to continue to time step {}.\n".format(t_iter))
                        f.write("Operation: Call function SOLVE with inputs {}.\n".format(inputs))
                        f.write("Result: {}\n".format(ut_pt_text))
                t_iter -= 1

            print("Thought: I have reached the current time step {}. Offering the price of {} would maximize my utility, so I can exit reasoning now.".format(t, p_t))
            with open(self.exmps_file, "a") as f:
                f.write("Thought: I have reached the current time step {}. Offering the price of {} would maximize my utility, so I can exit reasoning now.\n".format(t, p_t))

            self.b = deepcopy(self.working_memory["b_list"][t]) # update belief (though seller does not know whether buyer will accept yet)
            return p_t
        else:
            assert self.working_memory["SEPrice"] != {}
            op = GetSEPrice(time_step=state.time_step)
            se_price = op.execute(working_memory=self.working_memory)

            with open(self.exmps_file, "a") as f:
                print("Thought: Assuming my opponent is rational, then to maximize my expected utility, I should adopt the SE strategy. Therefore, I should retrive the computed SE price for the current time step {} by calling GetSEPrice.".format(state.time_step))
                f.write("Thought: Assuming my opponent is rational, then to maximize my expected utility, I should adopt the SE strategy. Therefore, I should retrive the computed SE price for the current time step {} by calling GetSEPrice.\n".format(state.time_step))
                inputs = {"time_step":state.time_step}
                print("Operation: call function GetSEPrice with inputs {}.".format(inputs))
                print("Result: {}".format(se_price))
                print("Thought: I can exit the reasoning process and propose a price of {}, which is my SE strategy at time step {}.".format(se_price, state.time_step))
                f.write("Operation: call function GetSEPrice with inputs {}.\n".format(inputs))
                f.write("Result: {}\n".format(se_price))
                se_price = float(se_price.split()[-1])
                f.write("Thought: I can exit the reasoning process and propose a price of {}, which is my SE strategy at time step {}.\n".format(se_price, state.time_step))
            return se_price


    def get_b_0(self, index_cur_b, b_Tminus1):
        t = self.working_memory["T"]-1
        b = b_Tminus1
        while t >= (index_cur_b+1):
            b = (2*(1-self.working_memory["delta_b"]+self.working_memory["delta_b"]*self.working_memory["c"][t+1])-self.working_memory["delta_s"]*self.working_memory["c"][t+1]) / (1-self.working_memory["delta_b"]+self.working_memory["delta_b"]*self.working_memory["c"][t+1]) * b
            t = t-1
        return b

class BuyerAgent:
    def __init__(self, working_memory, exmps_file) -> None:
        self.working_memory = working_memory
        self.exmps_file = exmps_file

        self.b = 1.0 # seller's initial belief about buyer value is [0,1]

    def move(self, state):
        assert state.cur_agent == "buyer"
        price = state.mathematical_descript[-1]
        t = state.time_step
        with open(self.exmps_file, "a") as f:
            f.write("==== USER ====\n")
            f.write("Question: {}\n".format(state.textual_descript))

        print("Thought: To make this decision, I need to first reason about the utility I can get by accepting the offer now, and then the utility I can get by rejecting it and waiting for the next time step.")
        with open(self.exmps_file, "a") as f:
            f.write("Thought: To make this decision, I need to first reason about the utility I can get by accepting the offer now, and the utility I can get by rejecting it and waiting for the next time step.\n")


        # reason about utility of accepting
        op = CalcUtil(agent="buyer", price=price, time_step=t)
        u_a = self.working_memory["delta_b"]**(t-1) * (self.working_memory["b_value"] - price)
        u_a_text = op.execute(working_memory=self.working_memory)
        assert abs(u_a - float(u_a_text.split()[-1])) <= 1e-2
        inputs = {"agent":"buyer", "price": price, "time_step":t}
        print("Operation: Call function CalcUtil with inputs {}.".format(inputs))
        print("Result: {}".format(u_a_text))
        with open(self.exmps_file, "a") as f:
            f.write("Operation: Call function CalcUtil with inputs {}.\n".format(inputs))
            f.write("Result: {}\n".format(u_a_text))

        # reason about utility of rejecting
        if t == 1:
            epsilon = 0.01

            low = 0
            high = 1
            b_Tminus1_guess = (high + low)/2.0

            inputs = {"b_last": b_Tminus1_guess}
            print("Thought: Now, to know the utility I can get by rejecting the offer at current time step {} and waiting for seller's new offer at time step {}, I need to first compute what seller's belief at time step T-1 about my value would be. I can achieve this via bisection search that terminates when I get close enough (the difference is smaller than {}) to the actual value of seller's current belief {}. My value is in the range of [0,1], so I should start with (0+1)/2=0.5. Then I should call function ComputeBt to compute what seller's current belief about my value would be, assuming that seller's belief at time step T-1 is 0.5.".format(t, t+1, epsilon, self.b))
            with open(self.exmps_file, "a") as f:
                f.write("Thought: Now, to know the utility I can get by rejecting the offer at current time step {} and waiting for seller's new offer at time step {}, I need to first compute what seller's belief at time step T-1 about my value would be. I can achieve this via bisection search that terminates when I get close enough (the difference is smaller than {}) to the actual value of seller's current belief {}. My value is in the range of [0,1], so I should start with (0+1)/2=0.5. Then I should call function ComputeBt to compute what seller's current belief about my value would be, assuming that seller's belief at time step T-1 is 0.5.".format(t, t+1, epsilon, self.b))
            print("Operation: Call function ComputeBt with inputs {}.".format(inputs))
            with open(self.exmps_file, "a") as f:
                f.write("Operation: Call function ComputeBt with inputs {}.\n".format(inputs))

            op = ComputeBt(time_step=t, b_last=b_Tminus1_guess)
            b_0_text = op.execute(working_memory=self.working_memory)
            b_0 = self.get_b_0(index_cur_b=t-1, b_Tminus1=b_Tminus1_guess)
            assert abs(float(b_0_text.split()[-1])-b_0) <= 1e-2

            print("Result: {}".format(b_0_text))
            with open(self.exmps_file, "a") as f:
                f.write("Result: {}\n".format(b_0_text))

            iter = 1
            while abs(b_0-self.b)>=epsilon:
                if b_0<self.b:
                    low = b_Tminus1_guess
                    print("Thought: Since the computed value of seller's current belief {} is smaller than the actual value of seller's current belief {}. In next iteration, I should focus on the upper half of my value range [{}, {}], and try ({}+{})/2.0={}.".format(b_0, self.b, low, high, low, high, (low+high)/2.0))
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: Since the computed value of seller's current belief {} is smaller than the actual value of seller's current belief {}. In next iteration, I should focus on the upper half of my value range [{}, {}], and try ({}+{})/2.0={}.\n".format(b_0, self.b, low, high, low, high, (low+high)/2.0))
                else:
                    high = b_Tminus1_guess
                    print("Thought: Since the computed value of seller's current belief {} is larger than the actual value of seller's current belief {}. In next iteration, I should focus on the lower half of my value range [{}, {}], and try ({}+{})/2.0={}.".format(b_0, self.b, low, high, low, high, (low+high)/2.0))
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: Since the computed value of seller's current belief {} is larger than the actual value of seller's current belief {}. In next iteration, I should focus on the lower half of my value range [{}, {}], and try ({}+{})/2.0={}.\n".format(b_0, self.b, low, high, low, high, (low+high)/2.0))

                b_Tminus1_guess = (high + low)/2
                inputs = {"time_step":t, "b_last": b_Tminus1_guess}
                print("Operation: Call function ComputeBt with inputs {}.".format(inputs))
                with open(self.exmps_file, "a") as f:
                    f.write("Operation: Call function ComputeBt with inputs {}.\n".format(inputs))
                op = ComputeBt(time_step=t, b_last=b_Tminus1_guess)
                b_0_text = op.execute(working_memory=self.working_memory)
                b_0 = self.get_b_0(index_cur_b=t-1, b_Tminus1=b_Tminus1_guess)
                print("Result: {}".format(b_0_text))
                with open(self.exmps_file, "a") as f:
                    f.write("Result: {}\n".format(b_0_text))
                assert abs(float(b_0_text.split()[-1])-b_0) <= 1e-2
                iter += 1

            print("Thought: Since |{}-{}| < {}, the computed value of seller's current belief is close enough to the actual value now. Therefore, {} is an accurate approximation of seller's belief about my value at time step T-1.".format(b_0, self.b, epsilon, b_Tminus1_guess))
            with open(self.exmps_file, "a") as f:
                f.write("Thought: Since |{}-{}| < {}, the computed value of seller's current belief is close enough to the actual value now. Therefore, {} is an accurate approximation of seller's belief about my value at time step T-1.\n".format(b_0, self.b, epsilon, b_Tminus1_guess))

            # now we have a good approximation of b_Tminus1
            # we can compute seller's price at current time step t

            t_iter = self.working_memory["T"]
            while t_iter >= t+1:
                if t_iter == self.working_memory["T"]:
                    # b_t = None
                    # b_t_minus_1 = deepcopy(b_Tminus1_guess)
                    # p_t = 0.5 * b_t_minus_1
                    # u_t = 0.25 * b_t_minus_1**2 / b_t_minus_1

                    op = SolveLast(b_last=b_Tminus1_guess)
                    ut_pt_text = op.execute(working_memory=self.working_memory)
                    # assert abs(float(ut_pt_text.split()[-7])-u_t) <= 1e-2
                    # assert abs(float(ut_pt_text.split()[-1])-p_t) <= 1e-2
                    u_t = float(ut_pt_text.split()[-7])
                    p_t = float(ut_pt_text.split()[-1])
                    print("Operation: Call function SOLVELAST with inputs {}.".format({"b_last": b_Tminus1_guess}))
                    print("Result: {}".format(ut_pt_text))
                    with open(self.exmps_file, "a") as f:
                        f.write("Operation: Call function SOLVELAST with inputs {}.\n".format({"b_last": b_Tminus1_guess}))
                        f.write("Result: {}".format(ut_pt_text))
                else:
                    inputs = {"u_t":u_t, "p_t":p_t, "t":t_iter}
                    op = Solve(u_t=u_t, p_t=p_t, t=t_iter)
                    ut_pt_text = op.execute(working_memory=self.working_memory)
                    u_t = float(ut_pt_text.split()[-7])
                    p_t = float(ut_pt_text.split()[-1])

                    print("Thought: Now I need to continue to time step {}.".format(t_iter))
                    print("Operation: Call function SOLVE with inputs {}.".format(inputs))
                    print("Result: {}".format(ut_pt_text))
                    with open(self.exmps_file, "a") as f:
                        f.write("Thought: Now I need to continue to time step {}.\n".format(t_iter))
                        f.write("Operation: Call function SOLVE with inputs {}.\n".format(inputs))
                        f.write("Result: {}\n".format(ut_pt_text))
                t_iter -= 1

            print("Thought: I have reached the next time step {}. Seller will offer a price of {} in the next time step. Now I need to calculate my utility if I accept this price at the next time step {}.".format(t+1, p_t, t+1))
            with open(self.exmps_file, "a") as f:
                f.write("Thought: I have reached the next time step {}. Seller will offer a price of {} in the next time step. Now I need to calculate my utility if I accept this price at the next time step {}.\n".format(t+1, p_t, t+1))

            u_r = self.working_memory["delta_b"]**(t) * (self.working_memory["b_value"] - p_t)
            op = CalcUtil(agent="buyer", price=p_t, time_step=t+1)
            u_r_text = op.execute(working_memory=self.working_memory)

            # print("============")
            # print(u_r)
            # print(u_r_text)

            assert abs(u_r - float(u_r_text.split()[-1])) <= 1e-2
            inputs = {"agent":"buyer", "price": p_t, "time_step":t+1}
            print("Operation: Call function CalcUtil with inputs {}.".format(inputs))
            print("Result: {}".format(u_r_text))
            with open(self.exmps_file, "a") as f:
                f.write("Operation: call function CalcUtil with inputs {}.\n".format(inputs))
                f.write("Result: {}\n".format(u_r_text))
        
        elif t > 1 and t < self.working_memory["T"]:
            # then asses utility of rejecting the offer
            assert self.working_memory["SEPrice"] != {}
            op = GetSEPrice(time_step=state.time_step+1)
            se_price = op.execute(working_memory=self.working_memory)
            op = CalcUtil(agent=state.cur_agent, price=float(se_price.split()[-1]), time_step=state.time_step+1)
            u_r = op.execute(working_memory=self.working_memory)

            with open(self.exmps_file, "a") as f:
                print("Thought: Then I should reason about the utility I can get by rejecting seller's current offer and waiting for the next time step {}. Since I have already computed the sequential equilirbium strategy and saved it in the working memory, I can retrive the previously computed price for the next time step {} by calling GetSEPrice. Then I can compute my utility by calling CalcUtil.".format(state.time_step+1, state.time_step+1))
                f.write("Thought: Then I should reason about the utility I can get by rejecting seller's current offer and waiting for the next time step {}. Since I have already computed the sequential equilirbium strategy and saved it in the working memory, I can retrive the previously computed price for the next time step {} by calling GetSEPrice. Then I can compute my utility by calling CalcUtil.\n".format(state.time_step+1, state.time_step+1))
                inputs = {"time_step":state.time_step+1}
                print("Operation: call function GetSEPrice with inputs {}.".format(inputs))
                print("Result: {}".format(se_price))
                f.write("Operation: call function GetSEPrice with inputs {}.\n".format(inputs))
                f.write("Result: {}\n".format(se_price))
                inputs = {"agent":state.cur_agent, "price": float(se_price.split()[-1]), "time_step":state.time_step+1}
                print("Operation: call function CalcUtil with inputs {}.".format(inputs))
                print("Result: {}".format(u_r))
                f.write("Operation: call function CalcUtil with inputs {}.\n".format(inputs))
                f.write("Result: {}\n".format(u_r))
            u_r = float(u_r.split()[-1])
        else:
            # last time step
            print("Thought: Since time step {} is the last time step, by rejecting seller's offer I will get utility 0.".format(t))
            with open(self.exmps_file, "a") as f:
                f.write("Thought: Since {} is the last time step, by rejecting seller's offer I will get utility 0.\n".format(t))
            u_r = 0.0

        if u_a >= u_r:
            print("Thought: Since the utility of accepting seller's offer at the current time step {}, which is {}, is greater or equal to the utility of waiting for seller's offer at next time step {}, which is {}, I can exit reasoning process and choose to accept.".format(t,u_a, t+1, u_r))
            with open(self.exmps_file, "a") as f:
                f.write("Thought: Since the utility of accepting seller's offer at the current time step {}, which is {}, is greater or equal to the utility of waiting for seller's offer at next time step {}, which is {}, I can exit reasoning process and choose to accept.\n".format(t,u_a, t+1, u_r))

            return "accept"
        else:
            print("Thought: Since the utility of accepting seller's offer at the current time step {}, which is {}, is less than the utility of waiting for seller's offer at next time step {}, which is {}, I can exit reasoning process and choose to reject.".format(t,u_a, t+1, u_r))
            with open(self.exmps_file, "a") as f:
                f.write("Thought: Since the utility of accepting seller's offer at the current time step {}, which is {}, is less than the utility of waiting for seller's offer at next time step {}, which is {}, I can exit reasoning process and choose to reject.\n".format(t,u_a, t+1, u_r))

            #TODO: update this
            # b_t, _, _ = self.get_p_u_t(index_cur_b=t-1, t_target=t)
            self.b = deepcopy(self.working_memory["b_list"][t]) # buyer knows seller's belief will be updated since buyer rejects
            return "reject"

    def get_b_0(self, index_cur_b, b_Tminus1):
        t = self.working_memory["T"]-1
        b = b_Tminus1
        while t >= (index_cur_b+1):
            b = (2*(1-self.working_memory["delta_b"]+self.working_memory["delta_b"]*self.working_memory["c"][t+1])-self.working_memory["delta_s"]*self.working_memory["c"][t+1]) / (1-self.working_memory["delta_b"]+self.working_memory["delta_b"]*self.working_memory["c"][t+1]) * b
            t = t-1
        return b