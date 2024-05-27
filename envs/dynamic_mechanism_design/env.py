import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

import random
import numpy as np
from copy import deepcopy
from pydantic import BaseModel, Field, NonNegativeInt
from typing import List, Optional

from envs.env_helper import BaseEnv

description_of_problem_class_known_mdp = """
The dynamic mechanism design problem involves creating allocation and pricing rules for decision-making, where the value of resource to the agents changes over time as the state of the environment changes.

Components:
Players: a mechanism designer and a set of $N$ agents
State Space $S$: ${s_{0}, s_{1}, \dots, s_{|S|-1}}$, where $|S|$ is the total number of states.
Action Space $A$: ${a_{0}, a_{1}, \dots, a_{|A|-1}}$, where $|A|$ is the total number of actions. Each action represents the mechanism designer's allocation of some scarce resource among $N$ agents.
Transition probability matrix $P$: a three-dimensional tensor with shape $|S| \times |A|\times |A|$, where each entry represents the probability of transitioning from one state after taking a specific action to another state.
Reward matrix $R$: a three-dimensional tensor with shape $N \times |S| \times |A|$, where each matrix $R[i,:,:]$ represents the reward matrix of an agent $i$ for $i=1,2,\dots,N$, and each of its entry gives the immediate reward received by agent $i$ after the mechanism designer takes an action in a state.
Horizon length $H$: The total number of time steps the decision process is constrained to.

Interaction protocol:
Before the interaction starts, each agent $i$ reports a reward matrix (can be different from its true reward matrix $R[i,:,:]$), denoted as $\tilde{R}[i,:,:]$, to the designer. Based on agents' reported reward matrix, the designer chooses a policy $\pi: S \rightarrow \Delta(A)$ and prices $\{p_{i}\}_{i=1}^{N}$ to be charged to each agent.
For time step $h=0,1,2,\dots,H-1$:
Mechanism designer takes an action $a_{h} \sim \pi(s_{h})$ based on the policy $\pi$ and the current state $s_{h}$
Each agent $i$ receives reward $R[i,s_{h},a_{h}]$ for $i=1,2\dots,N$
The environment transits to the next state $s_{h+1}$ with probability $P[s_{h},a_{h},s_{h+1}]$
After the interaction, the mechanism designer charges each agent $i$ with some price $p_{i}$

Goal of the agents:
Each agent wants to maximize its utility $u_{i}=E[\sum_{h=1}^{H}R[i,s_{h},a_{h}]]-p_{i}$, that is, the difference between the expected cumulative rewards, where the expectation is w.r.t. randomness of designer's policy and state transition, and the price charged by the mechanism designer. As the agents cannot directly take actions, their only leverage is to decide whether to truthfully report their reward matrix to the designer.

Goal of the mechanism designer:
Maximize the expected cumulative rewards of all agents $E[\sum_{i=1}^{N}\sum_{h=1}^{H}R[i, s_{h},a_{h}]]$, where the expectation is w.r.t. randomness of designer's policy and state transition. As the designer only observes agents' reported reward matrix $\tilde{R}$, to fulfil its objective, the designer needs to guarantee, with its policy and pricing strategy, no agent $i$ has incentive to report $\tilde{R}[i,:,:]$ that is different from the true reward matrix $R[i,:,:]$ unilaterally.

It is known that VCG mechanism guarantees truthfulnes of the agents, and uniquely maximizes the objective. It is defined as follows:
$$\pi^{\star} = \argmax_{\pi} E_{\pi,P}[\sum_{i=1}^{N}\sum_{h=1}^{H}\tilde{R}[i, s_{h},a_{h}]]$$
$$p^{\star}_{i} = E_{\pi^{\star}_{-i},P}[\sum_{j \neq i}\sum_{h=1}^{H}\tilde{R}[j, s_{h},a_{h}]] - E_{\pi^{\star},P}[\sum_{j \neq i}\sum_{h=1}^{H}\tilde{R}[j, s_{h},a_{h}]]$$
for $i=1,2,\dots,N$, where $\pi^{\star}_{-i} = \argmax_{\pi} E_{\pi,P}[\sum_{j \neq i}\sum_{h=1}^{H}\tilde{R}[j, s_{h},a_{h}]]$ is the optimal policy for a MDP with transition probability matrix P and reward matrix $\sum_{j \neq i} \tilde{R}[j,:,:]$, that is, excluding the reward matrix of agent $i$ itself.

Now as a strategic decision maker, your job is to compute the VCG mechanism based on the given transition probability matrix $P$ and the reward matrix $R$ reported by the agents. Then you should take an action at each time step and charges prices to each agent at the end, according to your computed VCG mechanism.
"""

class State(BaseModel):
    time_step: int
    cur_agent: str
    actions: List[NonNegativeInt]
    action_schema: List[tuple]
    textual_descript: str
    mathematical_descript: int

    def is_valid_action(self, action):
        return action in self.actions

class DynamicMechanismDesign(BaseEnv):
    '''
    Tabular MDP

    R - dict by (s,a) - each R[s,a] = (meanReward, sdReward)
    P - dict by (s,a) - each P[s,a] = transition vector size S
    '''

    def __init__(self, env_param):
        '''
        Initialize a tabular episodic MDP

        Args:
            nState  - int - number of states
            nAction - int - number of actions
            epLen   - int - episode length
            R      - dict - reward function
            P      - dict - transition kernel
        '''

        self.name = "dynamic_mechanism_design"
        self.required_agents = ["designer"]

        self.env_param = env_param
        self.nState = env_param["nState"]
        self.nAction = env_param["nAction"]
        self.epLen = env_param["epLen"]
        self.nAgent = env_param["nAgent"]
        self.R = env_param["R"] # now a three-way tensor
        self.P = env_param["P"]
        # set initial state
        self.reset()
        self.description_of_problem_class = description_of_problem_class_known_mdp

    def get_description(self, agent_role):
        """
        description of the current instance of dynamic mechanism design problem
        """
        assert agent_role == "designer"
        if self.nState <= 10 and self.nAction <= 10 and self.epLen <= 10:
            P_str = np.array2string(self.P, precision=2, floatmode='fixed')
            R_str = np.array2string(self.R[:,:,:,0], precision=2, floatmode='fixed')
        else:
            P_str = "stored in working memory. Full matrix is too large to be printed in context history."
            R_str = "stored in working memory. Full matrix is too large to be printed in context history."
        description = "Now you are going to play in a finite-horizon dynamic mechanism design problem, with number of agents N={}, length of horizon {} (with time indices starting from h=0 to {}), number of states |S|={}, number of actions |A|={}. The transition matrix P is:\n{}\nand reward matrix R reported by the agents is\n{}\n".format(self.env_param["nAgent"], self.env_param["epLen"], self.env_param["epLen"]-1, self.env_param["nState"], self.env_param["nAction"], P_str, R_str)
        return description

    def reset(self):
        '''Reset the environment'''
        # initial state is always 0
        state = 0 #random.choice([s for s in range(self.nState)])
        self.state = State(time_step=0, cur_agent="designer", actions=[a for a in range(self.nAction)], action_schema=[("action", "integer", "the action chosen by the mechanism designer, which should be in {}.".format([a for a in range(self.nAction)]))], textual_descript="This is time step {}, the current state is {}, and the available actions are {}.\nQuestion: Now which action the mechanism designer should take?".format(0, state, [a for a in range(self.nAction)]), mathematical_descript=0)
        self.is_done = False

    def step(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action

        Returns:
        reward - double - reward
        newState - int - new state
        pContinue - 0/1 - flag for end of the episode
        '''
        # the largest variance across all agents' reward distribution
        if np.max(self.R[:,self.state.mathematical_descript, action,1]) < 1e-9:
            # Hack for no noise
            mean_reward = deepcopy(np.sum(self.R[:,self.state.mathematical_descript, action,0]))
            reward = mean_reward
        else:
            # print("=======================================")
            mean_reward = deepcopy(np.sum(self.R[:,self.state.mathematical_descript, action,0]))
            reward = np.random.normal(loc=mean_reward,
                                      scale=np.sqrt(np.sum(np.square(self.R[:,self.state.mathematical_descript, action,1]))))
        newState = np.random.choice(self.nState, p=self.P[self.state.mathematical_descript, action])

        # Update the state of the environment
        self.state.time_step += 1
        self.state.mathematical_descript = newState
        self.state.textual_descript = "This is time step {}, the current state is {}, and the available actions are {}.\nQuestion: Now which action the mechanism designer should take?".format(self.state.time_step, newState, [a for a in range(self.nAction)])

        if self.state.time_step == self.epLen:
            self.is_done = True

        return self.state, round(reward, 2)

    def compute_qVals(self, agent_to_exclude=None):
        '''
        Compute the optimal Q values for the environment

        Returns:
            q - q[timestep, state] is vector of Q values for each action
            v - v[timestep] is the vector of optimal values at timestep
        '''
        # qVals = {}
        # qMax = {}

        v = np.zeros((self.epLen+1,self.nState))
        q = np.zeros((self.epLen,self.nState,self.nAction))

        for time in reversed(range(self.epLen)):
            for s in range(self.nState):
                for a in range(self.nAction):
                    if agent_to_exclude is None:
                        q[time, s, a] = np.sum(self.R[:, s, a, 0]) # if not agent is excluded, sum them all up
                    else:
                        q[time, s, a] = np.sum(self.R[:agent_to_exclude, s, a, 0]) + np.sum(self.R[agent_to_exclude+1:, s, a, 0])

            for s in range(self.nState):
                for a in range(self.nAction):
                    temp = 0.0
                    for s_prime in range(self.nState):
                        temp += self.P[s, a, s_prime] * v[time+1, s_prime]
                    q[time, s, a] += temp

            for s in range(self.nState):
                v[time, s] = np.max(q[time, s])

        return q, v
    
    def evaluate_policy(self, p_q, agent_to_exclude=None):
        v = np.zeros((self.epLen+1,self.nState))

        for time in reversed(range(self.epLen)):
            for s in range(self.nState):
                # action taken by the current policy
                a = np.random.choice(np.flatnonzero(p_q[time, s] == p_q[time, s].max()))
                
                temp = 0.0
                for s_prime in range(self.nState):
                    temp += self.P[s, a, s_prime] * v[time+1, s_prime]
                if agent_to_exclude is None:
                    v[time, s] = np.sum(self.R[:, s, a, 0]) + temp
                else:
                    # print("~~~~~~~~~")
                    # print(agent_to_exclude)
                    # print(self.R[:agent_to_exclude, s, a, 0])
                    # print(self.R[agent_to_exclude+1:, s, a, 0])
                    v[time, s] = np.sum(self.R[:agent_to_exclude, s, a, 0]) + np.sum(self.R[agent_to_exclude+1:, s, a, 0]) + temp
        return v