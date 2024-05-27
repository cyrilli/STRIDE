import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

import random
import numpy as np
from copy import deepcopy
from pydantic import BaseModel, NonNegativeInt
from typing import List

from envs.env_helper import BaseEnv

description_of_problem_class_known_mdp = """
A finite horizon tabular Markov Decision Process (MDP) is a model for decision-making in scenarios where outcomes are influenced by both randomness and controlled decisions, with decisions being made over a finite number of time steps.

Components:
State Space $S$: ${s_{0}, s_{1}, \dots, s_{|S|-1}}$, where $|S|$ is the total number of states.
Action Space $A$: ${a_{0}, a_{1}, \dots, a_{|A|-1}}$, where $|A|$ is the total number of actions.
Transition probability matrix $P$: a three-dimensional tensor with shape $|S| \times |A|\times |A|$, where each entry represents the probability of transitioning from one state after taking a specific action to another state.
Reward matrix $R$: a matrix with shape $|S| \times |A|$, where each entry gives the immediate reward received after taking an action in a state.
Horizon length $H$: The total number of time steps the decision process is constrained to.

Interaction protocol:
For time step $h=0,1,2,\dots,H$:
Agent takes an action $a_{h} \in A$ based on the current state $s_{h}$
Agent receives reward $R[s_{h},a_{h}]$
State of the environment transits to the next state $s_{h+1}$ with probability $P[s_{h},a_{h},s_{h+1}]$

Goal of the agent: 
Maximize expected cumulative rewards $E[\sum_{h=1}^{H}R[s_{h},a_{h}]]$, where the expectation is w.r.t. randomness of agent's policy and state transition.
"""

description_of_problem_class_unknown_mdp = """
A finite horizon tabular Markov Decision Process (MDP) is a model for decision-making in scenarios where outcomes are influenced by both randomness and controlled decisions, with decisions being made over a finite number of time steps.

Components:
State Space $S$: ${s_{0}, s_{1}, \dots, s_{|S|-1}}$, where $|S|$ is the total number of states.
Action Space $A$: ${a_{0}, a_{1}, \dots, a_{|A|-1}}$, where $|A|$ is the total number of actions.
Transition probability matrix $P$: a three-dimensional tensor with shape $|S| \times |A|\times |A|$, where each entry represents the probability of transitioning from one state after taking a specific action to another state.
Reward matrix $R$: a matrix with shape $|S| \times |A|$, where each entry gives the immediate reward received after taking an action in a state.
Horizon length $H$: The total number of time steps the decision process is constrained to.
Numpber of episodes $K$: The total number episodes the MDP is repeatedly played by the agent, where in each episode, the agent starts fresh, makes a series of $H$ decisions and then the episode ends. Note that learning achieved in earlier episodes influences the behavior in later episodes.
Unknown dynamics of the environment: The transition probability matrix $P$ is unknown to the agent, and the agent needs to estimate it based on the collected observations and improve its policy after each episode.

Interaction protocol:
For episode $k=0,1,2,\dots, K-1$:
For time step $h=0,1,2,\dots,H-1$:
Agent takes an action $a_{k,h} \in A$ based on the current state $s_{k,h}$
Agent receives reward $R[s_{k,h},a_{k,h}]$
State of the environment transits to the next state $s_{k,h+1}$ with probability $P[s_{k,h},a_{k,h},s_{k,h+1}]$
Agent can update its estimation of matrix $P$ based on the newly observed triplets $(s_{k,h},a_{k,h},s_{k,h+1})$ for $h=0,1,2,\dots,H-1$

Goal of the agent: 
Maximize expected cumulative rewards $E[\sum_{k=0}^{K-1}\sum_{h=0}^{H-1}R[s_{h},a_{h}]]$, where the expectation is w.r.t. randomness of agent's policy and state transition.
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

class TabularMDP(BaseEnv):
    '''
    Tabular MDP with finite horizon
    '''

    def __init__(self, env_param, mdp_known=True):
        '''
        Initialize a tabular episodic MDP

        env_param:
            nState  - int - number of states
            nAction - int - number of actions
            epLen   - int - episode length
            R      - dict - reward function
            P      - dict - transition kernel

        mdp_known: - bool - whether P and R are known to the agent
        '''

        self.name = "tabular_mdp"
        self.required_agents = ["agent"]

        self.env_param = env_param
        self.nState = env_param["nState"]
        self.nAction = env_param["nAction"]
        self.epLen = env_param["epLen"]
        self.R = env_param["R"]
        self.P = env_param["P"]
        self.mdp_known = mdp_known
        if "n_episodes" in self.env_param:
            self.n_episodes = self.env_param["n_episodes"]
        # set initial state
        self.reset()
        if self.mdp_known:
            self.description_of_problem_class = description_of_problem_class_known_mdp
        else:
            self.description_of_problem_class = description_of_problem_class_unknown_mdp

    def get_description(self, agent_role, episode_ind=None, estimated_P=None, estimated_R=None, Nsa=None, mdp_known=True, external_summarization=False):
        """
        description of the current instance of MDP
        """
        assert agent_role == "agent"
        if mdp_known:
            if self.nState <= 10 and self.nAction <= 10 and self.epLen <= 10:
                P_str = np.array2string(self.P, precision=2, floatmode='fixed')
                R_str = np.array2string(self.R[:,:,0], precision=2, floatmode='fixed')
            else:
                P_str = "stored in working memory. Full matrix is too large to be printed in context history."
                R_str = "stored in working memory. Full matrix is too large to be printed in context history."
            description = "Now you are going to play in a finite-horizon tabular Markov decision process, with length of horizon {} (with time indices starting from h=0 to {}), number of states |S|={}, number of actions |A|={}. The transition matrix P is:\n{}\nand reward matrix R is\n{}\n".format(self.env_param["epLen"], self.env_param["epLen"]-1, self.env_param["nState"], self.env_param["nAction"], P_str, R_str)
        else:
            if external_summarization:
                estimated_P_str = np.array2string(estimated_P, precision=2, floatmode='fixed')
                estimated_R_str = np.array2string(estimated_R, precision=2, floatmode='fixed')
                Nsa_str = np.array2string(Nsa, precision=2, floatmode='fixed')
                description = "Now you are going to play in a finite-horizon tabular Markov decision process, with total number of episodes K={}, length of horizon H={} (with time indices starting from h=0 to {}), number of states |S|={}, number of actions |A|={}. This is the {}-th episode you have played. Your estimated transition matrix P is\n{}\nYour estimated reward matrix R is\n{}\nYour count of visitation of all the state-action pairs is\n{}.\n".format(self.env_param["n_episodes"], self.env_param["epLen"], self.env_param["epLen"]-1, self.env_param["nState"], self.env_param["nAction"], episode_ind, estimated_P_str, estimated_R_str, Nsa_str)
            else:
                description = "Now you are going to play in a finite-horizon tabular Markov decision process, with total number of episodes K={}, length of horizon H={} (with time indices starting from h=0 to {}), number of states |S|={}, number of actions |A|={}. The transition matrix P and reward matrix R are unknown to you. This is the {}-th episode you have played.\n".format(self.env_param["n_episodes"], self.env_param["epLen"], self.env_param["epLen"]-1, self.env_param["nState"], self.env_param["nAction"], episode_ind)
        return description

    def reset(self):
        '''Reset the environment'''
        state = random.choice([s for s in range(self.nState)])
        self.state = State(time_step=0, cur_agent="agent", actions=[a for a in range(self.nAction)], action_schema=[("action", "integer", "the action chosen by the agent, which should be in {}.".format([a for a in range(self.nAction)]))], textual_descript="This is time step {}, the current state is {}, and the available actions are {}.\nQuestion: Now which action the agent should take?".format(0, state, [a for a in range(self.nAction)]), mathematical_descript=0)
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
        if self.R[self.state.mathematical_descript, action][1] < 1e-9:
            # Hack for no noise
            mean_reward = deepcopy(self.R[self.state.mathematical_descript, action][0])
            reward = mean_reward
        else:
            mean_reward = deepcopy(self.R[self.state.mathematical_descript, action][0])
            reward = np.random.normal(loc=mean_reward,
                                      scale=self.R[self.state.mathematical_descript, action][1])
        newState = np.random.choice(self.nState, p=self.P[self.state.mathematical_descript, action])

        # Update the state of the environment
        self.state.time_step += 1
        self.state.mathematical_descript = newState
        self.state.textual_descript = "This is time step {}, the current state is {}, and the available actions are {}.\nQuestion: Now which action the agent should take?".format(self.state.time_step, newState, [a for a in range(self.nAction)])

        if self.state.time_step == self.epLen:
            self.is_done = True

        return self.state, round(reward, 2)

    def compute_qVals(self):
        '''
        Compute the Q values for the environment

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
                    q[time, s, a] = self.R[s, a][0]

            for s in range(self.nState):
                for a in range(self.nAction):
                    temp = 0.0
                    for s_prime in range(self.nState):
                        temp += self.P[s, a, s_prime] * v[time+1, s_prime]
                    q[time, s, a] += temp

            for s in range(self.nState):
                v[time, s] = np.max(q[time, s])

        return q, v
    
    def evaluate_policy(self, p_q):
        v = np.zeros((self.epLen+1,self.nState))

        for time in reversed(range(self.epLen)):
            for s in range(self.nState):
                # action taken by the current policy
                a = np.random.choice(np.flatnonzero(p_q[time, s] == p_q[time, s].max()))
                
                temp = 0.0
                for s_prime in range(self.nState):
                    temp += self.P[s, a, s_prime] * v[time+1, s_prime]
                v[time, s] = self.R[s, a][0] + temp
        return v