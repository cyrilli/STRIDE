from rich.console import Console
from envs.env_helper import get_env_param
from agents.StriDe import StriDeAgent
import re
import numpy as np
from copy import deepcopy
import json
import random

class Logger(object):
    def __init__(self, log_file, verbose=True, writeToFile=True):
        # self.terminal = sys.stdout
        self.console = Console(record=True)
        # self.log = open(log_file, "w")
        self.log_file = log_file
        self.verbose = verbose
        self.writeToFile = writeToFile

        self.write("All outputs written to %s" % log_file)
        return 

    def write(self, message, color=None):
        if self.writeToFile:
            self.console.save_html(self.log_file, clear=False)
        if(self.verbose): 
            if color is not None:
                self.console.print("[{}]".format(color)+message+"[/{}]".format(color))
            else:
                self.console.print(message)

def load_initial_instructions(path_to_instructions):
    """Load initial instructions from textual format to a python dict"""
    pattern = r"==== (SYSTEM|USER|ASSISTANT) ===="

    # Use re.split to split the string by the pattern
    with open(path_to_instructions) as f:
        content = f.read()
        content = re.split(pattern, content)
        content_ = []
        for c in content: 
            if(c != ""): content_.append(c)
        content = content_
        l = len(content)
        assert(l % 2 == 0)
        initial_instruction = []
        for i in range(0, l, 2):
            instruction = {"role": content[i].strip().lower().replace("====", "").replace(" ", "").strip(), 
                           "content": content[i+1].strip()
                           }
            initial_instruction.append(instruction)
    return initial_instruction

def create_env(env_name, random_param, mdp_known):
    if env_name == "tabular_mdp":
        from envs.tabular_mdp.env import TabularMDP
        return TabularMDP(env_param=get_env_param(env_name, random_param), mdp_known=mdp_known)
    elif env_name == "dynamic_mechanism_design":
        from envs.dynamic_mechanism_design.env import DynamicMechanismDesign
        return DynamicMechanismDesign(env_param=get_env_param(env_name, random_param))
    elif env_name == "bargain_alternate_singleissue":
        from envs.bargain_alternate_singleissue.env import BargainAlternateSingleIssue
        return BargainAlternateSingleIssue(env_param=get_env_param(env_name, random_param))
    elif env_name == "bargain_onesided_uncertainty":
        from envs.bargain_onesided_uncertainty.env import BargainOneSidedUncertainty
        return BargainOneSidedUncertainty(env_param=get_env_param(env_name, random_param))

def create_agents(env, logger, agent_type, agent_engine, mdp_known=True):
    if env.name == "tabular_mdp" and mdp_known:
        demo = load_initial_instructions("envs/tabular_mdp/prompts/tabular_mdp_vi_exmps.txt")
        from envs.tabular_mdp.tools import tool_names_mdp_known
        init_memory = {"P":env.P, 
                       "R":env.R, 
                       "nState":env.nState, 
                       "nAction":env.nAction, 
                       "epLen":env.epLen,
                       "V": np.zeros((env.epLen,env.nState)),
                       "Q": np.zeros((env.epLen,env.nState,env.nAction)),
                       }

        agent = StriDeAgent(problem_description=env.description_of_problem_class, demo=demo, tool_names=tool_names_mdp_known, init_memory=init_memory, logger=logger, engine=agent_engine)
        return {"agent":agent}
    
    elif env.name == "tabular_mdp" and not mdp_known:
        demo = load_initial_instructions("envs/tabular_mdp/prompts/tabular_mdp_ucbvi_exmps.txt")
        R_init = np.zeros((env.nState, env.nAction, 2))
        P_init = np.ones((env.nState, env.nAction, env.nState)) * 1.0 / env.nState
        assert R_init.shape == env.R.shape
        assert P_init.shape == env.P.shape
        bonus_scale_factor = 0.1
        init_memory = {"P":P_init, "R":R_init, 
                        "nState":env.nState, "nAction":env.nAction, "epLen":env.epLen,
                        "V": np.zeros((env.epLen,env.nState)),
                        "Q": np.zeros((env.epLen,env.nState,env.nAction)),
                        "Nsa": np.ones((env.nState, env.nAction)),
                        "bonus_scale_factor": bonus_scale_factor,
                        "epNum":env.n_episodes,
                        }
        from envs.tabular_mdp.tools import tool_names_mdp_unknown
        agent = StriDeAgent(problem_description=env.description_of_problem_class, demo=demo, tool_names=tool_names_mdp_unknown, init_memory=init_memory, logger=logger, engine=agent_engine)

        return {"agent":agent}
    
    elif env.name == "dynamic_mechanism_design":
        designer_demo = load_initial_instructions("envs/dynamic_mechanism_design/prompts/dynamic_vcg_exmps.txt")
        from envs.dynamic_mechanism_design.tools import tool_names_dynamic_vcg
        working_memory = {"P":env.P, "R":env.R, "nState":env.nState, "nAction":env.nAction, "epLen":env.epLen, 
                        "nAgent":env.nAgent,
                        "V": np.zeros((env.epLen,env.nState)),
                        "Q": np.zeros((env.epLen,env.nState,env.nAction)),
                        "VExcluding": np.zeros((env.nAgent,env.epLen,env.nState)),
                        "QExcluding": np.zeros((env.nAgent,env.epLen,env.nState,env.nAction)),
                        }
        designer = StriDeAgent(problem_description=env.description_of_problem_class, demo=designer_demo, tool_names=tool_names_dynamic_vcg, init_memory=deepcopy(working_memory), llm_validator=False, logger=logger, engine=agent_engine)
        agents = {"designer":designer}
        return agents
    
    elif env.name == "bargain_alternate_singleissue":
        buyer_demo = load_initial_instructions("envs/bargain_alternate_singleissue/prompts/buyer_exmps.txt")
        seller_demo = load_initial_instructions("envs/bargain_alternate_singleissue/prompts/seller_exmps.txt")
        from envs.bargain_alternate_singleissue.tools import tool_names_bargain_complete_info_single
        working_memory = {"T":env.T, "delta_b":env.buyerDiscount, "delta_s":env.sellerDiscount,
                        "SPEPrice": {},
                        }
        buyer = StriDeAgent(problem_description=env.description_of_problem_class, demo=buyer_demo, tool_names=tool_names_bargain_complete_info_single, init_memory=deepcopy(working_memory), llm_validator=False, logger=logger, engine=agent_engine)
        seller = StriDeAgent(problem_description=env.description_of_problem_class, demo=seller_demo, tool_names=tool_names_bargain_complete_info_single, init_memory=deepcopy(working_memory), llm_validator=False, logger=logger, engine=agent_engine)
        agents = {"buyer":buyer, "seller":seller}
        return agents

    elif env.name == "bargain_onesided_uncertainty":
        buyer_demo = load_initial_instructions("envs/bargain_onesided_uncertainty/prompts/buyer_exmps.txt")
        seller_demo = load_initial_instructions("envs/bargain_onesided_uncertainty/prompts/seller_exmps.txt")
        from envs.bargain_onesided_uncertainty.tools import tool_names_bargain_incomplete_info_onsided
        # compute the constant c for all time steps
        c = np.zeros(env.T+1)
        for i in reversed(range(1, env.T+1)):
            if i == env.T:
                c[i] = 0.5
            else:
                c[i] = (1-env.buyerDiscount+env.buyerDiscount*c[i+1])**2 / (2*(1-env.buyerDiscount+env.buyerDiscount*c[i+1])-env.sellerDiscount*c[i+1])

        working_memory = {"T":env.T, "delta_b":env.buyerDiscount, "delta_s":env.sellerDiscount,
                        "SEPrice": {}, "c":c, "b_value":env.buyerVal
                        }
        buyer = StriDeAgent(problem_description=env.description_of_problem_class, demo=buyer_demo, tool_names=tool_names_bargain_incomplete_info_onsided, init_memory=deepcopy(working_memory), llm_validator=False, logger=logger, engine=agent_engine)
        working_memory = {"T":env.T, "delta_b":env.buyerDiscount, "delta_s":env.sellerDiscount,
                        "SEPrice": {}, "c":c
                        }
        seller = StriDeAgent(problem_description=env.description_of_problem_class, demo=seller_demo, tool_names=tool_names_bargain_incomplete_info_onsided, init_memory=deepcopy(working_memory), llm_validator=False, logger=logger, engine=agent_engine)
        agents = {"buyer":buyer, "seller":seller}
        return agents