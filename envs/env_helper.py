import numpy as np
import random

class BaseEnv:
    def __init__(self):
        self.name = "BaseEnv"
        self.required_agents = []
        self.is_done = False

    def get_description(self, agent_role=None):
        return ""

    def check_agents(self, agents):
        return set(agents.keys()) == set(self.required_agents)

    def reset(self):
        pass

    def step(self, action):
        pass

def get_env_param(env_name, random_param=False):
    if env_name == "bargain_alternate_singleissue":
        if random_param:
            # random.randint(3,4)
            # round(random.uniform(0.5, 0.99), 2)
            return {"T":3, "buyerDiscount":round(random.uniform(0.5, 0.99), 2), "sellerDiscount":round(random.uniform(0.5, 0.99), 2)}
        else:
            return {"T":3, "buyerDiscount":0.8, "sellerDiscount":0.7}
    elif env_name == "bargain_onesided_uncertainty":
        if random_param:
            # random.randint(3,4)
            # round(random.uniform(0.5, 0.99), 2)
            return {"T":3, "buyerDiscount":round(random.uniform(0.5, 0.99), 2), "sellerDiscount":round(random.uniform(0.5, 0.99), 2), "buyerVal":round(random.uniform(0.1, 0.9), 2)}
        else:
            return {"T":3, "buyerDiscount":0.7, "sellerDiscount":0.8, "buyerVal":0.5}
    elif env_name == "dynamic_mechanism_design":
        if random_param:
            # epLen = np.random.randint(low=3, high=6)
            # epLen = 5
            # nState = np.random.randint(low=2, high=3) #6
            # nAction = np.random.randint(low=2, high=5) #4
            epLen = 6
            nState = 3
            nAction = 3
            nAgent = 2
            # get one P_true and nAgent number of R_true
            R_true_ls = []
            for i in range(nAgent):
                P_true, R_true_mean = randmdp(nState, nAction)
                R_true_var = np.ones((nState,nAction)) * 0.1
                assert R_true_var.shape == R_true_mean.shape
                R_true_mean = np.expand_dims(R_true_mean, axis=-1)
                R_true_var = np.expand_dims(R_true_var, axis=-1)
                R_true = np.concatenate((R_true_mean,R_true_var), axis=-1)
                R_true_ls.append(np.expand_dims(R_true, axis=0))
                P_true = np.transpose(P_true, axes=[1,0,2])

            R_true = np.concatenate(R_true_ls,axis=0)
            
            assert P_true.shape == (nState, nAction, nState)
            assert R_true.shape == (nAgent, nState, nAction, 2)
            return {"nState":nState, "nAction":nAction, "nAgent":nAgent, "epLen":epLen, "R":R_true, "P":P_true}
        else:
            raise ValueError("fixed param not supported")
  
    elif env_name == "tabular_mdp":
        if random_param:
            # epLen = np.random.randint(low=3, high=6)
            # epLen = 5
            # nState = np.random.randint(low=2, high=3) #6
            # nAction = np.random.randint(low=2, high=5) #4
            epLen = 5
            nState = 3
            nAction = 3
            P_true, R_true_mean = randmdp(nState, nAction)
            R_true_var = np.ones((nState,nAction)) * 0.1
            assert R_true_var.shape == R_true_mean.shape
            R_true_mean = np.expand_dims(R_true_mean, axis=-1)
            R_true_var = np.expand_dims(R_true_var, axis=-1)
            R_true = np.concatenate((R_true_mean,R_true_var), axis=-1)

            P_true = np.transpose(P_true, axes=[1,0,2])
            assert P_true.shape == (nState, nAction, nState)
            assert R_true.shape == (nState, nAction, 2)
            return {"nState":nState, "nAction":nAction, "epLen":epLen, "R":R_true, "P":P_true}
        else:
            # RiverSwim MDP
            epLen= 5
            nState= 3
            nAction = 3
            R_true = np.zeros((nState, nAction, 2), dtype=np.float32)
            P_true = np.zeros((nState, nAction, nState), dtype=np.float32)

            for s in range(nState):
                for a in range(nAction):
                    R_true[s, a] = (0, 0)
                    P_true[s, a] = np.zeros(nState)

            # Rewards
            R_true[0, 0] = (5. / 1000, 0)
            R_true[nState - 1, 1] = (1, 0)

            # Transitions
            for s in range(nState):
                P_true[s, 0][max(0, s-1)] = 1.

            for s in range(1, nState - 1):
                P_true[s, 1][min(nState - 1, s + 1)] = 0.35
                P_true[s, 1][s] = 0.6
                P_true[s, 1][max(0, s-1)] = 0.05

            P_true[0, 1][0] = 0.4
            P_true[0, 1][1] = 0.6
            P_true[nState - 1, 1][nState - 1] = 0.6
            P_true[nState - 1, 1][nState - 2] = 0.4
            return {"nState":nState, "nAction":nAction, "epLen":epLen, "R":R_true, "P":P_true}
    else:
        raise ValueError("Unknown game {}".format(env_name))

def _randDense(states, actions, mask):
    """Generate random dense ``P`` and ``R``. See ``rand`` for details.

    """
    # definition of transition matrix : square stochastic matrix
    P = np.zeros((actions, states, states))
    # definition of reward matrix (values between -1 and +1)
    for action in range(actions):
        for state in range(states):
            # create our own random mask if there is no user supplied one
            if mask is None:
                m = np.random.random(states)
                r = np.random.random()
                m[m <= r] = 0
                m[m > r] = 1
            elif mask.shape == (actions, states, states):
                m = mask[action][state]  # mask[action, state, :]
            else:
                m = mask[state]
            # Make sure that there is atleast one transition in each state
            if m.sum() == 0:
                m[np.random.randint(0, states)] = 1
            P[action][state] = m * np.random.random(states)
            P[action][state] = P[action][state] / P[action][state].sum()

    R = np.random.uniform(low=-1, high=1, size=(states, actions))

    return P, R

def randmdp(S, A, is_sparse=False, mask=None):
    """Generate a random Markov Decision Process.

    Parameters
    ----------
    S : int
        Number of states (> 1)
    A : int
        Number of actions (> 1)
    is_sparse : bool, optional
        False to have matrices in dense format, True to have sparse matrices.
        Default: False.
    mask : array, optional
        Array with 0 and 1 (0 indicates a place for a zero probability), shape
        can be ``(S, S)`` or ``(A, S, S)``. Default: random.

    Returns
    -------
    out : tuple
        ``out[0]`` contains the transition probability matrix P  and ``out[1]``
        contains the reward matrix R. If ``is_sparse=False`` then P is a numpy
        array with a shape of ``(A, S, S)`` and R is a numpy array with a shape
        of ``(S, A)``. If ``is_sparse=True`` then P and R are tuples of length
        ``A``, where each ``P[a]`` is a scipy sparse CSR format matrix of shape
        ``(S, S)`` and each ``R[a]`` is a scipy sparse csr format matrix of
        shape ``(S, 1)``.

    Examples
    --------
    >>> import numpy, mdptoolbox.example
    >>> numpy.random.seed(0) # Needed to get the output below
    >>> P, R = mdptoolbox.example.rand(4, 3)
    >>> P
    array([[[ 0.21977283,  0.14889403,  0.30343592,  0.32789723],
            [ 1.        ,  0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.43718772,  0.54480359,  0.01800869],
            [ 0.39766289,  0.39997167,  0.12547318,  0.07689227]],
    <BLANKLINE>
           [[ 1.        ,  0.        ,  0.        ,  0.        ],
            [ 0.32261337,  0.15483812,  0.32271303,  0.19983549],
            [ 0.33816885,  0.2766999 ,  0.12960299,  0.25552826],
            [ 0.41299411,  0.        ,  0.58369957,  0.00330633]],
    <BLANKLINE>
           [[ 0.32343037,  0.15178596,  0.28733094,  0.23745272],
            [ 0.36348538,  0.24483321,  0.16114188,  0.23053953],
            [ 1.        ,  0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  1.        ,  0.        ]]])
    >>> R
    array([[[-0.23311696,  0.58345008,  0.05778984,  0.13608912],
            [-0.07704128,  0.        , -0.        ,  0.        ],
            [ 0.        ,  0.22419145,  0.23386799,  0.88749616],
            [-0.3691433 , -0.27257846,  0.14039354, -0.12279697]],
    <BLANKLINE>
           [[-0.77924972,  0.        , -0.        , -0.        ],
            [ 0.47852716, -0.92162442, -0.43438607, -0.75960688],
            [-0.81211898,  0.15189299,  0.8585924 , -0.3628621 ],
            [ 0.35563307, -0.        ,  0.47038804,  0.92437709]],
    <BLANKLINE>
           [[-0.4051261 ,  0.62759564, -0.20698852,  0.76220639],
            [-0.9616136 , -0.39685037,  0.32034707, -0.41984479],
            [-0.13716313,  0.        , -0.        , -0.        ],
            [ 0.        , -0.        ,  0.55810204,  0.        ]]])
    >>> numpy.random.seed(0) # Needed to get the output below
    >>> Psp, Rsp = mdptoolbox.example.rand(100, 5, is_sparse=True)
    >>> len(Psp), len(Rsp)
    (5, 5)
    >>> Psp[0]
    <100x100 sparse matrix of type '<... 'numpy.float64'>'
        with 3296 stored elements in Compressed Sparse Row format>
    >>> Rsp[0]
    <100x100 sparse matrix of type '<... 'numpy.float64'>'
        with 3296 stored elements in Compressed Sparse Row format>
    >>> # The number of non-zero elements (nnz) in P and R are equal
    >>> Psp[1].nnz == Rsp[1].nnz
    True

    """
    # making sure the states and actions are more than one
    if S>1:
        assert S > 1, "The number of states S must be greater than 1."
        assert A > 1, "The number of actions A must be greater than 1."
        # if the user hasn't specified a mask, then we will make a random one now
        if mask is not None:
            # the mask needs to be SxS or AxSxS
            try:
                assert mask.shape in ((S, S), (A, S, S)), (
                    "'mask' must have dimensions S×S or A×S×S."
                )
            except AttributeError:
                raise TypeError("'mask' must be a numpy array or matrix.")
        # generate the transition and reward matrices based on S, A and mask
        if is_sparse:
            pass
        else:
            P, R = _randDense(S, A, mask)
        return P, R
    else:
        P = np.ones((A, S, S))
        R = np.random.uniform(low=-1, high=1, size=(S,A))
        return P, R