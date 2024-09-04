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

def get_env_param(env_name, random_param=True, num_pne=1, num_strategies=None):
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
            return {"nState": nState, "nAction": nAction, "epLen": epLen, "R": R_true, "P": P_true}
        
    elif env_name == "ieds":
        if random_param:
            # Unique PNE
            if num_pne == 1:
                payoff1, payoff2 = generate_payoff_bimatrix_with_unique_pne(num_strategies[0], num_strategies[1])
                return {
                    "num_players": 2,
                    "strategies_per_player": [num_strategies[0], num_strategies[1]],
                    "payoff_matrix": [payoff1.tolist(), payoff2.tolist()]
                }
            # No PNE
            elif num_pne == 0:
                payoff1, payoff2 = generate_payoff_bimatrix_with_no_pne(num_strategies[0], num_strategies[1])
                return {
                    "num_players": 2,
                    "strategies_per_player": [num_strategies[0], num_strategies[1]],
                    "payoff_matrix": [payoff1.tolist(), payoff2.tolist()]
                }
            # Multiple PNE
            else:
                payoff1, payoff2 = generate_payoff_bimatrix_with_multiple_pne(num_strategies[0], num_strategies[1], num_pne)
                return {
                    "num_players": 2,
                    "strategies_per_player": [num_strategies[0], num_strategies[1]],
                    "payoff_matrix": [payoff1.tolist(), payoff2.tolist()]
                }
        else:
            # fixed payoff matrix
            # payoff_matrix = [[[10, 10], [9, 11]], [[10, 9], [10, 8]]]
            # payoff_matrix = [[[10, 10, 8], [9, 11, 11], [8, 9, 9]], [[10, 9, 8], [10, 9, 8], [8, 11, 7]]]
            # payoff_matrix = [[[0, 0], [0, 1]], [[0, 0], [0, 1]]]
            # payoff_matrix =  [[[5, 6], [8, 2]], [[5, 4], [7, 8]]]
            # payoff_matrix = [[[-5, 2, 3], [1, 1, 1], [0, 0, 0]], [[-1, 2, 3], [-3, 2, 1], [10, 0, -10]]]
            # payoff_matrix = [[
            #     [2, 3, 2, 1],   # Row A
            #     [8, 5, 4, 7],   # Row B
            #     [100, 2, 1, 3], # Row C
            #     [3, 6, 2, 0]    # Row D
            # ]
            # ,[
            #     [1, 10, 9, 3],  # Row A
            #     [7, 8, 9, 11],  # Row B
            #     [9, 11, 2, 3],  # Row C
            #     [0, 1, 2, 0]    # Row D
            # ]]
            payoff_matrix = [
                [[12,  6,  7,  8],
                [17,  3,  4,  5],
                [18,  4,  5,  6],
                [19,  5,  6,  7]],
                [[19,  4,  5,  6],
                [ 1,  8, 9, 10],
                [ 2,  9, 10, 11],
                [ 3, 10, 11, 12]]]
            num_players = len(payoff_matrix)
            strategies_per_player = [len(payoff_matrix[0]), len(payoff_matrix[0][0])]
            return {
                "num_players": num_players,
                "strategies_per_player": strategies_per_player,
                "payoff_matrix": payoff_matrix
            }
    else:
        raise ValueError("Unknown game {}".format(env_name))

def generate_payoff_bimatrix_with_no_pne(num_strategies_player1: int, num_strategies_player2: int):
    """
    Generates a random game with no pure Nash Equilibrium using iterated elimination of dominated strategies.

    Args:
        num_strategies_player1: Number of strategies for Player 1.
        num_strategies_player2: Number of strategies for Player 2.

    Returns:
        A tuple containing two numpy arrays representing the payoff matrices for Player 1 and Player 2.
    """
    if num_strategies_player1 <= 0 or num_strategies_player2 <= 0:
        raise ValueError("Number of strategies must be positive integers")
    
    elif num_strategies_player1 == 1 and num_strategies_player2 == 1:
        raise ValueError("Game must have at least two strategies for each player")
    
    else:
        # Generate distinct scores for each strategy
        # initial_value_a = np.random.randint(5, 20)
        # initial_value_b = np.random.randint(5, 20)
        initial_value_a = 1
        initial_value_b = 0

        m = num_strategies_player1
        n = num_strategies_player2
        A = np.zeros((m, n), dtype=int)
        B = np.zeros((m, n), dtype=int)
        
        # Fill in the matrices using the antisymmetric property
        for i in range(m):
            for j in range(n):
                if (i + j) % 2 == 0:
                    A[i, j] = initial_value_a
                    B[i, j] = initial_value_b
                else:
                    A[i, j] = initial_value_a
                    B[i, j] = initial_value_b
        
        print("Payoff Matrix for Player 1: \n", A)
        print("Payoff Matrix for Player 2: \n", B)
        return A, B

def generate_payoff_bimatrix_with_unique_pne(num_strategies_player1: int, num_strategies_player2: int):
    """
    Generates a random bi-matrix game with a unique pure Nash Equilibrium using iterated elimination of dominated strategies.

    Args:
        num_strategies_player1: Number of strategies for Player 1.
        num_strategies_player2: Number of strategies for Player 2.

    Returns:
        A tuple containing two numpy arrays representing the payoff matrices for Player 1 and Player 2.
    """
    def add_dominated_row(A, B, increment):
        min_values_A = np.min(A, axis=0)
        new_row_A = min_values_A - increment
        A = np.vstack((A, new_row_A))

        new_row_B = np.zeros(A.shape[1], dtype=int)
        for i in range(A.shape[1]):
            if i == 0:
                new_row_B[i] = np.min(B[:, i]) - increment
            else:
                new_row_B[i] = np.max(B[:, i-1]) + increment
        
        B = np.vstack((B, new_row_B))
        return A, B

    def add_dominated_column(A, B, increment):
        min_values_B = np.min(B, axis=1)
        new_col_B = min_values_B - increment
        B = np.column_stack((B, new_col_B))

        new_col_A = np.zeros(B.shape[0], dtype=int)
        for i in range(B.shape[0]):
            if i == 0:
                new_col_A[i] = np.min(A[i, :]) - increment
            else:
                new_col_A[i] = np.max(A[i-1, :]) + increment

        A = np.column_stack((A, new_col_A))
        return A, B
    
    if num_strategies_player1 <= 0 or num_strategies_player2 <= 0:
        raise ValueError("Number of strategies must be positive integers")
    
    elif num_strategies_player1 == 1 and num_strategies_player2 == 1:
        raise ValueError("Game must have at least two strategies for each player")
    
    elif num_strategies_player1 == num_strategies_player2:
        # Scenario 1: n x n bi-matrix game with unique PNE
        n = num_strategies_player1
        print("Number of strategies for each player: ", n)

        initial_value = np.random.randint(30, 50)
        increment = np.random.randint(1, 5)
        # increment = 1
        A = np.array([[initial_value, initial_value], 
                  [initial_value - increment, initial_value + increment]])
        B = np.array([[initial_value, initial_value - increment], [initial_value, initial_value - increment]])

        # Expand the matrices alternately to ensure unique IEDS order
        for size in range(3, n + 1):
            if A.shape[0] < n:  # Add row if needed
                A, B = add_dominated_row(A, B, increment)

            if B.shape[1] < n:  # Add column if needed
                A, B = add_dominated_column(A, B, increment)

        A, B, _, _ = permute_payoff_matrices([A, B])

        print("Payoff Matrix for Player 1: \n", A)
        print("Payoff Matrix for Player 2: \n", B)
        return A, B

    elif num_strategies_player1 > num_strategies_player2:
        # Scenario 2: m x n (m > n) bi-matrix game with unique IEDS order
        m = num_strategies_player1
        n = num_strategies_player2
        print("WARNING: The order of IEDS will not be unique for this case.")
        print("Number of strategies for Player 1: ", m)
        print("Number of strategies for Player 2: ", n)

        initial_value = np.random.randint(30, 50)
        increment = np.random.randint(1, 5)

        A = np.array([[initial_value, initial_value], [initial_value - increment, initial_value + increment]])
        B = np.array([[initial_value, initial_value - increment], [initial_value, initial_value - increment]])

        if n < len(B) and n == 1:
            B = np.delete(B, 1, axis=0)
            A = np.delete(A, 1, axis=0)

        # Building the base n x n matrix
        for size in range(3, n + 1):
            A, B = add_dominated_row(A, B, increment)
            A, B = add_dominated_column(A, B, increment)

        # Expanding to m x n by adding rows
        while A.shape[0] < m:
            A, B = add_dominated_row(A, B, increment)
        
        A, B, _, _ = permute_payoff_matrices([A, B])

        print("Payoff Matrix for Player 1: \n", A)
        print("Payoff Matrix for Player 2: \n", B)

        return A, B

    elif num_strategies_player1 < num_strategies_player2:
        # Scenario 3: m x n (m < n) bi-matrix game with unique IEDS order
        m = num_strategies_player1
        n = num_strategies_player2
        print("WARNING: The order of IEDS will not be unique for this case.")
        print("Number of strategies for Player 1: ", m)
        print("Number of strategies for Player 2: ", n)

        initial_value = np.random.randint(30, 50)
        increment = np.random.randint(1, 5)

        A = np.array([[initial_value, initial_value], [initial_value - increment, initial_value + increment]])
        B = np.array([[initial_value, initial_value - increment], [initial_value, initial_value - increment]])

        temp = A
        A = np.transpose(B)
        B = np.transpose(temp)

        if m < len(A) and m == 1:
            A = np.delete(A, 1, axis=0)
            B = np.delete(B, 1, axis=0)

        # Building the base m x m matrix
        for size in range(3, m + 1):
            A, B = add_dominated_row(A, B, increment)
            A, B = add_dominated_column(A, B, increment)

        # Expanding to m x n by adding columns
        while B.shape[1] < n:
            A, B = add_dominated_column(A, B, increment)

        A, B, _, _ = permute_payoff_matrices([A, B])

        print("Payoff Matrix for Player 1: \n", A)
        print("Payoff Matrix for Player 2: \n", B)

        return A, B

def generate_payoff_bimatrix_with_multiple_pne(num_strategies_player1: int, num_strategies_player2: int, num_pne: int):
    """
    Generates a random game with multiple pure Nash Equilibria to be solvable using iterated elimination of dominated strategies.
    
    Args:
    - num_strategies_player1 (int): Number of strategies for Player 1.
    - num_strategies_player2 (int): Number of strategies for Player 2.
    - num_pne (int): Number of pure Nash Equilibria desired.

    Returns:
    - A (np.array): Payoff matrix for Player 1.
    - B (np.array): Payoff matrix for Player 2.
    """
    m = num_strategies_player1
    n = num_strategies_player2

    # Ensure number of PNEs does not exceed the minimum number of strategies for both players
    assert num_pne < min(m, n), "Number of PNEs must be less than the minimum number of strategies for both players."

    # Initialize empty payoff matrices for Player 1 (A) and Player 2 (B)
    A = np.zeros((num_pne, num_pne), dtype=int)
    B = np.zeros((num_pne, num_pne), dtype=int)

    # Generate the initial l x l base matrix with PNEs on the diagonal
    initial_value = 10
    increment = 1
    
    for i in range(num_pne):
        for j in range(num_pne):
            if i == j:
                A[i, j] = initial_value
                B[i, j] = initial_value
            else:
                A[i, j] = A[i, i] - abs(i - j)
                B[i, j] = B[i, i] - abs(i - j)

    def add_dominated_row(A, B, increment):
        min_values_A = np.min(A, axis=0)
        new_row_A = min_values_A - increment
        A = np.vstack((A, new_row_A))

        new_row_B = np.zeros(A.shape[1], dtype=int)
        for i in range(A.shape[1]):
            if i == 0:
                new_row_B[i] = np.min(B[:, i]) - increment
            else:
                new_row_B[i] = np.max(B[:, i-1]) + increment
        
        B = np.vstack((B, new_row_B))
        return A, B

    def add_dominated_column(A, B, increment):
        min_values_B = np.min(B, axis=1)
        new_col_B = min_values_B - increment
        B = np.column_stack((B, new_col_B))

        new_col_A = np.zeros(B.shape[0], dtype=int)
        for i in range(B.shape[0]):
            if i == 0:
                new_col_A[i] = np.min(A[i, :]) - increment
            else:
                new_col_A[i] = np.max(A[i-1, :]) + increment

        A = np.column_stack((A, new_col_A))
        return A, B

    # Expand the l x l base matrix to m x n matrix
    for size in range(num_pne, max(m, n)):
        if A.shape[0] < m:  # Add a row if needed
            A, B = add_dominated_row(A, B, increment)
        if B.shape[1] < n:  # Add a column if needed
            A, B = add_dominated_column(A, B, increment)

    print("Payoff Matrix for Player 1: \n", A)
    print("Payoff Matrix for Player 2: \n", B)
    
    return A, B

def permute_matrix(matrix, row_permutation, col_permutation):
    """Permutes the rows and columns of a matrix using given permutations."""
    permuted_matrix = matrix.copy()

    # Apply the provided permutations
    permuted_matrix = permuted_matrix[row_permutation, :]
    permuted_matrix = permuted_matrix[:, col_permutation]
    return permuted_matrix

def permute_payoff_matrices(payoff_matrices):
    """Permutes the rows and columns of the payoff matrices for all players using the same permutations."""
    
    # Get the shape of the matrices
    n_rows, n_cols = payoff_matrices[0].shape
    
    # Generate the same random permutations for both matrices
    row_permutation = np.random.permutation(n_rows)
    col_permutation = np.random.permutation(n_cols)

    # Apply the same permutation to both payoff matrices
    permuted_matrices = []
    for matrix in payoff_matrices:
        permuted_matrix = permute_matrix(np.array(matrix), row_permutation, col_permutation)
        permuted_matrices.append(permuted_matrix)

    # Return the permuted matrices along with the applied permutations
    return permuted_matrices[0], permuted_matrices[1], row_permutation, col_permutation

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
