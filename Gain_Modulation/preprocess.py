''' 
preprocess.py: initialize inputs for testing whitening transformation. Use Cholesky decomposition
                to draw a set of vectors with a random covariance matrix. 
'''

import numpy as np
from scipy.stats import ortho_group

# Initialize a set of input vectors with random covariance matrix 
class Preprocess:
    def __init__(self, dim, t, contexts, rng=None):
        self.dim = dim # dimension of input (N)
        self.t = t     # number of input (stimulus) vectors in a statistical context
        self.contexts = contexts
        self.rng = np.random.default_rng(rng)
        self.rand_cov_matrix = None # initialize variable

    def random_cov(self):  # create random covariance matrix to sample from 
        A = ortho_group.rvs(dim=self.dim, random_state=self.rng) 
        eigen_vec = self.rng.uniform(0.01, 3, size=self.dim)
        self.rand_cov_matrix = (A @ np.diag(eigen_vec)) @ A.T
        return self.rand_cov_matrix

    def centered_inputs(self): # Input vectors need to be mean = 0 
        if self.rand_cov_matrix is None: 
            self.random_cov()

        S_0 = self.rng.normal(size=(self.t, self.dim)) # generate a group of t N-dimensional vecs (identity cov)
        L = np.linalg.cholesky(self.rand_cov_matrix) # Cholesky decomp for correlating samples
        S = S_0 @ L.T  # correlate the samples by multiplying by the cholesky decomp of covariance

        S -= S.mean(axis=0, keepdims=True)
        return S
    
    def synthetic_dataset(self):
        V_list = []
        Cov_contexts = [] # initializing list of covariance matrices (one per context)
        for i in range(self.dim): # Create the columns of V (context independent)
            col = self.rng.standard_normal(self.dim)
            col = col/np.linalg.norm(col)
            V_list.append(col)
        V = np.column_stack(V_list)
        for i in range(int(self.contexts)): # Create the lambda diagonal matrix
            l = []
            for j in range(self.dim):
                if self.rng.uniform(0,1) > 0.5:
                    l_i = 0
                else:
                    l_i = self.rng.uniform(0,4)
                l.append(l_i)
            l_matrix = np.diag(l) 
            M = np.eye(self.dim) + V @ l_matrix @ V.T  # Inverse whitening matrix
            Cov_matrix = M @ M.T
            Cov_contexts.append(Cov_matrix)

        # Now create the samples for each context. One long array of 2D inputs. 
        synthetic_inputs = []
        for i in range(self.contexts):
            S_0 = self.rng.normal(size=(self.t, self.dim)) # generate a group of t N-dimensional vecs (identity cov)
            L = np.linalg.cholesky(Cov_contexts[i]) # Cholesky decomp for correlating samples
            S = S_0 @ L.T  # correlate the samples by multiplying by the cholesky decomp of covariance
            S -= S.mean(axis=0, keepdims=True)
            synthetic_inputs.append(S)
        
        return Cov_contexts, synthetic_inputs, V


        



    

