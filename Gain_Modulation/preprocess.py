''' 
preprocess.py: initialize inputs for testing whitening transformation. Use Cholesky decomposition
                to draw a set of vectors with a random covariance matrix. 
'''

import numpy as np
from scipy.stats import ortho_group

# Initialize a set of input vectors with random covariance matrix 
class Preprocess:
    def __init__(self, dim, t, rng=None):
        self.dim = dim # dimension of input (N)
        self.t = t     # number of input (stimulus) vectors in a statistical context
        self.rng = np.random.default_rng(rng)
        self.rand_cov_matrix = None # initialize variable

    def random_cov(self):  # create random covariance matrix to sample from 
        A = ortho_group.rvs(dim=self.dim) 
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
