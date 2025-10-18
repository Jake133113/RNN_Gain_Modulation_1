import numpy as np
from scipy.stats import ortho_group

''' Class that will initialize a set of input vectors with random covariance matrix 
    to be whitened with whiten.py '''
class Preprocess:
    def __init__(self, dim, t, rng=None):
        self.dim = dim
        self.t = t
        self.rng = np.random.default_rng(rng)
        self.rand_cov_matrix = None

    def random_cov(self):
        A = ortho_group.rvs(dim=self.dim)
        eigen_vec = self.rng.uniform(0.01, 3, size=self.dim)
        self.rand_cov_matrix = (A @ np.diag(eigen_vec)) @ A.T
        return self.rand_cov_matrix

    def centered_inputs(self):
        if self.rand_cov_matrix is None:
            self.random_cov()
        #create t random input vectors that are N dimensional
        S_0 = self.rng.normal(size=(self.t, self.dim))
        S_0 -= S_0.mean(axis=0, keepdims=True) #center the means about 0
        L = np.linalg.cholesky(self.rand_cov_matrix)
        S = S_0 @ L.T
        return S
