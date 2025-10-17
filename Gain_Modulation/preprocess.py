import numpy as np
from scipy.stats import ortho_group

'''Inputs: t (the amount of input vectors, also the 'time steps';   N   
(Length of Input vectors))

Output: Random Covariance Matrix'''
def random_cov(N):
    A = ortho_group.rvs(dim=N)
    rng = np.random.default_rng()
    eigen_vec = lam = rng.uniform(0.01, 3, size=N)
    rand_cov_matrix = (A @ np.diag(eigen_vec)) @ A.T

    return rand_cov_matrix


def centered_inputs(t, N):
    Sigma = random_cov(N)
    rng = np.random.default_rng()

    #create t random input vectors that are N dimensional
    S_0 = rng.normal(size=(t, N))
    S_0 -= S_0.mean(axis=0, keepdims=True)

    L = np.linalg.cholesky(Sigma)
    S = S_0 @ L.T

    return S, Sigma

S, sigma = centered_inputs(100000, 3)
print("Sample covariance:\n", np.cov(S, rowvar=False))


'''
def eval_err(A):

'''

