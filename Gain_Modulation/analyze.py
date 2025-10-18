import numpy as np

class Analyze:
    def __init__(self, r_t):
        self.r_t = r_t
    
    def eval_err(self): #input is a matrix of the whitening output. r_t vectors should be colmum-stacked
        N,t = np.shape(self.r_t)
        I = np.eye(N)
        cov = np.cov(self.r_t)
        error = max(np.linalg.eig(cov-I))
        return error
