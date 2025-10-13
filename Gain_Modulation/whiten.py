#Python script to implement the learning of gains 
import numpy as np
import tqdm

def whiten(W, g):
    N, K = W.shape
    #First, define the analytically converged neuronal responses, r_stable:
    r_stable = np.linalg.inv(np.eye(N)+W @ np.diag(g) @ W.T) 

    



