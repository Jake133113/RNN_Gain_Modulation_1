''' whiten.py: main script of this repository. Performs semi-online whitening of a set of inputs through
                adaptively changing gains to meet the whitening objective. '''

import numpy as np
import tqdm
from typing import Optional
from frame import Frame
from preprocess import Preprocess
import pandas as pd  


class Whiten2: # Need to adjust this so that it takes in a context
    def __init__(self, 
                 dim: int, 
                 num_inputs: Optional[int] = None, 
                 ):
        self.dim = int(dim) # N
        self.num_inputs = num_inputs if num_inputs is not None else int(1e4) # default as t = 10000
        
    def i_vals2(self):  # initial values of inputs, Weight matrix, gains, gamma
        N = self.dim
        p = Preprocess(dim=N, t=self.num_inputs) # calling preprocess.py for input generation
        s_t = p.centered_inputs()  

        F = Frame(dim=N, multi_timescale=True) # calling frame.py for generation of fixed weight matrix
        W = F.mercedes() 
        gamma_r = F.gamma_r
        gamma_g = F.gamma_g
        gamma_w = F.gamma_w
        gains = F.g
        return s_t, W, gains, gamma_r, gamma_g, gamma_w

    def whiten2(self): # Actual simulation loop and whitening process
        t = self.num_inputs # num inputs
        s_t, W, gains, gamma_r, gamma_g, gamma_w = self.i_vals2() 
        N, K = W.shape
        C_ss = np.cov(s_t, rowvar=False) # covariance matrix of input vectors - not identity

        # unit-norm for weight matrix vectors 
        W = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)

        I = np.eye(N)

        gain_memory, err_memory, variance_memory = [], [], [] # initialize arrays for plotting
        a = 1 #alpha

        # Main double loop
        for i in tqdm.trange(t, desc="gain update step: "):
            #inititialize r --> 0 at each step
            r_t = np.zeros(N)

            while not_converged: #How do I classify convergence?
                z_t = W.T @ r_t
                n_t = gains * z_t
                r_t += gamma_r * (s_t[t] - W @ n_t - a * r_t)
            
            g += gamma_g*(z_t * z_t - np.diag(W.T @ W))
            W += gamma_w*(r_t @ n_t.T - W @ np.diag(g))

        return gains
    
    def get_variances(self, W, Css, M):
        N = self.dim
        Crr = M @ Css @ M.T # because r_stable = M * s_t, this transforms s cov --> r cov
        Czz = W.T @ Crr @ W # because z_t = W * r_t, this transforms r cov --> z cov
        variances = np.diag(Czz) # covariance of the interneurons
        return variances
    
    def eval_err(self, M, Css): #compute difference of output cov from identity (perfectly whitened)
        Crr = M @ Css @ M.T # same as get_variances why this works
        N = self.dim
        eigvals = np.linalg.eigvalsh(Crr) # eigenvals of output cov
        diff = eigvals - 1 
        error = 1/N * np.sum(diff**2) # average distance squared from 1 of Crr eigvals 
        return error


if __name__ == "__main__":

    w = Whiten2(dim=2, num_inputs=10000)  # N = 2, t = 10000
    gains, gain_memory, err_memory, variance_memory = w.whiten()

    rows = [
        np.concatenate([gain_memory[i], variance_memory[i], np.array([err_memory[i]])])
        for i in range(len(gain_memory))
    ]

    pd.DataFrame(rows).to_csv("white2_output.csv", index=False)
