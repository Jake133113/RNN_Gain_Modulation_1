''' whiten.py: main script of this repository. Performs semi-online whitening of a set of inputs through
                adaptively changing gains to meet the whitening objective. '''

import numpy as np
import tqdm
from typing import Optional
from frame import Frame
from preprocess import Preprocess
import pandas as pd  


class Whiten:
    def __init__(self, 
                 dim: int, 
                 num_inputs: Optional[int] = None, 
                 ):
        self.dim = int(dim) # N
        self.num_inputs = num_inputs if num_inputs is not None else int(1e4) # default as t = 10000
        
    def i_vals(self):  # initial values of inputs, Weight matrix, gains, gamma
        N = self.dim
        p = Preprocess(dim=N, t=self.num_inputs) # calling preprocess.py for input generation
        s_t = p.centered_inputs()  

        F = Frame(dim=N) # calling frame.py for generation of fixed weight matrix
        W = F.mercedes() 
        gamma = F.gamma 
        gains = F.g
        return s_t, W, gains, gamma

    def whiten(self): # Actual simulation loop and whitening process
        t = self.num_inputs # num inputs
        s_t, W, gains, gamma = self.i_vals() 
        N, K = W.shape 
        C_ss = np.cov(s_t, rowvar=False) # covariance matrix of input vectors - not identity

        # unit-norm for weight matrix vectors 
        W = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)

        I = np.eye(N)

        gain_memory, err_memory, variance_memory = [], [], [] # initialize arrays for plotting

        # Main gradient ascent loop
        for i in tqdm.trange(t, desc="gain update step: "):
            I_ss = np.eye(N) 
            WGW = W @ np.diag(gains) @ W.T 
            M = np.linalg.inv(I_ss + WGW) # Equiv to equation 2.8 from Lyndon Thesis
            r_stable = M @ s_t[i] # directly calc fixed point of the output
            z_t = W.T @ r_stable  # project output onto frame axes (interneuron vals)

            # compute the scale of the gain change  
            dg = (z_t * z_t) - 1

            # Perform gain update step using grad ascent step size (eta in Lyndons thesis)
            gains += gamma * dg

            #Compute the variance of the interneurons at each time
            variance = self.get_variances(W, C_ss, M) 

            #append important values to arrays for plotting 
            variance_memory.append(variance.copy())
            gain_memory.append(gains.copy())
            err_memory.append(self.eval_err(M, C_ss))

        return gains, gain_memory, err_memory, variance_memory
    
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

    w = Whiten(dim=2, num_inputs=10000)  # N = 2, t = 10000
    gains, gain_memory, err_memory, variance_memory = w.whiten()

    rows = [
        np.concatenate([gain_memory[i], variance_memory[i], np.array([err_memory[i]])])
        for i in range(len(gain_memory))
    ]

    pd.DataFrame(rows).to_csv("white_output.csv", index=False)
