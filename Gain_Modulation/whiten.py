#Python script to implement the learning of gains 
import numpy as np
#import tqdm
from typing import Optional
from frame import Frame
from analyze import Analyze
from preprocess import Preprocess

class Whiten:
    def __init__(self, 
                 dim: int, 
                 num_inputs: Optional[int] = None, 
                 ):
        self.dim = int(dim)

        self.num_inputs = num_inputs if num_inputs is not None else int(1e4)
        
    def i_vals(self): #initial values of inputs, Weight matrix, gains, gamma
        N = self.dim
        p = Preprocess(dim=N, #dimensions of inputs
                        t=self.num_inputs #number of input vectors
                        )
        s_t = p.centered_inputs()

        F = Frame(dim=N)
        W = F.mercedes()
        gamma = F.gamma
        gains = F.g
        return s_t, W, gains, gamma

    def whiten(self):
        t = self.num_inputs
        s_t, W, gains, gamma = self.i_vals()
        N, K = W.shape
        z_memory = []
        gain_memory = []

        for i in range(t):
            r_stable = np.linalg.inv(np.eye(N)+W @ np.diag(gains) @ W.T) @ s_t[i]
            z_t = W.T @ r_stable
            
            z_memory.append(z_t)
            gains += gamma*((z_t * z_t).T - 1) #make gain update
            gain_memory.append(gains)

        return gains, gain_memory, z_memory







