import numpy as np
import tqdm
from typing import Optional
from frame import Frame
from preprocess import Preprocess
import pandas as pd  # (added)

class Whiten:
    def __init__(self, 
                 dim: int, 
                 num_inputs: Optional[int] = None, 
                 ):
        self.dim = int(dim)
        self.num_inputs = num_inputs if num_inputs is not None else int(1e4)
        
    def i_vals(self):  # initial values of inputs, Weight matrix, gains, gamma
        N = self.dim
        p = Preprocess(dim=N, t=self.num_inputs)
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

        for i in tqdm.trange(t, desc="gain update step: "):  # (added tqdm ticker)
            r_stable = np.linalg.inv(np.eye(N) + W @ np.diag(gains) @ W.T) @ s_t[i]
            z_t = W.T @ r_stable
            
            z_memory.append(z_t.copy())          # (ensure values are frozen per step)
            gains += gamma * ((z_t * z_t).T - 1)  # gain update
            gain_memory.append(gains.copy())      # (ensure values are frozen per step)

        return gains, gain_memory, z_memory

if __name__ == "__main__":
    w = Whiten(dim=2, num_inputs=10000)  
    gains, gain_memory, z_memory = w.whiten()

    rows = [np.concatenate([gain_memory[i], z_memory[i]]) for i in range(len(z_memory))]
    pd.DataFrame(rows).to_csv("white_output.csv", index=False)
