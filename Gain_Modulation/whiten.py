import numpy as np
import tqdm
from typing import Optional
from frame import Frame
from preprocess import Preprocess
import pandas as pd  
import analyze 

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

        # 0) one-time: unit-norm columns (already added in your Frame.mercedes)
        W = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)

        I = np.eye(N)

        # MAIN LOOP
        z2_ema_norm = np.ones(K)  # for plotting normalized variance
        beta = 0.01
        z_memory, gain_memory, err_memory = [], [], []

        for i in tqdm.trange(t, desc="gain update step: "):
            A = I + W @ np.diag(gains) @ W.T
            r_stable = np.linalg.solve(A, s_t[i])
            z_t = W.T @ r_stable

            # normalize by baseline variance
            z2 = (z_t * z_t)

            # per-dimension online update (now target 1 is attainable)
            gains += gamma * (z2 - 1.0)

            # keep gains nonnegative but avoid a hard absorbing 0
            gains = np.maximum(gains, 1e-12)

            # for plots: smoothed normalized variance (should -> 1)
            z2_ema_norm = (1 - beta) * z2_ema_norm + beta * z2

            z_memory.append(z2_ema_norm.copy())
            gain_memory.append(gains.copy())
            err_memory.append(analyze.eval_err(r_stable))

        return gains, gain_memory, z_memory, err_memory


if __name__ == "__main__":
    w = Whiten(dim=2, num_inputs=10000)  
    gains, gain_memory, z_memory, err_memory = w.whiten()

    rows = [
        np.concatenate([gain_memory[i], z_memory[i], np.array([err_memory[i]])])
        for i in range(len(z_memory))
    ]

    pd.DataFrame(rows).to_csv("white_output3.csv", index=False)
