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
        C_ss = np.cov(s_t, rowvar=False)

        # 0) one-time: unit-norm for weight matrix and input vectors 
        W = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)

        I = np.eye(N)

        # MAIN LOOP
        #z2_ema_norm = np.ones(K)  # for plotting normalized variance
        beta = 0.01
        z_memory, gain_memory, err_memory, variance_memory = [], [], [], []

        #print('s_t: ', s_t)
        #print('initial covariance matrix: ', np.cov(s_t, rowvar=False))

        for i in tqdm.trange(t, desc="gain update step: "):
            I_ss = np.eye(N)
            WGW = W @ np.diag(gains) @ W.T
            M = np.linalg.inv(I_ss + WGW)
            r_stable = M @ s_t[i]
            z_t = W.T @ r_stable

            # normalize by baseline variance
            variances = (z_t * z_t)
            dg = variances - 1

            # per-dimension online update (now target 1 is attainable)
            gains += gamma * dg

            variance = self.get_variances(W, C_ss, M)

            # keep gains nonnegative but avoid a hard absorbing 0
            #gains = np.maximum(gains, 1e-12)

            # for plots: smoothed normalized variance (should -> 1)
            #z2_ema_norm = (1 - beta) * z2_ema_norm + beta * variances

            z_memory.append(variances.copy())
            variance_memory.append(variance.copy())
            gain_memory.append(gains.copy())
            err_memory.append(self.eval_err(M, C_ss))

        return gains, gain_memory, z_memory, err_memory, variance_memory
    
    def get_variances(self, W, Css, M):
        N = self.dim
        Crr = M @ Css @ M.T
        Czz = W.T @ Crr @ W
        variances = np.diag(Czz)
        return variances
    
    def eval_err(self, M, Css): #input is a matrix of the whitening output. r_t vectors should be colmum-stacked
        Crr = M @ Css @ M.T
        N = self.dim
        eigvals = np.linalg.eigvalsh(Crr)
        diff = eigvals - 1
        error = 1/N * np.sum(diff**2)
        return error


if __name__ == "__main__":

    w = Whiten(dim=2, num_inputs=10000)  
    gains, gain_memory, z_memory, err_memory, variance_memory = w.whiten()

    rows = [
        np.concatenate([gain_memory[i], variance_memory[i], np.array([err_memory[i]])])
        for i in range(len(gain_memory))
    ]

    pd.DataFrame(rows).to_csv("white_output3.csv", index=False)
