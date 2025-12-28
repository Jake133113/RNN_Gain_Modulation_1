import numpy as np
import tqdm
from typing import Optional
import pandas as pd  
from frame import Frame
from preprocess import Preprocess

class Whiten2:
    def __init__(self, 
                 dim: int, 
                 num_inputs: Optional[int] = None, 
                 num_contexts: int = 1, 
                 ):
        self.dim = int(dim) 
        self.num_inputs = num_inputs if num_inputs is not None else int(1e4)
        self.num_contexts = int(num_contexts)
        
    def i_vals2(self): 
        N = self.dim
        K = self.dim # relaxed condition on overcomplete basis 
        # Pass seed for reproducibility if desired
        p = Preprocess(dim=N, t=self.num_inputs, contexts=self.num_contexts) 
        
        # s_t comes out as a list of arrays [Context1, Context2...]
        # Returns V as well (Ideal Axes)
        cov_matrices, s_t_list, V = p.synthetic_dataset()  

        # Concatenate list into one long array
        s_t = np.concatenate(s_t_list, axis=0)

        F = Frame(dim=K, multi_timescale=True) 
        W = F.mercedes() 
        gamma_r = F.gamma_r
        gamma_g = F.gamma_g
        gamma_w = F.gamma_w
        gains = F.g
        return s_t, W, gains, gamma_r, gamma_g, gamma_w, cov_matrices, V

    def whiten2(self): 
        s_t, W, gains, gamma_r, gamma_g, gamma_w, cov_matrices, V = self.i_vals2() 
        
        # Normalize W columns and create initial copy
        W = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
        W_0 = W.copy()
        
        total_steps = s_t.shape[0]
        N, K = W.shape

        gain_memory = []
        err_memory = []
        err2_memory = []
        
        a = 1.0 # alpha

        # Main loop
        for i in tqdm.trange(total_steps, desc="gain update step: "):
            
            # Settling Phase (Finding r_t)
            r_t = np.zeros(N)
            dr_norm = 1.0 
            
            counter = 0
            while dr_norm > 1e-7 and counter < 5000:
                z_t = W.T @ r_t
                n_t = gains * z_t
                
                dr = gamma_r * (s_t[i] - W @ n_t - a * r_t)
                
                r_t += dr
                dr_norm = np.linalg.norm(dr)
                counter += 1
            
            # Re-calculate z_t/n_t with final settled r_t
            z_t = W.T @ r_t
            n_t = gains * z_t 

            # Update Gains
            gains += gamma_g * (z_t * z_t - np.diag(W.T @ W))
            
            # Update Weights (outer product for r @ n.T)
            W += gamma_w * (np.outer(r_t, n_t) - W @ np.diag(gains))

            # Store history
            gain_memory.append(gains.copy())

            # Calculate errors
            # M = Adapted, M2 = Fixed Initial
            M = a*np.eye(self.dim) + W @ np.diag(gains) @ W.T
            M2 = a*np.eye(self.dim) + W_0 @ np.diag(gains) @ W_0.T
            
            steps_per_context = total_steps // self.num_contexts 
            context_number = i // steps_per_context 
            cov = cov_matrices[context_number] 
            
            error = self.eval_err(M, cov)
            error2 = self.eval_err(M2, cov)
            
            err_memory.append(error)
            err2_memory.append(error2)

        return gains, gain_memory, err_memory, err2_memory, W_0, W, V
    
    def eval_err(self, M, Css): 
        # Using inverse of M based on your formula
        Crr = np.linalg.inv(M) @ Css @ np.linalg.inv(M) 
        N = self.dim
        eigvals = np.linalg.eigvalsh(Crr) 
        diff = eigvals - 1 
        error = 1/N * np.sum(diff**2) 
        return error

if __name__ == "__main__":
    # N=2, 1000 contexts, 1000 samples each = 300,000 total
    w = Whiten2(dim=2, num_inputs=1000, num_contexts=1000)  

    gains, gain_memory, err_memory, err2_memory, W_0, W_final, V_ideal = w.whiten2()

    # 1. SAVE TIME SERIES DATA (CSV)
    rows = []
    for i in range(len(gain_memory)):
        g_val = gain_memory[i]
        e_val = np.array([err_memory[i]])
        e_val2 = np.array([err2_memory[i]])
        
        row = np.concatenate([g_val, e_val, e_val2])
        rows.append(row)

    pd.DataFrame(rows).to_csv("white2_output.csv", index=False)
    print("Saved time-series data to 'white2_output.csv'")

    # 2. SAVE MATRICES FOR VISUALIZATION (NPZ)
    np.savez("frame_data.npz", 
             W_init=W_0, 
             W_final=W_final, 
             V_ideal=V_ideal)
    print("Saved matrix data to 'frame_data.npz'")