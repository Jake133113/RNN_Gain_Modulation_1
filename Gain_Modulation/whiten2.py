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
                 num_contexts: int = 1, # Type hint fixed
                 ):
        self.dim = int(dim) 
        self.num_inputs = num_inputs if num_inputs is not None else int(1e4)
        self.num_contexts = int(num_contexts)
        
    def i_vals2(self): 
        N = self.dim
        # Pass seed for reproducibility if desired
        p = Preprocess(dim=N, t=self.num_inputs, contexts=self.num_contexts) 
        
        # s_t comes out as a list of arrays [Context1, Context2...]
        cov_matrixes, s_t_list = p.synthetic_dataset()  

        # Concatenate list into one long array
        s_t = np.concatenate(s_t_list, axis=0)

        F = Frame(dim=N, multi_timescale=True) 
        W = F.mercedes() 
        gamma_r = F.gamma_r
        gamma_g = F.gamma_g
        gamma_w = F.gamma_w
        gains = F.g
        return s_t, W, gains, gamma_r, gamma_g, gamma_w

    def whiten2(self): 
        s_t, W, gains, gamma_r, gamma_g, gamma_w = self.i_vals2() 
        
        # Total samples is num_inputs * num_contexts
        total_steps = s_t.shape[0]
        
        N, K = W.shape

        # Normalize W columns
        W = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)

        gain_memory = []
        err_memory = []
        variance_memory = []
        
        a = 1.0 # alpha

        # Main loop
        for i in tqdm.trange(total_steps, desc="gain update step: "):
            
            # Settling Phase (Finding r_t)
            r_t = np.zeros(N)
            dr_norm = 1.0 
            
            # FIX 3: Convergence check using Norm, not vector comparison
            counter = 0
            while dr_norm > 1e-6 and counter < 1000: # Added safety break
                z_t = W.T @ r_t
                n_t = gains * z_t
                
                dr = gamma_r * (s_t[i] - W @ n_t - a * r_t)
                
                r_t += dr
                dr_norm = np.linalg.norm(dr)
                counter += 1
            
            # Re-calculate z_t/n_t with final settled r_t
            z_t = W.T @ r_t
            n_t = gains * z_t # This is effectively output y

            # Update Gains
            gains += gamma_g * (z_t * z_t - np.diag(W.T @ W))
            
            # Update Weights (outer product for r @ n.T)
            W += gamma_w * (np.outer(r_t, n_t) - W @ np.diag(gains))

            # Store history
            gain_memory.append(gains.copy())
            
            '''if i % 100 == 0:
                # Placeholder for variance/error tracking
                variance_memory.append(np.var(z_t)) 
                err_memory.append(np.sum((gains - 1)**2)) # Simple proxy error
            else:
                variance_memory.append(variance_memory[-1] if variance_memory else 0)
                err_memory.append(err_memory[-1] if err_memory else 0)'''

        # FIX 7: Return all the memory arrays
        return gains, gain_memory, err_memory
    
if __name__ == "__main__":
    # N=2, 5 contexts, 2000 samples each = 10,000 total
    w = Whiten2(dim=2, num_inputs=2000, num_contexts=5)  
    
    gains, gain_memory, err_memory, variance_memory = w.whiten2()

    # Handle list of arrays for DataFrame
    rows = []
    for i in range(len(gain_memory)):
        # Ensure items are 1D arrays or scalars before concatenation
        g_val = gain_memory[i]
        v_val = np.array([variance_memory[i]])
        e_val = np.array([err_memory[i]])
        
        row = np.concatenate([g_val, v_val, e_val])
        rows.append(row)

    pd.DataFrame(rows).to_csv("white2_output.csv", index=False)