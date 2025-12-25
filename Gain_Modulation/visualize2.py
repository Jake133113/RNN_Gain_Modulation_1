import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_whitening_comparison(csv_file="white2_output.csv", npz_file="frame_data.npz", num_contexts=5):
    # 1. Load Time Series Data
    df = pd.read_csv(csv_file, header=None)
    
    # Dynamic Column Detection
    num_cols = df.shape[1]
    num_gains = num_cols - 2 
    gain_cols = [f'Gain {i+1}' for i in range(num_gains)]
    col_names = gain_cols + ['Error (Adapted W)', 'Error (Fixed W0)']
    df.columns = col_names
    
    # 2. Setup Plotting Parameters
    total_steps = len(df)
    steps_per_context = total_steps // num_contexts
    plt.rcParams.update({'font.size': 14})
    
    # --- CHANGED: Create 3 Subplots (1x3) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(26, 8))
    
    # ==========================================
    # PLOT 1: GAIN EVOLUTION
    # ==========================================
    for col in gain_cols:
        ax1.plot(df.index, df[col], linewidth=2, label=col)
    
    ax1.set_title("Gain Evolution", fontsize=18, fontweight='bold')
    ax1.set_xlabel("Time Step", fontsize=16)
    ax1.set_ylabel("Gain Value", fontsize=16)
    # ax1.legend(fontsize=12) # Optional: comment out if too crowded
    ax1.grid(True, linestyle=':', alpha=0.4)

    # ==========================================
    # PLOT 2: ERROR COMPARISON
    # ==========================================
    ax2.plot(df.index, df['Error (Fixed W0)'], color='k',  
             linewidth=1.5, alpha=0.7, label='Error (Fixed $W_0$)')
    
    ax2.plot(df.index, df['Error (Adapted W)'], color='tab:red', 
             linewidth=2.5, label='Error (Adapted $W$)')
    
    ax2.set_title("Whitening Error", fontsize=18, fontweight='bold')
    ax2.set_xlabel("Time Step", fontsize=16)
    ax2.set_ylabel("Covariance Error (Log Scale)", fontsize=16)
    ax2.set_yscale('log')
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.4)

    # Add Vertical Context Lines to Plot 1 & 2
    for k in range(1, num_contexts):
        x_pos = k * steps_per_context
        line_props = {'color': 'gray', 'linewidth': 2, 'alpha': 0.6}
        ax1.axvline(x=x_pos, **line_props)
        ax2.axvline(x=x_pos, **line_props)

    # ==========================================
    # PLOT 3: FRAME ALIGNMENT (VECTORS)
    # ==========================================
    if os.path.exists(npz_file):
        data = np.load(npz_file)
        W_init = data['W_init']
        W_final = data['W_final']
        V_ideal = data['V_ideal']

        # Draw Unit Circle
        circle = plt.Circle((0, 0), 1, color='lightgray', fill=False, linestyle='-', linewidth=1.5)
        ax3.add_patch(circle)

        # 1. Plot Ideal Axes (V) - Grey Dashed
        # We assume V is (Dim, K) or (Dim, Dim)
        for i in range(V_ideal.shape[1]):
            vec = V_ideal[:, i]
            # Normalize for visualization if not already
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            
            label = "Ideal Axes ($V$)" if i == 0 else None
            # Plot line from center
            ax3.plot([0, vec[0]], [0, vec[1]], color='gray', linestyle='--', linewidth=2.5, label=label)

        # 2. Plot Initial Weights (W0) - Solid Blue
        for i in range(W_init.shape[1]):
            vec = W_init[:, i]
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            
            label = "Init Weights ($W_0$)" if i == 0 else None
            ax3.plot([0, vec[0]], [0, vec[1]], color='tab:blue', linestyle='-', linewidth=2, alpha=0.6, label=label)

        # 3. Plot Final Weights (W) - Solid Red
        for i in range(W_final.shape[1]):
            vec = W_final[:, i]
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            
            label = "Final Weights ($W$)" if i == 0 else None
            # Use arrow for final weights to show direction clearly
            ax3.arrow(0, 0, vec[0], vec[1], color='tab:red', alpha=1.0, 
                      width=0.015, head_width=0.05, length_includes_head=True, zorder=5, label=label)

        ax3.set_title("Frame Alignment", fontsize=18, fontweight='bold')
        ax3.set_xlabel("Dimension 1", fontsize=16)
        ax3.set_ylabel("Dimension 2", fontsize=16)
        ax3.set_xlim(-1.1, 1.1)
        ax3.set_ylim(-1.1, 1.1)
        ax3.set_aspect('equal') # Crucial for vector plots
        ax3.grid(True, linestyle=':', alpha=0.4)
        ax3.legend(fontsize=12, loc='upper right')
    
    else:
        ax3.text(0.5, 0.5, "frame_data.npz not found", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)

    plt.tight_layout()
    plt.savefig('whiten_comparison.png')
    print("Plot saved as 'whiten_comparison.png'")

if __name__ == "__main__":
    plot_whitening_comparison()