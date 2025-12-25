import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_whitening_comparison(csv_file="white2_output.csv", num_contexts=5):
    # 1. Load Data (No header in your file)
    df = pd.read_csv(csv_file, header=None)
    
    # 2. Dynamic Column Detection
    # We know the LAST two columns are errors. The rest are Gains.
    num_cols = df.shape[1]
    num_gains = num_cols - 2 
    
    # Create readable column names
    gain_cols = [f'Gain {i+1}' for i in range(num_gains)]
    col_names = gain_cols + ['Error (Adapted W)', 'Error (Fixed W0)']
    df.columns = col_names
    
    # 3. Setup Plotting Parameters
    total_steps = len(df)
    steps_per_context = total_steps // num_contexts
    
    # Increase font sizes globally for "Bigger Labels"
    plt.rcParams.update({'font.size': 14})
    
    # Create Subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- Plot 1: Gains ---
    for col in gain_cols:
        ax1.plot(df.index, df[col], linewidth=2, label=col)
    
    ax1.set_title("Gain Evolution", fontsize=18, fontweight='bold')
    ax1.set_xlabel("Time Step", fontsize=16)
    ax1.set_ylabel("Gain Value", fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.4)

    # --- Plot 2: Error Comparison ---
    # Plot Fixed W0 error first (background)
    ax2.plot(df.index, df['Error (Fixed W0)'], color='k',  
             linewidth=1.5, alpha=0.7, label='Error (Fixed $W_0$)')
    
    # Plot Adapted W error (foreground)
    ax2.plot(df.index, df['Error (Adapted W)'], color='tab:red', 
             linewidth=2.5, label='Error (Adapted $W$)')
    
    ax2.set_title("Whitening error", fontsize=18, fontweight='bold')
    ax2.set_xlabel("Time Step", fontsize=16)
    ax2.set_ylabel("Covariance Error (Log Scale)", fontsize=16)
    ax2.set_yscale('log')
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.4)

    # --- Vertical Context Lines (No Text) ---
    for k in range(1, num_contexts):
        x_pos = k * steps_per_context
        
        # Solid opaque grey line
        line_props = {'color': 'gray', 'linewidth': 2, 'alpha': 0.6}
        
        ax1.axvline(x=x_pos, **line_props)
        ax2.axvline(x=x_pos, **line_props)

    plt.tight_layout()
    plt.savefig('whiten_comparison.png')
    print("Plot saved as 'whiten_comparison.png'")

if __name__ == "__main__":
    plot_whitening_comparison()