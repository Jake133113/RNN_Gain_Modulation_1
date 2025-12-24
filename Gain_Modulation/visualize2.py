import pandas as pd
import matplotlib.pyplot as plt

def plot_whitening_results(csv_file="white2_output.csv"):
    # 1. Load Data
    # Assuming the file has no text headers, just numbers. 
    # If your file has a header row (0, 1, 2), change header=None to header=0
    df = pd.read_csv(csv_file, header=0)
    
    # Rename columns for clarity (assuming 3 cols: Gain1, Gain2, Error)
    df.columns = ['Gain 1', 'Gain 2', 'Error']
    
    # 2. Setup Context Lines
    # We infer the switch points based on total length and known number of contexts (5)
    NUM_CONTEXTS = 5
    total_steps = len(df)
    steps_per_context = total_steps // NUM_CONTEXTS
    
    # 3. Create Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 1: Gains vs Time ---
    ax1.plot(df.index, df['Gain 1'], label='Gain Neuron 1', alpha=0.9)
    ax1.plot(df.index, df['Gain 2'], label='Gain Neuron 2', alpha=0.9)
    
    ax1.set_title("Neural Gains Adaptation")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Gain Value")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Error vs Time ---
    ax2.plot(df.index, df['Error'], color='black', linewidth=1, label='Covariance Error')
    
    ax2.set_title("Whitening Error Convergence")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Error (Log Scale)")
    ax2.set_yscale('log') # Log scale helps see convergence better
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # --- Add Vertical Lines for Context Changes ---
    for k in range(1, NUM_CONTEXTS):
        switch_point = k * steps_per_context
        
        # Add line to Gain Plot
        ax1.axvline(x=switch_point, color='grey', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add line to Error Plot
        ax2.axvline(x=switch_point, color='grey', linestyle='--', alpha=0.7, linewidth=2)

    plt.tight_layout()
    plt.savefig('whiten_analysis.png')
    print("Plot saved as 'whiten_analysis.png'")

if __name__ == "__main__":
    plot_whitening_results()