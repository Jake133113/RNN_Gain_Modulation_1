import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV
df = pd.read_csv("white_output3.csv")

# --- Plot 1: first three columns ---
plt.figure(figsize=(8, 4))
for i in range(min(3, df.shape[1])):  # handle smaller CSVs safely
    plt.plot(df.index, df.iloc[:, i], label=f"col {i+1}")
plt.title("First Three Columns (Gains)")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: secnextond three columns ---
if df.shape[1] > 3:
    plt.figure(figsize=(8, 4))
    for i in range(3, min(6, df.shape[1])):
        plt.plot(df.index, df.iloc[:, i], label=f"col {i+1}")
    plt.title("Next Three Columns (Z-values)")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# --- Plot 1: first three columns ---
plt.figure(figsize=(8, 4))
  # handle smaller CSVs safely
plt.plot(df.index, df.iloc[:, df.shape[1]])
plt.title("C_rr - Identity")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()
