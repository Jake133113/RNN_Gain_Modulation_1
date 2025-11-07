import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV
df = pd.read_csv("white_output3.csv")

# --- Set up 3-row, 1-column subplot ---
fig, axes = plt.subplots(3, 1, figsize=(6, 8))
fig.suptitle("Gain Modulated Statistical Whitening", fontsize=18, fontweight="bold")

# Common style settings
line_width = 2.2
label_fontsize = 14
label_fontweight = "bold"
tick_fontsize = 12

# --- Plot 1: first three columns ---
ax = axes[0]
for i in range(min(3, df.shape[1])):  # handle smaller CSVs safely
    ax.plot(df.index, df.iloc[:, i], label=f"g_{i+1}", linewidth=line_width)
ax.set_ylabel("Gains", fontsize=label_fontsize, fontweight=label_fontweight)
ax.set_xlabel("")  # remove x-label for upper plots
ax.legend(fontsize=10, loc='upper right')
ax.tick_params(axis='both', labelsize=tick_fontsize)
ax.grid(False)

# --- Plot 2: next three columns ---
if df.shape[1] > 3:
    ax = axes[1]
    for i in range(3, min(6, df.shape[1])):
        ax.plot(df.index, df.iloc[:, i], label=f"z_{i+1}", linewidth=line_width)
    ax.set_ylabel("Variance", fontsize=label_fontsize, fontweight=label_fontweight)
    ax.set_xlabel("")
    ax.set_ylim(0, 2)
    ax.legend(fontsize=10)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.grid(False)
else:
    axes[1].axis("off")

# --- Plot 3: seventh column (||C_rr - I||op) ---
if df.shape[1] > 6:
    ax = axes[2]
    ax.plot(df.index, df.iloc[:, 6], linewidth=line_width)
    ax.set_yscale('log')
    ax.set_ylabel("||C_rr - I||op", fontsize=label_fontsize, fontweight=label_fontweight)
    ax.set_xlabel("t", fontsize=label_fontsize, fontweight=label_fontweight)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.grid(False)
else:
    axes[2].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for main title
plt.show()
