import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv("infedit_paper_vs_mine.csv")

# Clean column names in case of extra spaces
df.columns = df.columns.str.strip()

# Extract columns
metrics = df["Metric"]
paper = df["InfEdit Paper (VI* UAC)"]
mine = df["My Reproduction"]
diff = df["Difference (Mine - Paper)"]

# -------------------------
# Plot 1: Paper vs Mine
# -------------------------
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, paper, width, label="Paper")
plt.bar(x + width/2, mine, width, label="My Reproduction")

plt.xticks(x, metrics, rotation=20, ha="right")
plt.ylabel("Metric Value")
plt.title("InfEdit: Paper vs Our Reproduction")
plt.legend()
plt.tight_layout()
plt.show()