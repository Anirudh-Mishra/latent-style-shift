import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

infedit = pd.read_csv("infedit_paper_vs_mine.csv")
pix2pix = pd.read_csv("pix2pix_paper_vs_mine.csv")

infedit.columns = infedit.columns.str.strip()
pix2pix.columns = pix2pix.columns.str.strip()

df = pd.DataFrame({
    "Metric": pix2pix["Metric"],
    "Pix2Pix (My Reproduction)": pix2pix["My Reproduction"],
    "InfEdit (My Reproduction)": infedit["My Reproduction"],
    "InfEdit (Paper)": infedit["InfEdit Paper (VI* UAC)"]
})

print(df)

x = np.arange(len(df["Metric"]))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, df["Pix2Pix (My Reproduction)"], width, label="Pix2Pix (Our Reproduction)")
plt.bar(x, df["InfEdit (My Reproduction)"], width, label="InfEdit (Our Reproduction)")
plt.bar(x + width, df["InfEdit (Paper)"], width, label="InfEdit (Paper)")

plt.xticks(x, df["Metric"], rotation=20, ha="right")
plt.ylabel("Metric Value")
plt.title("Pix2Pix vs InfEdit on PIE-Bench")
plt.legend()
plt.tight_layout()
plt.show()