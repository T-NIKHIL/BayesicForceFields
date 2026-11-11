"""Bar plot per IDP: percent of each amino acid along the x axis.

Run with the bff env:
    /Users/nikhilthota/miniconda3/envs/bff/bin/python plot_composition.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
table = pd.read_csv(HERE / "idp_composition.csv")
percent_columns = [c for c in table.columns if c.endswith("_percent")]  # 20 of them
amino_acids = [c[:-8] for c in percent_columns]
x = range(len(amino_acids))

figure, axes = plt.subplots(4, 13, figsize=(39, 11), sharex=True)
for axis, (_, row) in zip(axes.ravel(), table.iterrows()):
    axis.bar(x, [row[column] for column in percent_columns], color="steelblue", width=0.8)
    axis.set_title(
        f"{row['protein_id']} ({row['length']} aa)", fontsize=14, fontweight="bold"
    )
    axis.set_xticks(x)
    axis.set_xticklabels(amino_acids, fontsize=8)
    axis.tick_params(axis="y", labelsize=8)

figure.suptitle("Amino-acid composition of the 51 M1_24 IDPs", fontsize=16)
figure.tight_layout(rect=[0, 0, 1, 0.97])
figure.savefig(HERE / "idp_composition.png", dpi=150)
print(f"wrote {HERE / 'idp_composition.png'}")

