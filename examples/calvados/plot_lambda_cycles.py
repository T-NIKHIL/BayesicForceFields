"""How the 20 stickiness lambdas evolve over the M1_24 optimization cycles.

Two figures:

- lambda_cycle_evolution.png: one panel per cycle, the histogram (over the
  20 residues) of the starting vector in green and the optimized end
  vector in red, half transparent. x = lambda value, y = lambda density.
- lambda_cycle_trajectories.png: the end-of-cycle lambda value of each
  residue, one line per residue, x = cycle number (0 = M1 start, c = end
  of cycle c).

Cycle c starts from lambdas_{c-1} and ends at lambdas_c (11 transitions,
lambdas_0 through lambdas_11; the plain 'lambdas' column is an exact
alias of lambdas_11). Consecutive cycles share their boundary vector, so
only the first panel's green and the last panel's red are genuinely new
information.

Run with the bff env from the repository root:

    /Users/nikhilthota/miniconda3/envs/bff/bin/python -m examples.calvados.plot_lambda_cycles
"""

import io
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from examples.calvados.fetch_data import RAW, verified_bytes

HERE = Path(__file__).resolve().parent
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def cycle_vectors(table):
    """(start, end) column-name pairs for every optimization cycle.

    The stored history is lambdas_0 ... lambdas_11 (11 transitions). The
    plain 'lambdas' column is an exact alias of lambdas_11 (verified), so
    it contributes no extra transition.
    """
    numbered = sorted(
        (c for c in table.columns if c.startswith("lambdas_")),
        key=lambda name: int(name.split("_")[1]),
    )
    return [
        (numbered[index], numbered[index + 1])
        for index in range(len(numbered) - 1)
    ]


def trajectory_figure():
    """End-of-cycle lambda values per residue, across optimization cycles.

    x = cycle number (0 = the M1 start vector lambdas_0; c = the vector
    obtained at the end of cycle c, i.e. the optimized lambdas_c),
    y = lambda. One colored line per residue.
    """
    table = pd.read_csv(io.BytesIO(verified_bytes(RAW / "residues.csv")))
    table = table.set_index("one").loc[AMINO_ACIDS]
    cycle_columns = [f"lambdas_{cycle}" for cycle in range(12)]
    cycles = np.arange(12)
    figure, axis = plt.subplots(figsize=(12, 7))
    # Residue-class colors: aliphatic red, negative blue, amidic orange,
    # hydroxylic pink, aromatic dark green, positive light blue, sulfur
    # yellow; three-letter side-chain names placed at the line ends.
    residue_colors = {
        "A": "red", "V": "red", "L": "red", "I": "red",          # aliphatic
        "D": "darkblue", "E": "darkblue",                        # negative
        "N": "orange", "Q": "orange",                            # amidic
        "P": "pink", "S": "pink", "T": "pink",                   # hydroxylic
        "F": "darkgreen", "W": "darkgreen", "Y": "darkgreen",    # aromatic
        "K": "lightskyblue", "R": "lightskyblue", "H": "lightskyblue",  # positive
        "C": "goldenrod", "M": "goldenrod",                      # sulfur
        "G": "gray",                                             # glycine
    }
    for residue in table.index:
        values = table.loc[residue, cycle_columns].to_numpy(float)
        axis.plot(
            cycles,
            values,
            color=residue_colors[residue],
            marker="o",
            markersize=3.5,
            linewidth=2.0,
            alpha=0.85,
        )
        axis.annotate(
            table.loc[residue, "three"],
            (cycles[-1], values[-1]),
            textcoords="offset points",
            xytext=(6, -3),
            fontsize=8,
            color=residue_colors[residue],
            fontweight="bold",
        )
    axis.set_xlabel("optimization cycle (0 = M1 start; c = end of cycle c)", fontsize=11)
    axis.set_ylabel("lambda", fontsize=11)
    axis.set_xticks(cycles)
    axis.set_xlim(cycles[0], cycles[-1] + 1.1)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    legend_handles = [
        plt.Line2D([], [], color=color, linewidth=2.5, label=label)
        for label, color in (
            ("aliphatic (A, V, L, I)", "red"),
            ("negative (D, E)", "darkblue"),
            ("amidic (N, Q)", "orange"),
            ("hydroxylic (P, S, T)", "pink"),
            ("aromatic (F, W, Y)", "darkgreen"),
            ("positive (K, R, H)", "lightskyblue"),
            ("sulfur (C, M)", "goldenrod"),
            ("glycine", "gray"),
        )
    ]
    axis.legend(handles=legend_handles, fontsize=8, loc="lower left")
    axis.set_title(
        "M1_24 stickiness lambdas at the end of each optimization cycle\n"
        "colored by residue class; three-letter side-chain name at line end",
        fontsize=13,
    )
    figure.tight_layout()
    output = HERE / "lambda_cycle_trajectories.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"wrote {output}")


def main():
    table = pd.read_csv(io.BytesIO(verified_bytes(RAW / "residues.csv")))
    table = table.set_index("one").loc[AMINO_ACIDS]
    pairs = cycle_vectors(table)

    figure, axes = plt.subplots(3, 4, figsize=(17, 11), sharex=True, sharey=True)
    bins = np.linspace(0.0, 1.0, 21)
    for cycle, ((start_column, end_column), axis) in enumerate(
        zip(pairs, axes.ravel()), start=1
    ):
        start = table[start_column].to_numpy(float)
        end = table[end_column].to_numpy(float)
        axis.hist(start, bins=bins, density=True, color="green", alpha=0.45,
                  label=f"start ({start_column})")
        axis.hist(end, bins=bins, density=True, color="red", alpha=0.45,
                  label=f"end ({end_column})")
        axis.set_title(
            f"cycle {cycle}: mean {start.mean():.3f} -> {end.mean():.3f}",
            fontsize=11,
        )
        axis.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("lambda value", fontsize=10)
    for axis in axes[:, 0]:
        axis.set_ylabel("lambda density", fontsize=10)
    figure.suptitle(
        "M1_24 lambda optimization: distribution of the 20 stickiness "
        "parameters per cycle\nstart (green) vs end (red); consecutive "
        "cycles share their boundary vector",
        fontsize=13,
    )
    for axis in axes.ravel()[len(pairs):]:
        axis.axis("off")
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    output = HERE / "lambda_cycle_evolution.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"wrote {output} ({len(pairs)} cycles)")
    trajectory_figure()


if __name__ == "__main__":
    main()
