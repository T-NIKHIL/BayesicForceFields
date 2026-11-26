"""Pairwise Jeffreys divergence between the 51 M1_24 IDP composition histograms.

Every off-diagonal cell shows the symmetric Jeffreys divergence between the
two compositions,

    J(p, q) = KL(p || q) + KL(q || p),

so the matrix is symmetric and no triangle distinction is needed. The
diagonal shows the one-directional divergence of each IDP from the
population-mean composition, KL(row || mean).

Compositions are smoothed with a 0.5 Jeffreys pseudocount per amino acid
before normalizing: raw histograms of these 51 proteins have many zeros
(tryptophan above all), which would make most pairwise divergences infinite.

Writes pairwise_kl_matrix.csv (full 51x51 matrix) and pairwise_kl_corner.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SMOOTHING = 0.5  # Jeffreys pseudocount per amino acid; see module docstring


def kl_divergence(p, q):
    """KL(p || q) in nats for finite nonnegative p, q; NaN on unsupported q."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.any(p < 0) or np.any(q < 0) or not np.isfinite(p).all():
        raise ValueError("KL divergence needs finite nonnegative inputs")
    mask = p > 0
    if np.any(q[mask] == 0):
        return np.nan
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def jeffreys_divergence(p, q):
    """Symmetric Jeffreys divergence J(p, q) = KL(p || q) + KL(q || p)."""
    forward = kl_divergence(p, q)
    reverse = kl_divergence(q, p)
    if np.isnan(forward) or np.isnan(reverse):
        return np.nan
    return forward + reverse


def main():
    table = pd.read_csv(HERE / "train_set_idp_composition.csv")
    counts = table[[f"{aa}_count" for aa in AMINO_ACIDS]].to_numpy(float) + SMOOTHING
    compositions = counts / counts.sum(axis=1, keepdims=True)
    identifiers = list(table["protein_id"])
    n = len(identifiers)
    reference_mean = compositions.mean(axis=0)

    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = kl_divergence(compositions[i], reference_mean)
            else:
                matrix[i, j] = jeffreys_divergence(
                    compositions[i], compositions[j]
                )
    print(f"maximum pairwise Jeffreys divergence: {np.nanmax(matrix):.3f} nats")

    pd.DataFrame(matrix, index=identifiers, columns=identifiers).to_csv(
        HERE / "pairwise_kl_matrix.csv"
    )

    figure, axis = plt.subplots(figsize=(13, 12))
    vmax = np.nanmax(matrix)
    mesh = axis.pcolormesh(
        matrix, cmap="viridis", vmin=0, vmax=vmax, edgecolors="white", linewidth=0.2
    )
    axis.set_xticks(np.arange(n) + 0.5)
    axis.set_xticklabels(identifiers, rotation=90, fontsize=6)
    axis.set_yticks(np.arange(n) + 0.5)
    axis.set_yticklabels(identifiers, fontsize=6)
    axis.invert_yaxis()
    axis.set_xlabel("IDP")
    axis.set_ylabel("IDP")
    axis.set_title(
        "Pairwise Jeffreys divergence of amino-acid compositions, 51 M1_24 IDPs\n"
        f"off-diagonal: J(p, q) = KL(p || q) + KL(q || p), symmetric; "
        f"diagonal: KL(row || population mean)",
        fontsize=10,
    )
    figure.colorbar(mesh, ax=axis, label="Jeffreys divergence (nats)")
    figure.tight_layout()
    figure.savefig(HERE / "pairwise_kl_corner.png", dpi=200)
    print(f"wrote {HERE / 'pairwise_kl_matrix.csv'} and pairwise_kl_corner.png")


if __name__ == "__main__":
    main()
