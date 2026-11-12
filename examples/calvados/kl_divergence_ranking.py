"""Rank the 51 M1_24 IDPs by divergence of their amino-acid compositions.

Two families of scores per IDP:

- Pairwise (primary): the mean Jeffreys divergence to every other IDP,
  J(p_i, p_j) = KL(p_i || p_j) + KL(p_j || p_i), averaged over the other
  50 compositions. This measures how distinct one IDP's composition is
  from the other IDPs themselves, not from a summary of the set. The mean
  one-directional KL(p_i || others) is kept as a secondary column.
- Reference-based (kept for comparison): KL against the population-mean
  composition and against the uniform 20-column distribution.

Compositions are smoothed with a 0.5 Jeffreys pseudocount per amino acid
before normalizing, matching pairwise_kl_corner.py: without it most
pairwise divergences are infinite because these sparse histograms have
many zeros (tryptophan above all).

Writes kl_divergence_ranking.csv and the ranked bar chart
kl_divergence_ranking.png, and prints the top 10 most distinct IDPs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SMOOTHING = 0.5  # Jeffreys pseudocount per amino acid; see module docstring


def kl_divergence(p, q):
    """KL(p || q) in nats for strictly positive p and q."""
    mask = p > 0
    if np.any(q[mask] == 0):
        raise ValueError("KL divergence needs strictly positive q; smooth first")
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def main():
    table = pd.read_csv(HERE / "train_set_idp_composition.csv")
    counts = table[[f"{aa}_count" for aa in AMINO_ACIDS]].to_numpy(float) + SMOOTHING
    compositions = counts / counts.sum(axis=1, keepdims=True)
    identifiers = list(table["protein_id"])
    n = len(identifiers)
    reference_mean = compositions.mean(axis=0)
    reference_uniform = np.full(len(AMINO_ACIDS), 1.0 / len(AMINO_ACIDS))

    # Symmetric pairwise matrix of Jeffreys divergences, then per-IDP means.
    jeffreys = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                jeffreys[i, j] = kl_divergence(
                    compositions[i], compositions[j]
                ) + kl_divergence(compositions[j], compositions[i])
    mean_jeffreys = (jeffreys.sum(axis=1)) / (n - 1)
    mean_kl_out = np.array(
        [
            np.mean(
                [kl_divergence(compositions[i], compositions[j]) for j in range(n) if j != i]
            )
            for i in range(n)
        ]
    )

    rows = []
    for index, protein_id in enumerate(identifiers):
        rows.append(
            {
                "protein_id": protein_id,
                "length": int(table["length"].iloc[index]),
                "mean_pairwise_jeffreys": mean_jeffreys[index],
                "mean_kl_to_others": mean_kl_out[index],
                "kl_vs_mean": kl_divergence(compositions[index], reference_mean),
                "kl_vs_uniform": kl_divergence(
                    compositions[index], reference_uniform
                ),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        "mean_pairwise_jeffreys", ascending=False, ignore_index=True
    )
    ranking.insert(1, "rank_pairwise", ranking.index + 1)
    for column in ("kl_vs_mean", "kl_vs_uniform"):
        ranking[f"rank_{column}"] = (
            ranking[column].rank(ascending=False, method="first").astype(int)
        )
    ranking.to_csv(HERE / "kl_divergence_ranking.csv", index=False)
    print(ranking.head(10).to_string(index=False))

    figure, axis = plt.subplots(figsize=(16, 5))
    colors = ["#d62728" if rank < 10 else "steelblue" for rank in ranking["rank_pairwise"]]
    axis.bar(range(n), ranking["mean_pairwise_jeffreys"], color=colors, width=0.8)
    axis.set_xticks(range(n))
    axis.set_xticklabels(ranking["protein_id"], rotation=90, fontsize=7)
    axis.set_ylabel("mean Jeffreys divergence to the other 50 IDPs (nats)")
    axis.set_title(
        "Pairwise amino-acid composition divergence, 51 M1_24 IDPs\n"
        "each bar: mean of J(p, q) = KL(p||q) + KL(q||p) against every other "
        "IDP; red = top 10"
    )
    figure.tight_layout()
    figure.savefig(HERE / "kl_divergence_ranking.png", dpi=200)
    plt.close(figure)
    print(f"\nfull ranking: {HERE / 'kl_divergence_ranking.csv'}")
    print(f"plot: {HERE / 'kl_divergence_ranking.png'}")


if __name__ == "__main__":
    main()
