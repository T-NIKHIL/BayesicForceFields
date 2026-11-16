"""Fit the CTD2 Rg surrogate to the 512-sample Sobol campaign (12-D).

Loads ``campaign_ctd2/rg_ctd2.pt`` (written by run_ctd2_campaign), extracts
the 12 active lambda columns - the residue types present in CTD2's
sequence; the other 8 columns are pinned at the optimized M1_24 values in
every sample and therefore carry no information - and fits one BFF GP
(single committee member) to the 12-dimensional vectors with mean Rg
outputs. Reports held-out metrics on a random 80/20 split.

Outputs in ``campaign_ctd2/fit<NNNN>/`` (NNNN = split tag, e.g. fit8020
for the default 80/20; pass a fraction to override, e.g. ``0.1`` for
90/10): the ``.lgp`` model, fit log, split manifest, test predictions
CSV, metrics.json, and the held-out scatter.

Run from the repository root:

    python -m examples.calvados_gmx.fit_ctd2_gp          # 80/20
    python -m examples.calvados_gmx.fit_ctd2_gp 0.1      # 90/10
"""

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from bff.bayes.fit import fit_surrogates  # noqa: E402
from bff.io.logs import Logger  # noqa: E402
from bff.qoi.dataset import QoIDataset  # noqa: E402
from examples.calvados.fetch_data import RAW, verified_bytes  # noqa: E402
from examples.calvados_gmx import run_pilot_openmm as rp  # noqa: E402

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / "campaign_ctd2"
FIT = CAMPAIGN / "fit8020"
SEED = 2026
TEST_FRACTION = 0.2
N_HYPER_MAX = 80


def active_columns(parameter_names):
    """Indices of the 20-column parameter list that CTD2 actually samples.

    The campaign varies only the residue types present in the sequence
    (12 of 20); the rest are pinned at the optimized values in every
    row and are dropped before fitting.
    """
    residues = pd.read_csv(io.BytesIO(verified_bytes(RAW / "residues.csv")))
    residues = residues.set_index("one")
    one_order = sorted(residues.index)
    sequence, _, _ = rp.type_sequence()
    present = sorted(set(sequence))
    active = [one_order.index(one) for one in present]
    names = [parameter_names[index] for index in active]
    if sorted(names) != sorted(set(names)):
        raise ValueError("CTD2 GP fit: duplicate active parameter names")
    return active


def regression_metrics(observed, predicted):
    residual = predicted - observed
    return {
        "rmse_nm": float(np.sqrt(np.mean(residual**2))),
        "mae_nm": float(np.mean(np.abs(residual))),
        "r2": float(
            1 - np.sum(residual**2) / np.sum((observed - observed.mean()) ** 2)
        ),
        "max_abs_error_nm": float(np.max(np.abs(residual))),
    }


def main():
    import sys

    fraction = float(sys.argv[1]) if len(sys.argv) > 1 else TEST_FRACTION
    tag = f"{int(round((1 - fraction) * 100)):02d}{int(round(fraction * 100)):02d}"
    fit_dir = CAMPAIGN / f"fit{tag}"
    torch.set_num_threads(4)
    fit_dir.mkdir(parents=True, exist_ok=True)
    dataset = QoIDataset.load(str(CAMPAIGN / "rg_ctd2.pt"))
    n = dataset.n_samples
    active = active_columns(dataset.parameter_names)
    print(f"dataset: {n} samples, {len(active)}/20 active parameters")
    if n != 512:
        print(f"warning: expected 512 samples, got {n}")

    x = dataset.X[:, active]
    y = dataset.y[:, 0]

    indices = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    n_test = int(n * fraction)
    test_indices = indices[:n_test].numpy()
    train_indices = indices[n_test:].numpy()

    # A trimmed dataset in the 12 active columns keeps the fit and the
    # saved model consistent with the parameters that actually vary.
    trimmed = QoIDataset(
        name=dataset.name,
        X=x,
        y=dataset.y,
        y_ref=dataset.y_ref,
        nuisance=dataset.nuisance,
        labels=dataset.labels,
        parameter_names=tuple(
            dataset.parameter_names[index] for index in active
        ),
        sample_ids=dataset.sample_ids,
        metadata={
            **dataset.metadata,
            "active_columns": active,
            "note": "20-column campaign trimmed to the 12 sampled residues",
        },
    )
    trimmed.write(str(fit_dir / "rg_ctd2_12d.pt"))

    torch.manual_seed(SEED)
    logger = Logger("fit-lgp", str(fit_dir / "rg_ctd2_12d.log"), mode="w")
    model = fit_surrogates(
        [trimmed],
        model_paths={"rg_ctd2": fit_dir / "rg_ctd2_12d.lgp"},
        reuse_models=False,
        n_hyper_max=N_HYPER_MAX,
        committee_size=1,
        test_fraction=fraction,
        logger=logger,
        max_iter=500,
    )["rg_ctd2"]

    np.testing.assert_array_equal(
        model.members[0].X_train.numpy(), x[train_indices]
    )
    predicted = model.predict(torch.tensor(x[test_indices])).numpy()[:, 0]
    observed = y[test_indices]
    baseline = np.full_like(observed, y[train_indices].mean())
    metrics = {
        "seed": SEED,
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "active_parameters": [
            dataset.parameter_names[index] for index in active
        ],
        "n_hyper_max": N_HYPER_MAX,
        "test": regression_metrics(observed, predicted),
        "baseline_train_mean": regression_metrics(observed, baseline),
        "units": "nm (mass-weighted Rg, frame mean over 500 ns)",
    }
    for name, block in (
        ("GP", metrics["test"]),
        ("training-mean baseline", metrics["baseline_train_mean"]),
    ):
        print(
            f"{name}: RMSE {block['rmse_nm']:.4f} nm, "
            f"MAE {block['mae_nm']:.4f} nm, R2 {block['r2']:.3f}, "
            f"max err {block['max_abs_error_nm']:.4f} nm"
        )
    (fit_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savetxt(
        fit_dir / "test_predictions.csv",
        np.column_stack([test_indices, observed, predicted]),
        delimiter=",",
        header="test_index,observed_rg_nm,predicted_rg_nm",
        comments="",
        fmt=["%d", "%.4f", "%.4f"],
    )
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(observed, predicted, s=25, alpha=0.7)
    limits = [min(observed.min(), predicted.min()) - 0.05,
              max(observed.max(), predicted.max()) + 0.05]
    axis.plot(limits, limits, "k--", linewidth=1)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("observed mean Rg (nm)")
    axis.set_ylabel("predicted mean Rg (nm)")
    axis.set_title(
        f"CTD2 held-out ({int(fraction * 100)}%): "
        f"RMSE {metrics['test']['rmse_nm']:.3f} nm, "
        f"R2 = {metrics['test']['r2']:.2f}"
    )
    figure.tight_layout()
    figure.savefig(fit_dir / f"held_out_predictions_{tag}.png", dpi=160)
    print(f"outputs in {fit_dir}")


if __name__ == "__main__":
    main()
