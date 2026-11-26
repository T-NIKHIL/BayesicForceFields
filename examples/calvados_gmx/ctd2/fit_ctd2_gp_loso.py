"""Leave-one-region-out evaluation of the CTD2 Rg surrogate.

Regions are the eight axis-aligned blocks obtained by median-splitting
the three most Rg-correlated active lambda dimensions of the 512-sample
Sobol campaign (~64 samples each). Each fold fits the BFF GP on seven
regions and predicts the held-out one - a far harsher test than the
random 80/20 split, because every held-out point is farther from the
training data than any point in a random split.

fit_surrogates only exposes a test fraction, so each fold reorders the
dataset rows so that the held-out region occupies exactly the positions
of fit_surrogates' deterministic internal split (randperm with the same
seed; the random-split scripts assert this equivalence). An assertion
per fold verifies the training set is the exact complement.

Outputs in ``campaign_ctd2/loso/``: per-fold predictions CSV, metrics
JSON, and the pooled comparison against the random 80/20 split.

Run from the repository root:

    python -m examples.calvados_gmx.ctd2.fit_ctd2_gp_loso [first_fold last_fold]
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bff.bayes.fit import fit_surrogates
from bff.io.logs import Logger
from bff.qoi.dataset import QoIDataset
from examples.calvados.fetch_data import RAW, verified_bytes
from examples.calvados_gmx import run_pilot_openmm as rp
from examples.calvados_gmx.ctd2.fit_ctd2_gp import (
    N_HYPER_MAX,
    SEED,
    TEST_FRACTION,
    active_columns,
    regression_metrics,
)

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / "campaign_ctd2"
LOSO = CAMPAIGN / "loso"
N_DIMS_SPLIT = 3


def split_dimensions(x12, names):
    """The N_DIMS_SPLIT active dimensions most correlated with mean Rg."""
    y = QoIDataset.load(str(CAMPAIGN / "rg_ctd2.pt")).y[:, 0]
    correlations = np.array([np.corrcoef(x12[:, c], y)[0, 1] for c in range(x12.shape[1])])
    order = np.argsort(-np.abs(correlations))[:N_DIMS_SPLIT]
    return order, [(names[i], float(correlations[i])) for i in order]


def regions(x12, dimensions):
    """Eight binary region labels from median splits of the given dims."""
    labels = np.zeros(len(x12), dtype=int)
    for bit, dim in enumerate(dimensions):
        median = np.median(x12[:, dim])
        labels |= (x12[:, dim] > median).astype(int) << bit
    return labels


def main():
    import sys

    first, last = 0, 8
    if len(sys.argv) == 3:
        first, last = int(sys.argv[1]), int(sys.argv[2])
    LOSO.mkdir(parents=True, exist_ok=True)

    dataset = QoIDataset.load(str(CAMPAIGN / "rg_ctd2.pt"))
    active = active_columns(dataset.parameter_names)
    x12 = dataset.X[:, active]
    y = dataset.y[:, 0]
    names = [dataset.parameter_names[i] for i in active]

    dims, correlations = split_dimensions(x12, names)
    labels = regions(x12, dims)
    counts = np.bincount(labels)
    print(
        f"split dims: {correlations} | region sizes: {counts.tolist()}",
        flush=True,
    )
    assert len(set(labels.tolist())) == 8
    assert counts.min() >= 32, "unbalanced regions"

    n = len(x12)
    fold_size = int(counts[labels[0]])
    fraction = fold_size / n
    # fit_surrogates' internal test positions under the shared seed
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).numpy()

    rows = []
    pooled_obs, pooled_pred = [], []
    for fold in range(first, last):
        holdout = np.where(labels == fold)[0]
        # row order: held-out samples at the internal split's test positions
        order = np.empty(n, dtype=int)
        order[perm[:fold_size]] = holdout
        rest = [i for i in np.where(labels != fold)[0]]
        order[perm[fold_size:]] = rest

        trimmed = QoIDataset(
            name="rg_ctd2_loso",
            X=x12[order],
            y=dataset.y[order],
            y_ref=dataset.y_ref,
            nuisance=dataset.nuisance,
            labels=dataset.labels,
            parameter_names=tuple(names),
            sample_ids=dataset.sample_ids,
            metadata={**dataset.metadata, "loso_fold": fold},
        )
        torch.manual_seed(SEED)
        logger = Logger("fit-lgp", str(LOSO / f"fold{fold}.log"), mode="w")
        model = fit_surrogates(
            [trimmed],
            model_paths={"rg_ctd2_loso": LOSO / f"fold{fold}.lgp"},
            reuse_models=False,
            n_hyper_max=N_HYPER_MAX,
            committee_size=1,
            test_fraction=fraction,
            logger=logger,
            max_iter=500,
        )["rg_ctd2_loso"]

        expected_train = np.setdiff1d(np.arange(n), holdout)
        np.testing.assert_array_equal(
            np.sort(model.members[0].X_train.numpy().tolist()),
            np.sort(x12[expected_train].tolist()),
        )
        predicted = model.predict(torch.tensor(x12[holdout])).numpy()[:, 0]
        observed = y[holdout]
        metrics = regression_metrics(observed, predicted)
        metrics.update(fold=fold, n_holdout=int(len(holdout)))
        rows.append(metrics)
        pooled_obs.append(observed)
        pooled_pred.append(predicted)
        np.savetxt(
            LOSO / f"fold{fold}_predictions.csv",
            np.column_stack([holdout, observed, predicted]),
            delimiter=",",
            header="sample_index,observed_rg_nm,predicted_rg_nm",
            comments="",
            fmt=["%d", "%.4f", "%.4f"],
        )
        print(
            f"fold {fold}: RMSE {metrics['rmse_nm']:.4f} nm, "
            f"R2 {metrics['r2']:.3f}, max {metrics['max_abs_error_nm']:.4f} nm",
            flush=True,
        )

    if last - first == 8:
        pooled_obs = np.concatenate(pooled_obs)
        pooled_pred = np.concatenate(pooled_pred)
        pooled = regression_metrics(pooled_obs, pooled_pred)
        random_split = json.loads((CAMPAIGN / "fit8020" / "metrics.json").read_text())
        report = {
            "split_dimensions": [name for name, _ in correlations],
            "split_correlations": correlations,
            "region_sizes": counts.tolist(),
            "folds": rows,
            "pooled_loso": pooled,
            "random_8020": random_split["test"],
        }
        (LOSO / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savetxt(
            LOSO / "loso_predictions.csv",
            np.column_stack([np.concatenate(
                [np.where(labels == fold)[0] for fold in range(8)]), pooled_obs, pooled_pred]),
            delimiter=",",
            header="sample_index,observed_rg_nm,predicted_rg_nm",
            comments="",
            fmt=["%d", "%.4f", "%.4f"],
        )
        print(
            f"pooled LOSO: RMSE {pooled['rmse_nm']:.4f} nm, "
            f"R2 {pooled['r2']:.3f} vs random 80/20: "
            f"RMSE {random_split['test']['rmse_nm']:.4f} nm, "
            f"R2 {random_split['test']['r2']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
