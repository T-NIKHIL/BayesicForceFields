"""Recover the leave-one-region-out metrics from the saved fold models.

fit_ctd2_gp_loso ran all eight folds and saved fold<NN>.lgp models, but
stopped before writing the per-fold predictions and the pooled report.
This script reloads the eight models, reproduces the median-split region
definitions, predicts each fold's held-out region, and writes what the
original run was missing: fold<NN>_predictions.csv, loso_predictions.csv,
and metrics.json. No refitting - predictions come from the saved
hyperparameters, and each fold's training set is verified to be the exact
complement of its held-out region.

Run from the repository root:

    python -m examples.calvados_gmx.ctd2.collect_loso
"""

import json
from pathlib import Path

import numpy as np
import torch

from bff.bayes.gaussian_process import LGPCommittee
from bff.qoi.dataset import QoIDataset
from examples.calvados_gmx.ctd2.fit_ctd2_gp import (
    active_columns,
    regression_metrics,
)
from examples.calvados_gmx.ctd2.fit_ctd2_gp_loso import (
    CAMPAIGN,
    LOSO,
    regions,
    split_dimensions,
)


def sorted_rows(a):
    return a[np.lexsort(a.T[::-1])]


def main():
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

    rows, pooled_obs, pooled_pred = [], [], []
    for fold in range(8):
        holdout = np.where(labels == fold)[0]
        model = LGPCommittee.load(str(LOSO / f"fold{fold}.lgp"))
        expected_train = np.setdiff1d(np.arange(len(x12)), holdout)
        np.testing.assert_array_equal(
            sorted_rows(model.members[0].X_train.numpy()),
            sorted_rows(x12[expected_train]),
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

    pooled_obs = np.concatenate(pooled_obs)
    pooled_pred = np.concatenate(pooled_pred)
    pooled = regression_metrics(pooled_obs, pooled_pred)
    random_split = json.loads((CAMPAIGN / "fit8020" / "metrics.json").read_text())
    (LOSO / "metrics.json").write_text(
        json.dumps(
            {
                "split_dimensions": [name for name, _ in correlations],
                "split_correlations": correlations,
                "region_sizes": counts.tolist(),
                "folds": rows,
                "pooled_loso": pooled,
                "random_8020": random_split["test"],
            },
            indent=2,
        )
        + "\n"
    )
    np.savetxt(
        LOSO / "loso_predictions.csv",
        np.column_stack([
            np.concatenate([np.where(labels == fold)[0] for fold in range(8)]),
            pooled_obs,
            pooled_pred,
        ]),
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
