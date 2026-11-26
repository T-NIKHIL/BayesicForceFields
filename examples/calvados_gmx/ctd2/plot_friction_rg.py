"""Rg traces and autocorrelation across the friction correction.

Compares the three CTD2 Sobol vectors re-run at the corrected Langevin
friction of 0.01 ps^-1 (campaign_ctd2_friction_0_01, 10 ns equilibration
+ 100 ns production) against the same vectors from the original 512-
sample campaign at the misread friction of 1.16 ps^-1 (campaign_ctd2,
50 ns equilibration + 500 ns production). Equilibrium means are
friction-independent in exact Langevin dynamics, but the dynamics - and
with them the Rg relaxation time - scale with the inverse friction
(ratio 116), which the autocorrelation panel makes visible.

Outputs in campaign_ctd2_friction_0_01/:

- rg_traces.png     Rg(t) per vector, both frictions
- rg_acf.png        normalized Rg autocorrelation vs physical lag
- acf_summary.json  1/e relaxation times and the estimator setup
- comparison.json   rewritten: every vector, mean +- 1 ns-block SEM

Run from the repository root:

    python -m examples.calvados_gmx.ctd2.plot_friction_rg [indices ...]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from examples.calvados_gmx import run_pilot_openmm as rp  # noqa: E402
from examples.calvados_gmx.ctd2 import (  # noqa: E402
    run_ctd2_campaign as parent,
    run_ctd2_friction_campaign as corrected,
)

PARENT = parent.OUT
CORRECTED = corrected.OUT
FRAME_NS = parent.REPORT_INTERVAL * rp.DT / 1000  # 10 ps = 0.01 ns
SEGMENT_NS = 50.0
COLOR_NEW = "tab:blue"
COLOR_OLD = "tab:orange"
FRICTION_RATIO = 1.16 / rp.FRICTION_COEFF


def available(campaign):
    return sorted(
        int(p.name.split("_")[1])
        for p in campaign.glob("vec_*")
        if (p / "rg.txt").is_file()
    )


def load(campaign, index):
    return np.loadtxt(campaign / f"vec_{index:03d}" / "rg.txt")


def moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def segment_acf(rg, segment_frames):
    """Normalized autocorrelation, averaged over 50 ns segments.

    FFT autocovariance with the unbiased (n - lag) normalization; the
    first half of each segment defines the usable lag range.
    """
    max_lag = segment_frames // 2
    lags = np.arange(max_lag)
    curves = []
    for start in range(0, len(rg) - segment_frames + 1, segment_frames):
        d = rg[start : start + segment_frames]
        d = d - d.mean()
        nfft = 1 << (2 * segment_frames - 1).bit_length()
        spectrum = np.fft.rfft(d, nfft)
        acov = np.fft.irfft(spectrum * spectrum.conjugate(), nfft)[:max_lag]
        acov = acov / (segment_frames - lags)
        curves.append(acov / acov[0])
    return np.mean(curves, axis=0), len(curves)


def tau_one_e(lag_ns, acf):
    """First 1/e crossing, linearly interpolated; None beyond the range."""
    target = 1 / np.e
    below = np.flatnonzero(acf < target)
    if below.size == 0:
        return None
    k = int(below[0])
    if k == 0:
        return float(lag_ns[0])
    x0, x1 = lag_ns[k - 1], lag_ns[k]
    y0, y1 = acf[k - 1], acf[k]
    return float(x0 + (y0 - target) * (x1 - x0) / (y0 - y1))


def block_stats(rg, block_frames=100):
    """Mean and SEM over consecutive 1 ns blocks (friction-run convention)."""
    n_blocks = len(rg) // block_frames
    means = rg[: n_blocks * block_frames].reshape(n_blocks, block_frames).mean(axis=1)
    sem = means.std(ddof=1) / np.sqrt(n_blocks)
    return float(means.mean()), float(sem), n_blocks


def plot_traces(indices, series, out_file):
    figure, axes = plt.subplots(
        len(indices), 1, figsize=(9, 2.6 * len(indices)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, index in zip(axes, indices):
        t_old, rg_old, t_new, rg_new = series[index]
        axis.plot(
            t_old[::10], rg_old[::10], color=COLOR_OLD, alpha=0.25,
            linewidth=0.5, label="friction 1.16 ps$^{-1}$, raw (100 ps)",
        )
        axis.plot(
            t_old[499:][::100], moving_average(rg_old, 500)[::100],
            color=COLOR_OLD, linewidth=1.6,
            label="friction 1.16 ps$^{-1}$, 5 ns mean",
        )
        axis.plot(
            t_new[::10], rg_new[::10], color=COLOR_NEW, alpha=0.3,
            linewidth=0.5, label="friction 0.01 ps$^{-1}$, raw (100 ps)",
        )
        axis.plot(
            t_new[499:][::100], moving_average(rg_new, 500)[::100],
            color=COLOR_NEW, linewidth=1.6,
            label="friction 0.01 ps$^{-1}$, 5 ns mean",
        )
        axis.set_ylabel("Rg (nm)")
        axis.set_title(f"vec_{index:03d}", fontsize=10)
    axes[0].legend(fontsize=7, ncol=2, loc="lower right")
    axes[-1].set_xlabel("time (ns)")
    figure.tight_layout()
    figure.savefig(out_file, dpi=160)
    plt.close(figure)


def plot_acf(indices, acfs, out_file, taus):
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for index in indices:
        lag, corr = acfs[index]["new"]
        axis.plot(
            lag, corr, color=COLOR_NEW, alpha=0.3, linewidth=0.8,
        )
        lag, old = acfs[index]["old"]
        axis.plot(lag, old, color=COLOR_OLD, alpha=0.3, linewidth=0.8)
    pooled_new = np.mean([acfs[i]["new"][1] for i in indices], axis=0)
    pooled_old = np.mean([acfs[i]["old"][1] for i in indices], axis=0)
    lag = acfs[indices[0]]["new"][0]
    label_new = f"pooled 0.01 ps$^{{-1}}$: {taus['new']}"
    label_old = f"pooled 1.16 ps$^{{-1}}$: {taus['old']}"
    axis.plot(lag, pooled_new, color=COLOR_NEW, linewidth=2, label=label_new)
    axis.plot(
        acfs[indices[0]]["old"][0], pooled_old, color=COLOR_OLD,
        linewidth=2, linestyle="--", label=label_old,
    )
    axis.axhline(1 / np.e, color="gray", linewidth=0.8, linestyle=":")
    axis.text(
        0.98, 1 / np.e, "1/e", transform=axis.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=8, color="gray",
    )
    axis.set_xscale("log")
    axis.set_xlabel("lag time (ns)")
    axis.set_ylabel("normalized Rg autocorrelation")
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(out_file, dpi=160)
    plt.close(figure)


def main():
    both = sorted(set(available(PARENT)) & set(available(CORRECTED)))
    if len(sys.argv) > 1:
        indices = [int(arg) for arg in sys.argv[1:]]
        missing = [i for i in indices if i not in both]
        if missing:
            raise SystemExit(f"vectors {missing} incomplete in one campaign")
    else:
        indices = both
    if not indices:
        raise SystemExit("no vectors complete in both campaigns")

    segment_frames = int(SEGMENT_NS / FRAME_NS)
    series, acfs = {}, {}
    for index in indices:
        rg_old = load(PARENT, index)
        rg_new = load(CORRECTED, index)
        series[index] = (
            np.arange(len(rg_old)) * FRAME_NS,
            rg_old,
            np.arange(len(rg_new)) * FRAME_NS,
            rg_new,
        )
        corr_curve, n_new = segment_acf(rg_new, segment_frames)
        old_curve, n_old = segment_acf(rg_old, segment_frames)
        lag_ns = np.arange(len(corr_curve)) * FRAME_NS
        acfs[index] = {
            "new": (lag_ns, corr_curve),
            "old": (lag_ns, old_curve),
        }

    taus = {
        "new": tau_one_e(*acfs[indices[0]]["new"]),
        "old": tau_one_e(*acfs[indices[0]]["old"]),
    }

    comparison = []
    record = rp.protein_record()
    exp_rg, exp_err, cycle11_rg = record[1], record[2], record[3]
    for index in indices:
        mean_new, sem_new, n_new = block_stats(series[index][3])
        mean_old, sem_old, n_old = block_stats(series[index][1])
        comparison.append(
            {
                "index": index,
                "rg_mean_nm": mean_new,
                "rg_sem_nm": sem_new,
                "n_blocks": n_new,
                "parent_1_16_mean_nm": mean_old,
                "parent_1_16_sem_nm": sem_old,
                "difference_nm": mean_new - mean_old,
                "exp_minus_new_nm": exp_rg - mean_new,
                "exp_minus_old_nm": exp_rg - mean_old,
            }
        )
        print(
            f"vec_{index:03d}: 0.01 -> {mean_new:.4f} +- {sem_new:.4f} nm "
            f"({n_new} blocks) | 1.16 -> {mean_old:.4f} +- {sem_old:.4f} nm "
            f"| difference {mean_new - mean_old:+.4f} nm",
            flush=True,
        )

    CORRECTED.mkdir(parents=True, exist_ok=True)
    plot_traces(indices, series, CORRECTED / "rg_traces.png")
    pooled_taus = {
        "0.01": taus["new"],
        "1.16": taus["old"],
    }
    print(
        f"tau 1/e: 0.01 -> {taus['new']} ns | 1.16 -> {taus['old']} ns "
        f"(friction ratio {FRICTION_RATIO:.0f})",
        flush=True,
    )
    plot_acf(
        indices, acfs, CORRECTED / "rg_acf.png",
        {"new": f"{taus['new']:.1f} ns" if taus["new"] else ">25 ns",
         "old": f"{taus['old']:.1f} ns" if taus["old"] else ">25 ns"},
    )
    (CORRECTED / "acf_summary.json").write_text(
        json.dumps(
            {
                "definitions": {
                    "frame_interval_ns": (
                        "spacing between trajectory frames (10 ps): converts "
                        "ACF lags from frames to physical time"
                    ),
                    "segment_ns": (
                        "each pooled ACF averages autocovariance over "
                        "consecutive 50 ns segments, mean-subtracted per "
                        "segment to remove drift"
                    ),
                    "max_lag_ns": (
                        "lags kept only up to half the segment length, where "
                        "the estimator is still unbiased"
                    ),
                    "segments_per_curve": (
                        "number of 50 ns segments averaged per curve; only 2 "
                        "for the 100 ns runs, so expect a noisier tail there"
                    ),
                    "tau_one_e_ns": (
                        "lag where the normalized ACF first falls to 1/e "
                        "(linear interpolation): the Rg relaxation time"
                    ),
                    "measured_speedup_one_e": (
                        "tau(1.16) / tau(0.01); below the friction ratio "
                        "116 because fast local modes dominate the early ACF"
                    ),
                    "friction_ratio": (
                        "1.16 / 0.01: nominal Langevin dynamics speedup from "
                        "the friction correction (HOOMD MW/100 misread)"
                    ),
                    "note": (
                        "a null tau_one_e_ns means the 1/e crossing lies "
                        "beyond max_lag_ns, not an error"
                    ),
                },
                "frame_interval_ns": FRAME_NS,
                "segment_ns": SEGMENT_NS,
                "max_lag_ns": SEGMENT_NS / 2,
                "segments_per_curve": {
                    "0.01": int(len(series[indices[0]][3]) // segment_frames),
                    "1.16": int(len(series[indices[0]][1]) // segment_frames),
                },
                "tau_one_e_ns": pooled_taus,
                "measured_speedup_one_e": (
                    taus["old"] / taus["new"]
                    if taus["old"] and taus["new"] else None
                ),
                "friction_ratio": FRICTION_RATIO,
                "note": "null tau: the 1/e crossing lies beyond max_lag_ns",
            },
            indent=2,
        )
        + "\n"
    )
    (CORRECTED / "comparison.json").write_text(
        json.dumps(
            {
                "friction_ps": rp.FRICTION_COEFF,
                "parent_friction_ps": 1.16,
                "equil_ns": corrected.EQUIL_STEPS * rp.DT * 0.5 / 1000,
                "production_ns": corrected.PROD_STEPS * rp.DT / 1000,
                "block_ns": 1.0,
                "experimental_rg_nm": exp_rg,
                "experimental_rg_err_nm": exp_err,
                "calvados2_cycle11_rg_nm": cycle11_rg,
                "samples": comparison,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"outputs in {CORRECTED}", flush=True)


if __name__ == "__main__":
    main()
