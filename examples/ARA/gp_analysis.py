from argparse import ArgumentParser
import csv
from math import isfinite
from pathlib import Path

import bff
import numpy as np
import torch
import yaml
from bff.bayes.priors import Priors
from bff.domain.specs import Specs

ROOT = Path(__file__).resolve().parent
SPECS_PATH = ROOT / "03-sample/specs.yaml"
MODEL_DIR = ROOT / "05-fit/models"
OUTPUT_DIR = ROOT / "06-learn/gp-analysis"
DIST_TYPE = "normal"
RDF_TOLERANCE = 0.01
SEED = 42


# AGENT COMMIT: summarize posterior draws and check the charge constraints.
def summarize_posterior(results, specs, output_dir):
    results.prepare_samples(discard=0, thin=1, strip_outliers=False)
    draws = results.prepared_samples
    labels = results.labels

    charge_names = specs.parameter_names()
    charge_indices = {name: i for i, name in enumerate(charge_names)}
    charges = draws[:, [labels.index(name) for name in charge_names]]
    lower, upper = specs.bounds.array.T
    near_bound_width = 0.05 * (upper - lower)

    within_bounds = ((charges >= lower) & (charges <= upper)).all(axis=1)
    residual = charges @ specs.constraint_matrix.T - specs.constraint_targets

    output_path = output_dir / "posterior_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "parameter", "q025", "median", "q975",
            "fraction_near_lower", "fraction_near_upper",
        ])
        for label, values in zip(labels, draws.T):
            quantiles = np.quantile(values, [0.025, 0.5, 0.975])
            if label in charge_indices:
                j = charge_indices[label]
                near_lower = np.mean(values <= lower[j] + near_bound_width[j])
                near_upper = np.mean(values >= upper[j] - near_bound_width[j])
            else:
                near_lower = near_upper = ""

            # BFF has already converted log_sigma draws to sigma.
            writer.writerow([
                label.removeprefix("log_"), *quantiles, near_lower, near_upper,
            ])

    converged = results.metadata.get("mcmc", {}).get("converged")
    print(f"Converged: {converged} | saved draws: {len(draws):,}")
    print(f"Within charge bounds: {within_bounds.mean():.2%}")
    print(f"Maximum charge-constraint residual: {np.abs(residual).max():.3g}")
    print(f"Posterior summary: {output_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--rdf-tolerance", type=float, default=RDF_TOLERANCE)
    args = parser.parse_args()

    if args.learn and (
        not isfinite(args.rdf_tolerance) or args.rdf_tolerance <= 0
    ):
        parser.error("--rdf-tolerance must be positive and finite")

    specs = Specs.load(SPECS_PATH)
    states = {}
    for name in ("hb", "rdf"):
        state = torch.load(
            MODEL_DIR / f"{name}.lgp",
            map_location="cpu",
            weights_only=True,
        )
        states[name] = state
        print(f"\n{name}: {len(state['lgps'])} GP member(s)")
        print("Reference size:", len(state["reference_values"]))
        print("Stored validation error:", state["error"])
        for member in state["lgps"]:
            print("Training shapes:", member["X_train"].shape, member["y_train"].shape)
            print("Length scales:", member["lengths"].tolist())
            print("Kernel width and sigma:", member["width"].item(), member["sigma"].item())

    nuisance_names = [
        f"log_sigma_{name}"
        for name, state in states.items()
        if state["nuisance"] is None
    ]
    priors = Priors.from_bounds(
        specs.explicit_bounds.array,
        dist_type=DIST_TYPE,
        names=specs.parameter_names(explicit_only=True),
        n_nuisance=len(nuisance_names),
        nuisance_names=nuisance_names,
    )
    print("\nPriors BFF will construct:")
    for prior in priors.items:
        print(prior.to_dict())

    if not args.learn:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    models = {}
    for name, state in states.items():
        # The original archives were saved on CUDA.
        for member in state["lgps"]:
            member["device"] = "cpu"
        cpu_path = OUTPUT_DIR / f"{name}-cpu.lgp"
        torch.save(state, cpu_path)
        models[name] = {"model_path": str(cpu_path)}

    models["hb"]["independent_observations"] = True
    models["rdf"]["tolerance"] = args.rdf_tolerance
    config = {
        "specs": str(SPECS_PATH),
        "models": models,
        "mcmc": {
            "device": "cpu",
            "priors_disttype": DIST_TYPE,
            "total_steps": 1500,
            "warmup": 500,
            "include_implicit_charge": True,
        },
        "output": {"directory": str(OUTPUT_DIR), "overwrite": False},
    }
    config_path = OUTPUT_DIR / "learn.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    # AGENT COMMIT: run the pilot and report the posterior on its physical scale.
    torch.manual_seed(SEED)
    results = bff.learn(config_path)
    summarize_posterior(results, specs, OUTPUT_DIR)


if __name__ == "__main__":
    main()
