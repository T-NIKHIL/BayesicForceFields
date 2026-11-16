"""CTD2 Sobol campaign: 512 lambda vectors x 500 ns, one Rg each.

The Sobol-campaign pattern of run_sobol_campaign.py generalized to the
protocol and protein of run_pilot_openmm.py:

- 512-sample Sobol sequence over the residues present in the CTD2
  sequence (12 of the 20 amino acids; the remaining 8 columns stay
  pinned at their optimized M1_24 values), reported in the full
  20-column GP order (scipy.stats.qmc)
- per sample: CTD2 (83 beads) at the row-driven conditions of the pilot
  (293 K, 0.12 M, pH 7.5), 50 ns equilibration at dt 0.005 discarded,
  500 ns production at dt 0.01, frames every 10 ps
- per sample: fresh OpenMM context, CUDA -> OpenCL -> CPU platform
  preference, checkpoint/resume via checkpoint.chk, wrap-safe mass-
  weighted Rg per frame; the trajectory stays on the node (SCRATCH/tmp
  when running under Slurm), only rg.txt and lambda.json land next to
  this script
- skip-and-resume: a sample whose rg.txt exists is not rerun, so the
  campaign (and every array task) can be restarted freely

Writes, under campaign_ctd2/:

- sobol_points.txt      the (512, 20) lambda matrix in GP column order
- vec_<index>/rg.txt    mass-weighted Rg per frame (nm), 50000 frames
- vec_<index>/lambda.json  the vector as {'THREE': lambda}
- manifest.csv          per-sample mean/std Rg and wall time
- rg_ctd2.pt            BFF QoIDataset: X = lambdas, y = mean Rg per
                        vector, y_ref = experimental Rg (2.61 nm)

One sample per process: pass the sample index as the only argument and
this module runs exactly that vector (the Slurm array-task entry point).
Without arguments it runs every pending sample in sequence, for local
use.

Run from the repository root:

    python -m examples.calvados_gmx.run_ctd2_campaign          # all
    python -m examples.calvados_gmx.run_ctd2_campaign 42       # one sample

On Aurum, one array task per sample:

    sbatch --array=1-512%50 campaign_ctd2/job.sh
"""

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from examples.calvados.fetch_data import RAW, verified_bytes
from examples.calvados_gmx import run_pilot_openmm as rp

PROTEIN_ID = 'CTD2'

HERE = rp.HERE
OUT = HERE / "campaign_ctd2"

# Sobol parameters
SEED = 2026
LOW = 0.001
HIGH = 0.999
N_SAMPLES = 512          # Sobol sequences are balanced at powers of 2

# Integrator and simulation settings
FRICTION_COEFF = 0.01         # ps^-1 (for openMM)
# then 500 ns production at dt 0.01, frames every 10 ps.
# Equilibration 50 ns = 50_000 ps / 0.005 ps = 10_000_000 steps.
# Production 500 ns = 500_000 ps / 0.01 ps = 50_000_000 steps.
EQUIL_STEPS = 10_000_000      # 50 ns at dt 0.005
PROD_STEPS = 50_000_000       # 500 ns at dt 0.01
REPORT_INTERVAL = 1_000       # steps between trajectory frames = 10 ps at dt 0.01
CHECKPOINT_INTERVAL = 500_000 # steps between context checkpoints (~40 s on OpenCL)
PROGRESS_INTERVAL = 50_000    # steps between progress prints (~5 s on OpenCL; user-tunable)

def parameter_names():
    """GP column order: lambda <THREE> sorted by one-letter code."""
    residues = pd.read_csv(io.BytesIO(verified_bytes(RAW / "residues.csv")))
    one_to_three = dict(zip(residues["one"], residues["three"]))
    return tuple(f"lambda {three}" for three in sorted(one_to_three.values()))


def sobol_points(protein_id=None):
    """The (N_SAMPLES, 20) Sobol matrix, regenerated deterministically.

    All 20 columns are sampled (comparable across proteins), but the
    Sobol cube is only as wide as the sequence's unique residue set:
    CTD2 contains 12 of the 20 amino acids, so the other 8 columns stay
    pinned at their optimized M1_24 values and every sample varies the
    12 residue types that actually occur in the sequence.
    """
    residues = pd.read_csv(io.BytesIO(verified_bytes(RAW / "residues.csv")))
    # Sorting based on one letter codes
    residues = residues.set_index("one")
    sequence, _, _ = rp.type_sequence(protein_id)
    # All the residues (one-letter code) present in the sequence.
    present = sorted(set(sequence))
    # Residues not present in sequence
    absent = [one for one in sorted(residues.index) if one not in present]
    print(
        f"sampling {len(present)} residue types present in {rp.PROTEIN_ID}"
        f"; {len(absent)} absent residues pinned at optimized lambdas: "
        f"{', '.join(absent)}",
        flush=True,
    )
    sampler = qmc.Sobol(d=len(present), scramble=True, seed=SEED)
    active = LOW + (HIGH - LOW) * sampler.random(N_SAMPLES)
    # The GP column order is one-letter sorted; the CSV row order is not
    # (R, D, N, ...), so reindex explicitly before tiling.
    # M1_24 optimized lambdas; Same as cycle 11 lambda values.
    optimized_sorted = residues["lambdas"].astype(float).reindex(sorted(residues.index))
    # Copy optimized values across N_SAMPLES rows
    points = np.tile(optimized_sorted.to_numpy(), (N_SAMPLES, 1))
    # Get the column index \in {0,1 ..19} of the AA that is present.
    columns = [sorted(residues.index).index(one)
               for one in present]
    for position, column in enumerate(columns):
        points[:, column] = active[:, position]
    return points


def set_lambda_in_particle_records(lam_by_three):
    """CTD2 bead records with per-sample stickiness overrides applied.

    The lambda of every bead - including the X/Z terminal pseudo-types,
    whose base residue comes from the sequence - is replaced by the
    sample's value; sigma, charge and mass stay at the optimized values.
    """
    records, sequence = rp.particle_records()
    residues = pd.read_csv(io.BytesIO(verified_bytes(RAW / "residues.csv")))
    one_to_three = dict(zip(residues["one"], residues["three"]))
    for record in records:
        base = record["base"]
        record["lambda"] = lam_by_three[one_to_three[base]]
    return records, sequence


def run_sample(index, lam_by_three):
    """One 500 ns trajectory; caches to campaign_ctd2/vec_<index>/.

    Heavy files (checkpoint, DCD) live in the node-local scratch when
    running under Slurm so the shared filesystem sees only rg.txt and
    lambda.json; without SLURM_JOB_ID everything lands in vec_<index>.
    """
    out_dir = OUT / f"vec_{index:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rg_file = out_dir / "rg.txt"
    if rg_file.is_file():
        rg = np.loadtxt(rg_file)
        return rg, None  # None seconds marks a cache hit

    scratch = Path(os.environ.get("SCRATCH") or os.environ.get("TMPDIR") or out_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    # This extracts the exp_rg, exp_rg_err, temp, ionic and pH from c2_rg.csv file.
    protein_record = rp.protein_record('CTD2')
    temp, ionic, ph = rp.protein_conditions(protein_record)
    particle_records, sequence = set_lambda_in_particle_records(lam_by_three)
    # Compute the parameters needed for the Yukawa potential
    kappa, lB, RT = rp.debye(temp, ionic)

    system, box = rp.simulation_box(particle_records)
    # The potential energy function, bonded parameters, cutoffs and LJ parameters are defined in run_pilot_openmm.py
    rp.add_pair_force(system, particle_records, kappa, lB, RT)
    positions = rp.input_positions(particle_records)
    rp.write_pdb(out_dir / "ctd2.pdb", particle_records, positions, box)
    topology = rp.build_topology(particle_records, sequence)

    integrator = mm.LangevinMiddleIntegrator(
        temp * unit.kelvin, FRICTION_COEFF, rp.DT * 0.5
    )
    integrator.setRandomNumberSeed(SEED + index)
    platforms = [
        mm.Platform.getPlatform(i).getName()
        for i in range(mm.Platform.getNumPlatforms())
    ]
    platform_name = next(
        (name for name in ("CUDA", "OpenCL", "CPU") if name in platforms),
        "CPU",
    )
    simulation = app.Simulation(
        topology, system, integrator,
        mm.Platform.getPlatformByName(platform_name),
    )

    checkpoint = scratch / f"ctd2_vec{index:03d}.chk"
    resumed = checkpoint.is_file()
    if resumed:
        simulation.loadCheckpoint(str(checkpoint))
        print(
            f"vec_{index:03d}: resumed from checkpoint at step "
            f"{simulation.currentStep} ({platform_name})",
            flush=True,
        )
    else:
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(
            temp * unit.kelvin, SEED + index
        )
        print(
            f"vec_{index:03d}: equilibration {rp.EQUIL_STEPS} steps at "
            f"dt {rp.DT * 0.5} ps on {platform_name}",
            flush=True,
        )
        simulation.step(rp.EQUIL_STEPS)

    integrator.setStepSize(rp.DT)
    # .dcd stored on scratch
    dcd_path = scratch / f"ctd2_vec{index:03d}.dcd"
    simulation.reporters.append(
        app.DCDReporter(str(dcd_path), REPORT_INTERVAL, append=resumed)
    )
    simulation.reporters.append(
        app.CheckpointReporter(str(checkpoint), CHECKPOINT_INTERVAL)
    )
    remaining = max(rp.EQUIL_STEPS + rp.PROD_STEPS - simulation.currentStep, 0)
    complete = resumed and not (out_dir / "rg.txt").is_file() and \
        (out_dir / "final_state.xml").is_file()
    if complete:
        remaining = 0
    print(
        f"vec_{index:03d}: production {remaining}/{rp.PROD_STEPS} steps "
        f"at dt {rp.DT} ps",
        flush=True,
    )
    step = max(simulation.currentStep - rp.EQUIL_STEPS, 0)
    for block in range(0, remaining, PROGRESS_INTERVAL):
        chunk = min(PROGRESS_INTERVAL, remaining - block)
        simulation.step(chunk)
        step += chunk
        energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        # This gets logged in the slurm.out
        print(
            f"  vec_{index:03d} {step}/{rp.PROD_STEPS} steps, "
            f"potential {energy.value_in_unit(unit.kilojoule_per_mole):8.2f}"
            " kJ/mol",
            flush=True,
        )
    # The positions and velocities at the final step.
    simulation.saveState(str(out_dir / "final_state.xml"))

    masses = np.array([r["mass"] for r in particle_records])
    rg = rp.radius_of_gyration(out_dir / "ctd2.pdb", dcd_path, masses, box)
    np.savetxt(rg_file, rg, fmt="%.4f")
    (out_dir / "lambda.json").write_text(
        json.dumps(lam_by_three, indent=2) + "\n"
    )
    return rg, time.time()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Lambda for each AA sorted by One Letter AA Codes
    names = parameter_names()
    points = sobol_points(PROTEIN_ID)
    np.savetxt(OUT / "sobol_points.txt", points, fmt="%.6f")

    # Array-task entry point: exactly one sample index when given.
    indices = range(N_SAMPLES)
    if len(sys.argv) > 1:
        index = int(sys.argv[1])
        if not 0 <= index < N_SAMPLES:
            raise SystemExit(
                f"sample index {index} outside 0..{N_SAMPLES - 1}"
            )
        indices = [index]

    for index in indices:
        lam_by_three = {
            name.split()[1]: float(value)
            for name, value in zip(names, points[index])
        }
        rg, cached_seconds = run_sample(index, lam_by_three)
        if cached_seconds is None:
            print(
                f"vec_{index:03d}: Rg {rg.mean():.4f} nm (cached)",
                flush=True,
            )
            continue
        seconds = time.time() - cached_seconds
        print(
            f"vec_{index:03d}: Rg {rg.mean():.4f} +- {rg.std():.4f} nm "
            f"({seconds:.0f} s)",
            flush=True,
        )

    # The QoI dataset is assembled by whoever runs last; every array task
    # refreshes the manifest cheaply and only index 0 rebuilds the dataset.
    manifest_rows = []
    for index in range(N_SAMPLES):
        rg_file = OUT / f"vec_{index:03d}" / "rg.txt"
        if rg_file.is_file():
            rg = np.loadtxt(rg_file)
            manifest_rows.append(
                {
                    "sample_id": f"vec_{index:03d}",
                    "rg_mean_nm": float(rg.mean()),
                    "rg_std_nm": float(rg.std()),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(OUT / "manifest.csv", index=False)
    if len(sys.argv) <= 1 or int(sys.argv[1]) == 0:
        build_dataset(points, names)


def build_dataset(points, names):
    """Assemble the BFF QoIDataset from whatever samples are complete."""
    from bff.qoi.dataset import QoIDataset

    rows = []
    for index in range(N_SAMPLES):
        rg_file = OUT / f"vec_{index:03d}" / "rg.txt"
        if rg_file.is_file():
            rows.append(np.loadtxt(rg_file).mean())
    if len(rows) < N_SAMPLES:
        print(
            f"dataset deferred: {len(rows)}/{N_SAMPLES} samples complete",
            flush=True,
        )
        return
    means = np.asarray(rows)
    record = rp.protein_record()
    # The entire 20D parameter space is stored in QoIdataset
    dataset = QoIDataset(
        name="rg_ctd2",
        X=points,
        y=means[:, None],
        y_ref=np.array([record[1]]),  # tuple: (fasta, exp_rg, exp_err, ...)
        parameter_names=names,
        sample_ids=tuple(f"vec_{i:03d}" for i in range(N_SAMPLES)),
        metadata={
            "protein": rp.PROTEIN_ID,
            "temperature_k": record[-3],
            "ionic_m": record[-2],
            "ph": record[-1],
            "equilibration_ns": rp.EQUIL_STEPS * rp.DT * 0.5 / 1000,
            "production_ns": rp.PROD_STEPS * rp.DT / 1000,
            "sobol_seed": SEED,
            "engine": "openmm",
            "experimental_rg_nm": record[1],
            "experimental_rg_err_nm": record[2],
            "qoi": "mass-weighted Rg, frame mean, wrap-safe",
        },
    )
    dataset.write(str(OUT / "rg_ctd2.pt"))
    print(f"QoI dataset: {OUT / 'rg_ctd2.pt'}", flush=True)


if __name__ == "__main__":
    main()
