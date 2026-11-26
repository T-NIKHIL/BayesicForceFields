"""Friction-validation campaign: re-run selected CTD2 Sobol vectors at
the corrected Langevin friction.

The original campaign (campaign_ctd2/) ran with friction 1.16 ps^-1,
which misread HOOMD's mass/time gamma (MW/100, i.e. tau = 100 ps) as
OpenMM's 1/ps friction. The corrected value is 0.01 ps^-1
(FRICTION_COEFF in run_pilot_openmm.py). Equilibrium means are
friction-independent in exact Langevin dynamics, so comparing the two
campaigns checks exactly this: whether the 1.16 runs were equilibrated
enough despite the 116x slowed dynamics.

This script reuses every mechanism of run_ctd2_campaign (row-driven
conditions, per-sample lambda overrides, checkpoint/resume, wrap-safe
Rg) but points OUT at campaign_ctd2_friction_0_01/ and runs only the
requested sample indices (default 0, 1, 2). Outputs per sample:
vec_<index>/rg.txt, lambda.json, final_state.xml, ctd2.pdb. No Sobol
matrix or QoI dataset is written - the validation compares rg.txt means
against campaign_ctd2's manifest.

Because the shorter protocol is justified only by the faster dynamics
(116x), this script uses 10 ns equilibration + 100 ns production
(EQUIL_STEPS/PROD_STEPS below) unless overridden the same way as the
parent campaign - edit the constants here, not in run_pilot_openmm.

Run from the repository root:

    python -m examples.calvados_gmx.ctd2.run_ctd2_friction_campaign           # 0,1,2
    python -m examples.calvados_gmx.ctd2.run_ctd2_friction_campaign 5 17      # chosen indices
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from examples.calvados_gmx import run_pilot_openmm as rp
from examples.calvados_gmx.ctd2 import run_ctd2_campaign as parent

HERE = Path(__file__).resolve().parent
OUT = HERE / "campaign_ctd2_friction_0_01"

# Validation protocol: dynamics are 116x faster at the correct friction,
# so 10 ns equilibration + 100 ns production samples ~11.6x more true
# time than the parent campaign's 50 + 500 ns did at the wrong friction.
EQUIL_STEPS = 2_000_000       # 10 ns at dt 0.005
PROD_STEPS = 10_000_000       # 100 ns at dt 0.01
REPORT_INTERVAL = 1_000       # 10 ps per frame
CHECKPOINT_INTERVAL = 500_000
PROGRESS_INTERVAL = 50_000
DEFAULT_INDICES = (0, 1, 2)


def run_sample(index, lam_by_three):
    """One corrected-friction trajectory; mirrors parent.run_sample with
    this campaign's constants and output directory."""
    out_dir = OUT / f"vec_{index:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rg_file = out_dir / "rg.txt"
    if rg_file.is_file():
        rg = np.loadtxt(rg_file)
        return rg, None

    import os

    scratch = Path(os.environ.get("SCRATCH") or os.environ.get("TMPDIR") or out_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    record = rp.protein_record()
    temp, ionic, ph = rp.protein_conditions(record)
    records, sequence = parent.set_lambda_in_particle_records(lam_by_three)
    kappa, lB, RT = rp.debye(temp, ionic)

    system, box = rp.simulation_box(records)
    rp.add_pair_force(system, records, kappa, lB, RT)
    positions = rp.input_positions(records)
    rp.write_pdb(out_dir / "ctd2.pdb", records, positions, box)
    topology = rp.build_topology(records, sequence)

    integrator = mm.LangevinMiddleIntegrator(
        temp * unit.kelvin, rp.FRICTION_COEFF, rp.DT * 0.5
    )
    integrator.setRandomNumberSeed(parent.SEED + index)
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

    checkpoint = scratch / f"ctd2_friction_vec{index:03d}.chk"
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
            temp * unit.kelvin, parent.SEED + index
        )
        print(
            f"vec_{index:03d}: equilibration {EQUIL_STEPS} steps at "
            f"dt {rp.DT * 0.5} ps on {platform_name} "
            f"(friction {rp.FRICTION_COEFF} ps^-1)",
            flush=True,
        )
        simulation.step(EQUIL_STEPS)

    integrator.setStepSize(rp.DT)
    dcd_path = scratch / f"ctd2_friction_vec{index:03d}.dcd"
    simulation.reporters.append(
        app.DCDReporter(str(dcd_path), REPORT_INTERVAL, append=resumed)
    )
    simulation.reporters.append(
        app.CheckpointReporter(str(checkpoint), CHECKPOINT_INTERVAL)
    )
    remaining = max(EQUIL_STEPS + PROD_STEPS - simulation.currentStep, 0)
    if resumed and (out_dir / "final_state.xml").is_file():
        print("final_state.xml present: trajectory already complete", flush=True)
        remaining = 0
    print(f"vec_{index:03d}: production {remaining}/{PROD_STEPS} steps", flush=True)
    step = max(simulation.currentStep - EQUIL_STEPS, 0)
    for block in range(0, remaining, PROGRESS_INTERVAL):
        chunk = min(PROGRESS_INTERVAL, remaining - block)
        simulation.step(chunk)
        step += chunk
        energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        print(
            f"  vec_{index:03d} {step}/{PROD_STEPS} steps, "
            f"potential {energy.value_in_unit(unit.kilojoule_per_mole):8.2f}"
            " kJ/mol",
            flush=True,
        )
    simulation.saveState(str(out_dir / "final_state.xml"))

    masses = np.array([r["mass"] for r in records])
    rg = rp.radius_of_gyration(out_dir / "ctd2.pdb", dcd_path, masses, box)
    np.savetxt(rg_file, rg, fmt="%.4f")
    (out_dir / "lambda.json").write_text(json.dumps(lam_by_three, indent=2) + "\n")
    return rg, time.time()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = parent.parameter_names()
    points = parent.sobol_points(parent.PROTEIN_ID)

    indices = list(DEFAULT_INDICES)
    if len(sys.argv) > 1:
        indices = [int(arg) for arg in sys.argv[1:]]
    for index in indices:
        if not 0 <= index < parent.N_SAMPLES:
            raise SystemExit(f"sample index {index} outside 0..{parent.N_SAMPLES - 1}")

    comparison = []
    for index in indices:
        lam_by_three = {
            name.split()[1]: float(value)
            for name, value in zip(names, points[index])
        }
        rg, cached = run_sample(index, lam_by_three)
        frames_per_block = int(1000 / (REPORT_INTERVAL * rp.DT))
        n_blocks = len(rg) // frames_per_block
        means = rg[: n_blocks * frames_per_block].reshape(n_blocks, frames_per_block).mean(axis=1)
        mean, sem = means.mean(), means.std(ddof=1) / np.sqrt(n_blocks)
        reference_file = parent.OUT / f"vec_{index:03d}" / "rg.txt"
        reference_mean = float(np.loadtxt(reference_file).mean()) if reference_file.is_file() else float("nan")
        comparison.append(
            {
                "index": index,
                "rg_mean_nm": float(mean),
                "rg_sem_nm": float(sem),
                "n_blocks": int(n_blocks),
                "parent_1_16_mean_nm": reference_mean,
                "difference_nm": float(mean - reference_mean),
            }
        )
        print(
            f"vec_{index:03d}: friction 0.01 -> {mean:.4f} +- {sem:.4f} nm "
            f"({n_blocks} blocks) | parent 1.16 -> {reference_mean:.4f} nm "
            f"| difference {mean - reference_mean:+.4f} nm",
            flush=True,
        )

    import json as _json

    (OUT / "comparison.json").write_text(
        _json.dumps(
            {
                "friction_ps": rp.FRICTION_COEFF,
                "parent_friction_ps": 1.16,
                "equil_ns": EQUIL_STEPS * rp.DT * 0.5 / 1000,
                "production_ns": PROD_STEPS * rp.DT / 1000,
                "samples": comparison,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"comparison written to {OUT / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
