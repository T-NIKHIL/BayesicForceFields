"""CALVADOS2 pilot in OpenMM: the protein PROTEIN_ID (default "CTD2") at
the M1_24 optimized lambdas.

Reproduces upstream simulate.py (HOOMD + azplugins ashbaugh) exactly, in
one readable place:

- one bead per residue, terminal beads X (+1 e, +2 Da) and Z (-1 e, +16 Da)
- Ashbaugh-Hatch pair energy
    U = 4*eps*[(sig_ij/r)^12-(sig_ij/r)^6] + eps*(1-lam_ij)
        for r <= 2^(1/6)*sig_ij
    U = 4*eps*lam_ij*[(sig_ij/r)^12-(sig_ij/r)^6]
        for r >  2^(1/6)*sig_ij
  energy-shifted to zero at 2.4 nm (upstream mode='shift')
- sig_ij and lam_ij are arithmetic means of the per-residue values
- Yukawa (Debye-Huckel) electrostatics q_i q_j lB RT exp(-kappa r)/r,
  energy-shifted to zero at 4.0 nm
- harmonic bonds k = 8033 kJ/mol/nm^2, r0 = 0.38 nm; only bonded pairs
  are excluded from nonbonded; no angles or dihedrals
- Langevin dynamics, dt = 0.01 ps; upstream gamma = MW/100 ps^-1 varies
  per residue (0.57-1.63 here); we use the global 1.16 ps^-1 (ADAPT)

Residue numbers come from the checksummed upstream residues.csv via
build_pilot.py, which also keeps the GROMACS-format artifacts
(pilot/PARAMETERS.md, pilot/tables/) on disk for eyeballing - modern
GROMACS cannot run per-pair user tables (no group cutoff scheme), so
OpenMM evaluates the same pair functions directly as algebra.

Outputs, in examples/calvados_gmx/pilot_openmm/:

- {PROTEIN_ID.lower()}.pdb  input coordinates (inspectable)
- trajectory.dcd     frames every REPORT_INTERVAL steps after 10k equilibration
- rg.txt             mass-weighted Rg per frame (nm)
- energy_check.txt   the two-bead H-Y single-point verification against
                     the same pair math as the GROMACS tables

Run from the repository root:

    python -m examples.calvados_gmx.run_pilot_openmm
"""

from pathlib import Path
import io
import math

import numpy as np
import pandas as pd

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from examples.calvados_gmx.build_pilot import pair_table
from examples.calvados.fetch_data import RAW, verified_bytes

PROTEIN_ID = 'CTD2'

HERE = Path(__file__).resolve().parent
OUT = HERE / "pilot_openmm"
LAMBDA_COLUMN = "lambdas"     # 'lambdas' = optimized; 'lambdas_0' = M1 start
SEED = 2026

# Simulation parameters
EPS_HI = 4.184 * 0.2          # kJ/mol, lj_eps
R_AH = 2.4                    # nm, Ashbaugh-Hatch cutoff (mode='shift')
R_YK = 4.0                    # nm, Yukawa cutoff (mode='shift')
BOND_K = 8033.0               # kJ/mol/nm^2
BOND_R0 = 0.38                # nm
DT = 0.01                     # ps
TABLE_STEP = 0.002            # nm, user-table grid spacing
TABLE_MAX = 5.0               # nm, table range: must cover rvdw + 1
# Conditions (temp, ionic, pH) are per-protein: read from the protein's
# c2_rg.csv row via protein_conditions(); Hst5's row matches the upstream
# analyse.py values (293 K, 0.15 M, pH 7.5).
EQUIL_STEPS = 10_000          # upstream: 10000 steps at dt 0.005
PROD_STEPS = 110_000          # 1.1 ns pilot at dt 0.01
REPORT_INTERVAL = 100         # steps between trajectory frames

# The whole pair physics, evaluated by OpenMM per pair. OpenMM takes one
# flat expression, so the algebra is assembled from named pieces: sigma_ij
# and lambda_ij are arithmetic means; the Ashbaugh-Hatch and Yukawa parts
# are each energy-shifted at their own cutoffs (upstream mode='shift').
_SIGMA = "0.5*(sigma1 + sigma2)"
_LAMBDA = "0.5*(lambda1 + lambda2)"
_QQ = "q1*q2*lb*rt"
_SR6 = f"({_SIGMA}/r)^6"
_LJ = f"(4*epsilon*({_SR6}*{_SR6} - {_SR6}))"
_WCA = f"({_LJ} + epsilon*(1 - {_LAMBDA}))"
_ATT = f"({_LAMBDA}*{_LJ})"
# step is a heaviside function. If r<=rmin then _WCA else _ATT
_AH = f"select(step(1.1224620483093721*{_SIGMA} - r), {_WCA}, {_ATT})"
_AH_SHIFT = f"(4*epsilon*{_LAMBDA}*(({_SIGMA}/rc_ah)^12 - ({_SIGMA}/rc_ah)^6))"
_YK = f"({_QQ}*exp(-kappa*r)/r)"
_YK_SHIFT = f"({_QQ}*exp(-kappa*rc_yk)/rc_yk)"
PAIR_ENERGY = (
    f"step(rc_ah - r)*({_AH} - {_AH_SHIFT})"
    f" + step(rc_yk - r)*({_YK} - {_YK_SHIFT})"
)

def protein_record(protein_id=None):
    """Full c2_rg.csv row (fasta, conditions, experimental Rg) for the
    requested protein; the single source for sequence AND conditions.

    Hst5's entry is identical to the sequence hardcoded upstream (verified
    byte-for-byte), so switching PROTEIN_ID changes the system without
    touching the physics. Raises a clear error naming the table and ID
    when absent.
    """
    if protein_id is None:
        protein_id = PROTEIN_ID  # module global, read at call time
    table = pd.read_csv(HERE.parent / "calvados" / "c2_rg.csv")
    hit = table[table.iloc[:, 0] == protein_id]
    if hit.empty:
        raise ValueError(
            f"calvados_gmx pilot: protein {protein_id!r} not found in "
            "examples/calvados/c2_rg.csv"
        )
    return hit.iloc[0]


def protein_conditions(record):
    """(temp, ionic, ph) from the protein's c2_rg.csv row."""
    return float(record["temp"]), float(record["ionic"]), float(record["pH"])

def residues_table(lambda_column="lambdas", ph=None):
    """The 20 CALVADOS residues; ``lambda_column`` selects the
    parameterization ('lambdas' = optimized, 'lambdas_0' = M1 start).

    The pH-dependent HIS charge is evaluated at ``ph`` (the protein's
    c2_rg.csv value); upstream HPS default when omitted.
    """
    table = pd.read_csv(io.BytesIO(verified_bytes(RAW / "residues.csv")))
    table = table.set_index("one", drop=False)
    table["q"] = table["q"].astype(float)
    if lambda_column not in table.columns:
        raise ValueError(f"CALVADOS residues: no column {lambda_column!r}")
    table["lambdas"] = table[lambda_column].astype(float)
    # pH-dependent HIS charge, exactly as upstream genParams
    if ph is None:
        ph = protein_conditions(protein_record())[2]
    table.loc["H", "q"] = 1.0 / (1.0 + 10 ** (ph - 6))
    return table

def debye(temperature, ionic):
    """(kappa in 1/nm, lB in nm, RT in kJ/mol) exactly as upstream."""
    RT = 8.3145 * temperature * 1e-3
    fepsw = lambda T: (5321 / T + 233.76 - 0.9297 * T + 0.1417e-2 * T * T - 0.8292e-6 * T ** 3)  # noqa: E731
    epsw = fepsw(temperature)
    lB = 1.6021766 ** 2 / (4 * math.pi * 8.854188 * epsw) * 6.022 * 1000 / RT
    kappa = math.sqrt(8 * math.pi * lB * ionic * 6.022 / 10)
    return kappa, lB, RT

def type_sequence(protein_id=None):
    """The pilot protein with upstream terminal handling: X = N-term
    (+1 charge, +2 MW), Z = C-term (-1 charge, +16 MW)."""
    sequence = list(protein_record(protein_id)["fasta"])
    types = list(sequence)
    first, last = types[0], types[-1]
    # Replacing the first and last AA types with N-term and C-term type
    types[0] = "X"
    types[-1] = "Z"
    terminal = {
        "X": dict(base=first, dq=+1.0, dm=+2.0),
        "Z": dict(base=last, dq=-1.0, dm=+16.0),
    }
    return sequence, types, terminal

def extended_residues(residues, terminal):
    """Residue table extended with the terminal pseudo-types X and Z."""
    table = residues.copy()
    for name, spec in terminal.items():
        # Name : Terminal Bead ; spec : Type of Side chain
        row = residues.loc[spec["base"]].copy()
        row["one"] = name
        row["q"] = row["q"] + spec["dq"]
        row["MW"] = row["MW"] + spec["dm"]
        # Adjust the 'q' and 'MW' attributes for the first and last bead in the residues table.
        table.loc[name] = row
    return table

def particle_records(lambdas=LAMBDA_COLUMN, protein_id=PROTEIN_ID):
    """One record per bead: (type, sigma, lambda, charge, mass)."""
    # All the 20 AA with MW, sigmas, charge, hydropathys from HPS,
    residues = residues_table(lambdas)
    sequence, types, terminal = type_sequence(protein_id)
    records_ext = extended_residues(residues, terminal)
    records = []
    for i, one in enumerate(types):
        row = records_ext.loc[one]
        records.append(
            {
                "type": one,
                "base": sequence[i],
                "sigma": float(row.sigmas),
                "lambda": float(row.lambdas),
                "q": float(row.q),
                "mass": float(row.MW),
            }
        )
    return records, sequence


def add_pair_force(system, records, kappa, lB, RT):
    """The Ashbaugh-Hatch + Yukawa pair force with per-particle parameters."""
    force = mm.CustomNonbondedForce(PAIR_ENERGY)
    for name in ("sigma", "lambda", "q"):
        force.addPerParticleParameter(name)
    force.addGlobalParameter("epsilon", EPS_HI)
    force.addGlobalParameter("lb", lB)
    force.addGlobalParameter("rt", RT)
    force.addGlobalParameter("kappa", kappa)
    force.addGlobalParameter("rc_ah", R_AH)
    force.addGlobalParameter("rc_yk", R_YK)
    force.setCutoffDistance(R_YK * unit.nanometer)
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    for record in records:
        force.addParticle(
            [record["sigma"], record["lambda"], record["q"]]
        )
    for i in range(len(records) - 1):
        force.addExclusion(i, i + 1)
    system.addForce(force)
    return force


def pilot_system(records):
    """One bonded bead chain in a periodic box sized to the chain.

    The box must be at least 2x the nonbonded cutoff (2 * R_YK = 8 nm)
    for minimum-image correctness under CutoffPeriodic; the chain term
    (n * BOND_R0 + 1) dominates for every protein of 19+ residues.
    """
    n = len(records)
    box = max(n * BOND_R0 + 1.0, 2 * R_YK + 1.0)
    system = mm.System()
    for record in records:
        system.addParticle(record["mass"])
    bonds = mm.HarmonicBondForce()
    for i in range(n - 1):
        bonds.addBond(i, i + 1, BOND_R0, BOND_K)
    system.addForce(bonds)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box)
    )
    return system, box


def input_positions(records):
    """Beads on a line along z, exactly as upstream make_snapshot."""
    n = len(records)
    positions = [
        mm.Vec3(0, 0, (i - n / 2) * BOND_R0) for i in range(n)
    ]
    return positions


def single_point_check(energy_text):
    """Two H/Y beads at 0.6 nm: OpenMM energy vs the GROMACS-table math."""
    temp, ionic, ph = protein_conditions(protein_record())
    residues = residues_table(LAMBDA_COLUMN, ph=ph)
    kappa, lB, RT = debye(temp, ionic)
    system = mm.System()
    for one in ("H", "Y"):
        system.addParticle(float(residues.loc[one, "MW"]))
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(10, 0, 0), mm.Vec3(0, 10, 0), mm.Vec3(0, 0, 10)
    )
    records = [
        {
            "sigma": float(residues.loc["H", "sigmas"]),
            "lambda": float(residues.loc["H", "lambdas"]),
            "q": float(residues.loc["H", "q"]),
        },
        {
            "sigma": float(residues.loc["Y", "sigmas"]),
            "lambda": float(residues.loc["Y", "lambdas"]),
            "q": float(residues.loc["Y", "q"]),
        },
    ]
    force = mm.CustomNonbondedForce(PAIR_ENERGY)
    for name in ("sigma", "lambda", "q"):
        force.addPerParticleParameter(name)
    force.addGlobalParameter("epsilon", EPS_HI)
    force.addGlobalParameter("lb", lB)
    force.addGlobalParameter("rt", RT)
    force.addGlobalParameter("kappa", kappa)
    force.addGlobalParameter("rc_ah", R_AH)
    force.addGlobalParameter("rc_yk", R_YK)
    force.setCutoffDistance(R_YK * unit.nanometer)
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    for record in records:
        force.addParticle([record[k] for k in ("sigma", "lambda", "q")])
    system.addForce(force)
    integrator = mm.LangevinMiddleIntegrator(temp * unit.kelvin, 1.0, DT)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CPU"))
    context.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.6, 0, 0)])
    openmm_energy = context.getState(getEnergy=True).getPotentialEnergy()
    _, values, _, grid = pair_table(residues, "H", "Y", kappa, lB, RT)
    expected = values[int(round(0.6 / 0.002))]
    text = (
        f"two-bead check (H-Y at 0.6 nm, no bonds)\n"
        f"  openmm : {openmm_energy.value_in_unit(unit.kilojoule_per_mole):.8f}"
        " kJ/mol\n"
        f"  tables : {expected:.8f} kJ/mol (build_pilot.py pair math)\n"
    )
    assert abs(
        openmm_energy.value_in_unit(unit.kilojoule_per_mole) - expected
    ) < 1e-4, "pair energy implementations disagree"
    return text + "  agreement within 1e-4 kJ/mol\n"


def build_topology(records, sequence):
    """An OpenMM topology matching the beads, for trajectory reporting."""
    residues = residues_table(LAMBDA_COLUMN)
    three_of = residues["three"].to_dict()
    topology = app.Topology()
    chain = topology.addChain("A")
    for i, record in enumerate(records):
        # Terminal pseudo-types X/Z label as their base residue; the
        # charge/mass/sigma/lambda corrections live in the records, not
        # in the display name.
        base = record.get("base", record["type"])
        residue = topology.addResidue(
            f"{three_of[base]}{i + 1}", chain
        )
        element = app.element.carbon
        topology.addAtom(record["type"], element, residue)
    return topology


def write_pdb(path, records, positions, box):
    """A minimal inspectable PDB of the input coordinates."""
    residues = residues_table(LAMBDA_COLUMN)
    three_of = residues["three"].to_dict()
    lines = [f"REMARK   CALVADOS2 pilot input: {PROTEIN_ID} beads on a line"]
    for i, (record, position) in enumerate(zip(records, positions), start=1):
        one = record["type"]
        base = record.get("base", one)
        lines.append(
            f"ATOM  {i:5d} {one:>4s} {three_of[base]:>3s} A{i:4d}    "
            f"{position[0]:8.3f}{position[1]:8.3f}{position[2]:8.3f}"
            f"  1.00  0.00          {base:>2s}"
        )
    lines.append(f"CRYST1{box:9.3f}{box:9.3f}{box:9.3f} 90.00 90.00 90.00")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def radius_of_gyration(pdb, dcd, masses):
    """Mass-weighted Rg per frame, as upstream analyse.py calcRg."""
    import MDAnalysis as mda

    universe = mda.Universe(str(pdb), str(dcd))
    values = []
    for _ in universe.trajectory:
        # MDAnalysis positions are in Angstrom; convert to nm.
        positions = universe.atoms.positions / 10.0
        com = np.average(positions, axis=0, weights=masses)
        values.append(
            float(np.sqrt(np.average(((positions - com) ** 2).sum(1), weights=masses)))
        )
    return np.array(values)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record = protein_record()
    temp, ionic, ph = protein_conditions(record)
    records, sequence = particle_records()
    kappa, lB, RT = debye(temp, ionic)

    check_text = single_point_check(None)
    (OUT / "energy_check.txt").write_text(check_text)
    print(check_text, end="")

    system, box = pilot_system(records)
    add_pair_force(system, records, kappa, lB, RT)
    positions = input_positions(records)
    write_pdb(OUT / f"{PROTEIN_ID.lower()}.pdb", records, positions, box)

    integrator = mm.LangevinMiddleIntegrator(
        temp * unit.kelvin, 1.16, DT * 0.5
    )
    integrator.setRandomNumberSeed(SEED)
    topology = build_topology(records, sequence)
    simulation = app.Simulation(
        topology,
        system,
        integrator,
        mm.Platform.getPlatformByName("CPU"),
    )
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(
        temp * unit.kelvin, SEED
    )

    print(f"equilibration: {EQUIL_STEPS} steps at dt {DT * 0.5} ps", flush=True)
    simulation.step(EQUIL_STEPS)
    integrator.setStepSize(DT)
    simulation.reporters.append(
        app.DCDReporter(str(OUT / "trajectory.dcd"), REPORT_INTERVAL)
    )
    print(f"production: {PROD_STEPS} steps at dt {DT} ps", flush=True)
    for block in range(0, PROD_STEPS, 5_000):
        chunk = min(5_000, PROD_STEPS - block)
        simulation.step(chunk)
        energy = simulation.context.getState(
            getEnergy=True
        ).getPotentialEnergy()
        print(
            f"  step {block + chunk}/{PROD_STEPS}, "
            f"potential {energy.value_in_unit(unit.kilojoule_per_mole):8.2f}"
            " kJ/mol",
            flush=True,
        )
    masses = np.array([record["mass"] for record in records])
    rg = radius_of_gyration(
        OUT / f"{PROTEIN_ID.lower()}.pdb", OUT / "trajectory.dcd", masses
    )
    np.savetxt(OUT / "rg.txt", rg, fmt="%.4f")
    tail = rg[len(rg) // 2 :]
    exp_rg, exp_err = float(record["exp_rg"]), float(record["exp_rg_err"])
    print(
        f"{PROTEIN_ID} Rg (mass-weighted): mean over second half "
        f"{tail.mean():.3f} nm; "
        f"experimental {exp_rg:.2f} +- {exp_err:.2f} nm "
        f"({len(records)}-residue pilot - "
        f"{PROD_STEPS * DT // 1000:g} ns is far short of the upstream 18 us)"
    )
    print(f"outputs in {OUT}")


if __name__ == "__main__":
    main()
