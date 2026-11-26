"""Compare LLPSDB v2.0 IDR composition with the 51 CALVADOS 2 Rg proteins.

Run in the bff environment (pandas, openpyxl, matplotlib):
    python examples/calvados/IDP_LLPS_datasets/analyze_llpsdb_idrs.py

The official positive/unambiguous ZIP and its two Excel tables are read in
memory only. No database files are saved. Derived results go to llpsdb_idr_analysis.

Selection: natural, single-protein systems without nucleic acids, mutations,
fusions, engineered repeats or PTMs. Cleaved constructs are allowed only when
their sequence maps exactly to the parent sequence. Intersect tested construct
coordinates with the parent's LLPSDB IDR annotations, and merge overlapping
coverage per protein so repeated experiments do not multiply residue counts.
These are IDRs within LLPS-positive constructs; isolated IDR sufficiency is not
implied. LLPSDB IDR annotations can be curated or predicted (the export does
not distinguish these sources). No disorder prediction is performed here.

Study: https://doi.org/10.1093/bioinformatics/btac026
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

AA20 = "RDNEKHQSCGTAMYVWLIPF"
BINS = ("<100", "100-200", ">200")
HERE = Path(__file__).resolve().parent
URL = (
    "http://bio-comp.org.cn/llpsdbv2/download/"
    "Phase_separation_unambiguous.zip"
)
REFERENCE = HERE.parent / "M1_24_analysis_scripts/train_set_idp_composition.csv"


def length_bin(length):
    return BINS[0] if length < 100 else BINS[1] if length <= 200 else BINS[2]


def fetch_database(url=URL):
    import pandas as pd

    request = Request(url, headers={"User-Agent": "CALVADOS-IDR-analysis/1.0"})
    with urlopen(request, timeout=120) as response:
        payload = response.read()
        resolved_url = response.geturl()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        tables = {}
        for filename in ("LLPS.xls", "protein.xls"):
            matches = [
                name for name in archive.namelist()
                if name.rsplit("/", 1)[-1] == filename
            ]
            if len(matches) != 1:
                raise ValueError(f"LLPSDB: expected exactly one {filename} in export")
            # Official .xls-named files are actually OOXML workbooks.
            tables[filename] = pd.read_excel(
                io.BytesIO(archive.read(matches[0])), engine="openpyxl"
            ).fillna("").to_dict("records")
    return tables["LLPS.xls"], tables["protein.xls"], {
        "url": url, "resolved_url": resolved_url,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "response_sha256": hashlib.sha256(payload).hexdigest(),
    }


def parse_sequence(value):
    """Accept one FASTA record, including semicolon-wrapped parent sequences."""
    text = str(value).strip().replace(";", "\n")
    if text.count(">") != 1 or not text.startswith(">"):
        raise ValueError("expected exactly one FASTA sequence")
    sequence = "".join(text.splitlines()[1:]).replace(" ", "").upper()
    if not sequence or set(sequence) - set(AA20):
        raise ValueError("missing sequence or nonstandard amino acids")
    return sequence


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(set(intervals)):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def parse_idrs(value, length):
    text = str(value).strip()
    if text in ("", "-", "0-0"):
        return []
    intervals = []
    for token in text.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", token)
        if not match:
            raise ValueError(f"unrecognized IDR coordinates: {text}")
        start, end = map(int, match.groups())
        if not 1 <= start <= end <= length:
            raise ValueError(f"IDR coordinates outside parent sequence: {token}")
        intervals.append((start, end))
    return merge_intervals(intervals)


def construct_interval(fasta, sequence, parent):
    match = re.search(r"\|residues=(\d+)-(\d+)(?:\s|$)", str(fasta).splitlines()[0])
    if match:
        start, end = map(int, match.groups())
        if 1 <= start <= end <= len(parent) and parent[start - 1:end] == sequence:
            return start, end
    # Coordinates can refer to a different sequence version. Accept a unique
    # exact mapping; never infer alignment across mutations/deletions.
    start = parent.find(sequence)
    if start < 0 or parent.find(sequence, start + 1) >= 0:
        raise ValueError("construct does not map uniquely and exactly to parent")
    return start + 1, start + len(sequence)


def select_regions(entries, proteins):
    required_entry = {
        "Entry ID", "Protein ID", "Components type", "Protein type (N/D)",
        "Fusion", "Mutation", "Repeat", "PTM", "Sequence", "Phase separation",
    }
    required_protein = {"PID", "Uniprot ID", "Sequence", "IDR"}
    if not entries or not proteins:
        raise ValueError("LLPSDB: empty experiment or protein table")
    for name, rows, required in (
        ("experiments", entries, required_entry),
        ("proteins", proteins, required_protein),
    ):
        missing = required - rows[0].keys()
        if missing:
            raise ValueError(f"LLPSDB {name}: missing columns {sorted(missing)}")
    parents = {str(row["PID"]): row for row in proteins}
    if len(parents) != len(proteins):
        raise ValueError("LLPSDB: duplicate parent protein IDs")
    covered, evidence, audit, seen = defaultdict(list), defaultdict(set), [], set()
    parent_sequences = {}
    for row in entries:
        # Collapse repeated conditions, retaining distinct construct definitions.
        key = tuple(str(row[name]) for name in sorted(required_entry))
        if key in seen:
            continue
        seen.add(key)
        pid = str(row["Protein ID"]).strip()
        entry = str(row["Entry ID"]).strip()
        reason = ""
        if str(row["Phase separation"]).strip().lower() != "yes":
            reason = "not a positive phase-separation experiment"
        elif str(row["Components type"]).strip().lower() != "protein(1)":
            reason = "not a single-protein system without nucleic acids"
        elif str(row["Protein type (N/D)"]).strip() != "N":
            reason = "not a natural protein"
        elif any(str(row[name]).strip() not in ("", "-") for name in (
            "Fusion", "Mutation", "Repeat", "PTM",
        )):
            reason = "fusion, mutation, engineered repeat or PTM"
        elif pid not in parents:
            reason = "missing parent protein ID"
        else:
            try:
                parent = parse_sequence(parents[pid]["Sequence"])
                idrs = parse_idrs(parents[pid]["IDR"], len(parent))
                if not idrs:
                    raise ValueError("no annotated parent IDR")
                sequence = parse_sequence(row["Sequence"])
                start, end = construct_interval(row["Sequence"], sequence, parent)
                intersections = [
                    (max(start, left), min(end, right))
                    for left, right in idrs if max(start, left) <= min(end, right)
                ]
                if not intersections:
                    raise ValueError("tested construct does not overlap an annotated IDR")
            except ValueError as error:
                reason = str(error)
            else:
                parent_sequences[pid] = parent
                covered[pid].extend(intersections)
                for left, right in intersections:
                    evidence[(pid, left, right)].add(entry)
        audit.append({
            "entry_id": entry, "protein_id": pid,
            "included": not bool(reason), "reason": reason or "eligible IDR coverage",
        })
    regions = []
    for pid, intervals in sorted(covered.items()):
        protein = parents[pid]
        for start, end in merge_intervals(intervals):
            sequence = parent_sequences[pid][start - 1:end]
            ids = set()
            for (source_pid, left, right), entries_for_interval in evidence.items():
                if source_pid == pid and max(start, left) <= min(end, right):
                    ids.update(entries_for_interval)
            regions.append({
                "protein_id": pid, "accession": protein["Uniprot ID"] or pid,
                "name": protein.get("Protein name abbreviation", ""),
                "start": start, "end": end, "length": len(sequence),
                "length_bin": length_bin(len(sequence)), "sequence": sequence,
                "entry_ids": ";".join(sorted(ids)),
            })
    if not regions:
        raise ValueError("LLPSDB: no IDR regions passed selection; inspect source schema")
    return regions, audit


def composition_rows(regions):
    rows = []
    for label in ("all", *BINS):
        members = [
            row for row in regions if label == "all" or row["length_bin"] == label
        ]
        counts = Counter("".join(row["sequence"] for row in members))
        total = sum(counts.values())
        for aa in AA20:
            rows.append({
                "length_bin": label, "amino_acid": aa, "count": counts[aa],
                "fraction": counts[aa] / total if total else None,
                "total_residues": total, "n_regions": len(members),
            })
    return rows


def reference_counts(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 51 or len({row["protein_id"] for row in rows}) != 51:
        raise ValueError(f"CALVADOS 2: expected 51 distinct training proteins in {path}")
    counts = Counter()
    for row in rows:
        values = {aa: int(row[f"{aa}_count"]) for aa in AA20}
        if any(value < 0 for value in values.values()):
            raise ValueError(f"CALVADOS 2: negative residue count in {path}")
        if sum(values.values()) != int(row["length"]):
            raise ValueError(f"CALVADOS 2: counts disagree with length in {path}")
        counts.update(values)
    return counts


def compare(composition, counts):
    total = sum(counts.values())
    reference = {aa: counts[aa] / total for aa in AA20}
    rows, metrics = [], {}
    for label in ("all", *BINS):
        profile = [row for row in composition if row["length_bin"] == label]
        if not profile[0]["total_residues"]:
            continue
        q = {row["amino_acid"]: row["fraction"] for row in profile}
        midpoint = {aa: (q[aa] + reference[aa]) / 2 for aa in AA20}
        metrics[label] = {
            "jensen_shannon_divergence_bits": sum(
                0.5 * p[aa] * math.log2(p[aa] / midpoint[aa])
                for p in (q, reference) for aa in AA20 if p[aa]
            ),
            "total_variation": sum(abs(q[aa] - reference[aa]) for aa in AA20) / 2,
            "missing_amino_acids": [aa for aa in AA20 if q[aa] == 0],
        }
        for aa in AA20:
            rows.append({
                "length_bin": label, "amino_acid": aa,
                "llpsdb_fraction": q[aa], "calvados2_fraction": reference[aa],
                "difference_percentage_points": 100 * (q[aa] - reference[aa]),
            })
    return rows, metrics


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(regions, composition, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    length_figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist([row["length"] for row in regions], bins="auto",
                 color="steelblue", edgecolor="white")
    for cutoff in (100, 200):
        axes[0].axvline(cutoff, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Selected IDR length (aa)", ylabel="Number of regions")
    axes[1].bar(BINS, [
        sum(row["length_bin"] == label for row in regions) for label in BINS
    ], color="steelblue")
    axes[1].set(xlabel="Length bin (aa)", ylabel="Number of regions")
    length_figure.suptitle("LLPSDB v2.0: IDRs within LLPS-positive constructs")
    length_figure.tight_layout()

    aa_figure, axis = plt.subplots(figsize=(12, 4))
    for index, label in enumerate(BINS):
        profile = {row["amino_acid"]: row for row in composition
                   if row["length_bin"] == label}
        if not profile[AA20[0]]["total_residues"]:
            continue
        axis.bar(
            [i + (index - 1) * 0.26 for i in range(20)],
            [profile[aa]["fraction"] for aa in AA20], width=0.26,
            label=f"{label} aa (n={profile[AA20[0]]['n_regions']})",
        )
    axis.set_xticks(range(20), list(AA20))
    axis.set(xlabel="Amino acid", ylabel="Fraction of residues within length bin",
             title="LLPSDB v2.0: pooled IDR composition by length")
    axis.legend()
    aa_figure.tight_layout()

    comparison_figure, axis = plt.subplots(figsize=(12, 4))
    overall = {row["amino_acid"]: row for row in comparison
               if row["length_bin"] == "all"}
    for offset, field, label in (
        (-0.2, "llpsdb_fraction", "LLPSDB selected IDRs"),
        (0.2, "calvados2_fraction", "CALVADOS 2: 51 Rg training proteins"),
    ):
        axis.bar([i + offset for i in range(20)],
                 [overall[aa][field] for aa in AA20], width=0.4, label=label)
    axis.set_xticks(range(20), list(AA20))
    axis.set(xlabel="Amino acid", ylabel="Fraction of pooled residues",
             title="Amino acid composition: LLPSDB vs CALVADOS 2")
    axis.legend()
    comparison_figure.tight_layout()
    return {
        "length_distribution": length_figure,
        "amino_acid_composition": aa_figure,
        "calvados2_comparison": comparison_figure,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=URL, help="Official positive-system ZIP URL")
    parser.add_argument("--calvados-csv", type=Path, default=REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "llpsdb_idr_analysis")
    args = parser.parse_args()
    # Check local reference first, before making the network request.
    counts = reference_counts(args.calvados_csv)
    entries, proteins, provenance = fetch_database(args.url)
    regions, audit = select_regions(entries, proteins)
    composition = composition_rows(regions)
    comparison, metrics = compare(composition, counts)
    figures = make_plots(regions, composition, comparison)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "regions.csv", regions, (
        "protein_id", "accession", "name", "start", "end",
        "length", "length_bin", "entry_ids",
    ))
    write_csv(output / "selection_audit.csv", audit, (
        "entry_id", "protein_id", "included", "reason",
    ))
    write_csv(output / "amino_acid_composition.csv", composition, (
        "length_bin", "amino_acid", "count", "fraction", "total_residues", "n_regions",
    ))
    write_csv(output / "calvados2_comparison.csv", comparison, (
        "length_bin", "amino_acid", "llpsdb_fraction", "calvados2_fraction",
        "difference_percentage_points",
    ))
    lengths = [row["length"] for row in regions]
    summary = {
        "source": provenance,
        "source_experiment_rows": len(entries),
        "source_unique_entries": len({row["Entry ID"] for row in entries}),
        "source_proteins": len(proteins),
        "selected_regions": len(regions),
        "selected_protein_ids": len({row["protein_id"] for row in regions}),
        "selected_accessions": len({row["accession"] for row in regions}),
        "length_aa": {
            "min": min(lengths), "median": statistics.median(lengths),
            "mean": statistics.mean(lengths), "max": max(lengths),
        },
        "length_bins": {
            label: {
                "regions": sum(row["length_bin"] == label for row in regions),
                "residues": sum(row["length"] for row in regions
                                if row["length_bin"] == label),
            } for label in BINS
        },
        "audit_reasons": dict(Counter(row["reason"] for row in audit)),
        "calvados2_reference": {
            "path": str(args.calvados_csv.resolve()), "proteins": 51,
            "residues": sum(counts.values()),
            "sha256": hashlib.sha256(args.calvados_csv.read_bytes()).hexdigest(),
        },
        "composition_comparison_with_all_51_training_proteins": metrics,
        "selection_and_limitations": __doc__,
        "counting": (
            "1-based inclusive parent coordinates; no minimum region length. "
            "Merge overlapping/adjacent tested IDR coverage per parent protein. "
            "Retain distinct parent proteins even if sequences match. "
            "Pool residue counts within bins; 100 and 200 belong to middle bin. "
            "Missing annotation and predicted/curated disorder are not equivalent. "
            "Dataset overlap/homology with CALVADOS is not removed; composition "
            "similarity alone does not establish model generalization."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    import matplotlib.pyplot as plt
    for name, figure in figures.items():
        for extension in ("png", "jpg"):
            figure.savefig(output / f"{name}.{extension}", dpi=180)
        plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"Analysis outputs: {output}")


if __name__ == "__main__":
    main()
