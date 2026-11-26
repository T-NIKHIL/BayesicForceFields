"""Analyze explicitly disordered PhaSePro LLPS-driver regions.

Run in the bff environment:
    python examples/calvados/IDP_LLPS_datasets/analyze_phasepro_idrs.py

Fetches JSON into memory; never saves the database response. Outputs derived
CSVs, a provenance/summary JSON, and PNG plots beside this script by default.
Selection uses region descriptions, not a disorder predictor. It is deliberately
incomplete: low complexity or a prion-like label alone does not establish disorder.
Source: https://doi.org/10.1093/nar/gkz848
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

AA20 = "RDNEKHQSCGTAMYVWLIPF"
BINS = ("<100", "100-200", ">200")
URL = "https://phasepro.elte.hu/download_full.json"
HERE = Path(__file__).resolve().parent
DISORDER = re.compile(
    r"\b(?:disordered|unstructured|IDRs?|IDPs?|[mc]IDR)\b", re.IGNORECASE
)
# A whole annotated interval containing both an IDR and a folded domain cannot
# be treated as an IDR. Unclear descriptions remain available in excluded.csv.
MIXED = re.compile(
    r"full[ -](?:protein|length)|only full|\bRRMs?\b|"
    r"RNA[- ](?:binding|recognition)|\bKH\b|ubiquitin.like|"
    r"\b(?:ordered|folded|structured)\b|\b(?:SH[23]|PDZ|BTB|MATH)\b",
    re.IGNORECASE,
)


def length_bin(length):
    """100 and 200 belong to the middle bin."""
    if length < 100:
        return BINS[0]
    if length <= 200:
        return BINS[1]
    return BINS[2]


def fetch_database(url=URL):
    """Fetch once, retaining response bytes in memory only."""
    request = Request(url, headers={"User-Agent": "CALVADOS-IDR-analysis/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    records = json.loads(payload)
    if not isinstance(records, dict) or not records:
        raise ValueError("PhaSePro: expected a nonempty accession-to-record JSON object")
    provenance = {
        "url": url,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "response_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return records, provenance


def extract_regions(records):
    """Pair semicolon-separated descriptions/boundaries; use 1-based closed ranges."""
    selected, excluded = [], []
    for key, record in records.items():
        accession = record.get("accession") or key
        sequence = str(record.get("sequence") or "").strip().upper()
        boundaries = str(record.get("boundaries") or "").strip()
        annotation = str(record.get("segment") or "").strip()
        common = {
            "accession": accession,
            "name": record.get("common_name") or record.get("name", ""),
        }
        if not sequence or set(sequence) - set(AA20):
            excluded.append({
                **common, "boundaries": boundaries, "annotation": annotation,
                "reason": "missing sequence or nonstandard amino acids",
            })
            continue
        ranges = [part.strip() for part in boundaries.split(";")]
        descriptions = [part.strip() for part in annotation.split(";")]
        if len(ranges) != len(descriptions):
            excluded.append({
                **common, "boundaries": boundaries, "annotation": annotation,
                "reason": "cannot pair interval descriptions with boundaries",
            })
            continue
        seen = set()
        for coordinates, description in zip(ranges, descriptions):
            context = {
                **common, "boundaries": coordinates, "annotation": description,
            }
            match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", coordinates)
            if not match:
                excluded.append({**context, "reason": "unrecognized boundary format"})
                continue
            start, end = map(int, match.groups())
            if not 1 <= start <= end <= len(sequence):
                excluded.append({**context, "reason": "coordinates outside sequence"})
                continue
            if not DISORDER.search(description):
                excluded.append({**context, "reason": "no explicit disorder annotation"})
                continue
            if MIXED.search(description):
                excluded.append({**context, "reason": "mixed or folded-domain annotation"})
                continue
            if (start, end) in seen:
                excluded.append({**context, "reason": "duplicate interval in entry"})
                continue
            seen.add((start, end))
            region_sequence = sequence[start - 1:end]
            selected.append({
                **common, "start": start, "end": end,
                "length": len(region_sequence),
                "length_bin": length_bin(len(region_sequence)),
                "annotation": description, "sequence": region_sequence,
            })
    return selected, excluded


def summarize(regions):
    """Pool residue counts, weighting each residue equally within its length bin."""
    composition, bins = [], {}
    for label in BINS:
        members = [row for row in regions if row["length_bin"] == label]
        counts = Counter("".join(row["sequence"] for row in members))
        total = sum(counts.values())
        bins[label] = {
            "regions": len(members),
            "proteins": len({row["accession"] for row in members}),
            "residues": total,
        }
        for aa in AA20:
            composition.append({
                "length_bin": label, "amino_acid": aa, "count": counts[aa],
                # No composition exists for an empty bin; do not imply zero frequency.
                "fraction": counts[aa] / total if total else None,
                "total_residues": total, "n_regions": len(members),
            })
    lengths = sorted(row["length"] for row in regions)
    if not lengths:
        raise ValueError("PhaSePro: no explicitly disordered driver regions selected")
    n = len(lengths)
    median = (lengths[(n - 1) // 2] + lengths[n // 2]) / 2
    return composition, {
        "selected_regions": n,
        "selected_proteins": len({row["accession"] for row in regions}),
        "total_residues": sum(lengths),
        "length_aa": {
            "min": lengths[0], "median": median,
            "mean": sum(lengths) / n, "max": lengths[-1],
        },
        "length_bins": bins,
    }


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(regions, composition):
    """Return figures without showing them or writing files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengths = [row["length"] for row in regions]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(lengths, bins="auto", color="steelblue", edgecolor="white")
    for cutoff in (100, 200):
        axes[0].axvline(cutoff, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Driver-region length (aa)", ylabel="Number of regions")
    axes[1].bar(BINS, [
        sum(row["length_bin"] == label for row in regions) for label in BINS
    ], color="steelblue")
    axes[1].set(xlabel="Length bin (aa)", ylabel="Number of regions")
    figure.suptitle("PhaSePro: explicitly disordered LLPS-driver regions")
    figure.tight_layout()

    aa_figure, axis = plt.subplots(figsize=(12, 4))
    for index, label in enumerate(BINS):
        rows = sorted(
            (row for row in composition if row["length_bin"] == label),
            key=lambda row: AA20.index(row["amino_acid"]),
        )
        if not rows[0]["total_residues"]:
            continue
        axis.bar(
            [i + (index - 1) * 0.26 for i in range(len(AA20))],
            [row["fraction"] for row in rows], width=0.26,
            label=f"{label} aa (n={rows[0]['n_regions']})",
        )
    axis.set_xticks(range(len(AA20)), list(AA20))
    axis.set(
        xlabel="Amino acid", ylabel="Fraction of residues within length bin",
        title="PhaSePro: pooled amino acid composition by region length",
    )
    axis.legend()
    aa_figure.tight_layout()
    return figure, aa_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=URL, help="PhaSePro full JSON endpoint")
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "phasepro_idr_analysis",
        help="Directory for derived analysis outputs (not the source database)",
    )
    args = parser.parse_args()
    records, provenance = fetch_database(args.url)
    regions, excluded = extract_regions(records)
    composition, summary = summarize(regions)
    figures = make_plots(regions, composition)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "regions.csv", regions, (
        "accession", "name", "start", "end", "length", "length_bin", "annotation",
    ))
    write_csv(output / "excluded.csv", excluded, (
        "accession", "name", "boundaries", "annotation", "reason",
    ))
    write_csv(output / "amino_acid_composition.csv", composition, (
        "length_bin", "amino_acid", "count", "fraction", "total_residues", "n_regions",
    ))
    summary.update({
        "source": provenance, "source_entries": len(records),
        "excluded_items": len(excluded),
        "selection": (
            "Explicit disorder/unstructured/IDR/IDP description; exclude mixed "
            "folded-domain annotations and unpaired descriptions. Not a disorder "
            "prediction or a complete catalog of LLPS IDRs. Separate intervals "
            "are separate observations, not necessarily autonomous LLPS constructs."
        ),
        "counting": (
            "1-based inclusive coordinates; no minimum length; retain distinct "
            "accessions even if sequences match; pool residues within each bin. "
            "100 and 200 aa are included in the middle bin."
        ),
    })
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    import matplotlib.pyplot as plt
    for figure, filename in zip(
        figures, ("length_distribution.png", "amino_acid_composition.png")
    ):
        figure.savefig(output / filename, dpi=180)
        plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"Analysis outputs: {output}")


if __name__ == "__main__":
    main()
