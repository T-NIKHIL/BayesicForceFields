"""Markov-chain generation of disordered-protein sequences.

Fits an order-k amino-acid chain with START/END tokens and backoff to a
curated dataset of intrinsically disordered proteins and samples sequences
from a given starting residue. All dataset utilities live here:

- disprot: region-level export ("DisProt current IDPO-GO") from
  https://disprot.org/download. There is no programmatic bulk URL, so the
  TSV must be saved manually to examples/calvados/IDP_datasets/DisProt_current_IDPO-GO.tsv;
  fetch_disprot verifies it. Sequences are the annotated disordered regions.
- phasepro: full-length proteins that undergo liquid-liquid phase
  separation, fetched from https://phasepro.elte.hu/download_full.json.

Standard library only.
"""

import argparse
import csv
import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

AA20 = "ACDEFGHIKLMNPQRSTVWY"
START, END = "^", "$"

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "IDP_datasets"
DISPROT_TSV = DATA_DIR / "DisProt_current_IDPO-GO.tsv"
PHASEPRO_JSON = DATA_DIR / "phasepro.json"
PHASEPRO_URL = "https://phasepro.elte.hu/download_full.json"
# Pins from the first verified downloads; update after re-downloading.
PHASEPRO_SHA256 = "ad2a5aed1ef6a2dac6a1da4259d0f3747ee8a35a77140b1582f4d27909662092"
DISPROT_SHA256 = "58ce49288d33cab04b1f41de73a067d5165cae93d37eae079d10d83ba06672b8"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_phasepro(path: Path = PHASEPRO_JSON, url: str = PHASEPRO_URL) -> Path:
    """Download the PhaSePro JSON dump; verify against the pinned sha256."""
    if path.exists():
        if sha256_bytes(path.read_bytes()) == PHASEPRO_SHA256:
            print(f"PhaSePro: verified {path}")
            return path
        raise ValueError(f"PhaSePro checksum mismatch: {path}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if sha256_bytes(data) != PHASEPRO_SHA256:
        raise ValueError(f"PhaSePro checksum mismatch on download from {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)
    print(f"PhaSePro: downloaded {len(data)} bytes to {path}")
    return path


def fetch_disprot(path: Path = DISPROT_TSV) -> Path:
    """Verify the manually downloaded DisProt region TSV against its pin."""
    if not path.exists():
        raise FileNotFoundError(
            f"DisProt: {path} missing; save the 'DisProt current IDPO-GO' TSV "
            "manually from https://disprot.org/download"
        )
    if sha256_bytes(path.read_bytes()) != DISPROT_SHA256:
        raise ValueError(
            f"DisProt checksum mismatch: {path}; re-pin DISPROT_SHA256 "
            "if you intentionally updated the export"
        )
    print(f"DisProt: verified {path}")
    return path


def parse_disprot_tsv(path: Path = DISPROT_TSV, min_length: int = 20):
    """Disordered regions from the IDPO-GO export: one region per row."""
    sequences, seen = [], set()
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sequence = (row.get("Region sequence") or "").strip().upper()
            if (
                len(sequence) >= min_length
                and set(sequence) <= set(AA20)
                and sequence not in seen
            ):
                seen.add(sequence)
                sequences.append(sequence)
    return sequences


def parse_phasepro_json(path: Path = PHASEPRO_JSON, min_length: int = 20):
    """Full-length protein sequences from the PhaSePro JSON dump."""
    records = json.loads(path.read_text())
    sequences, seen = [], set()
    for record in records.values():
        sequence = (record.get("sequence") or "").strip().upper()
        if (
            len(sequence) >= min_length
            and set(sequence) <= set(AA20)
                and sequence not in seen
        ):
            seen.add(sequence)
            sequences.append(sequence)
    return sequences


LOADERS = {
    "disprot": lambda: parse_disprot_tsv(fetch_disprot()),
    "phasepro": lambda: parse_phasepro_json(fetch_phasepro()),
}


def load_sequences(dataset: str):
    """Fetch, verify, and parse a dataset by name."""
    if dataset not in LOADERS:
        raise ValueError(f"unknown dataset {dataset!r}; expected {sorted(LOADERS)}")
    return LOADERS[dataset]()


def amino_acid_composition(sequences):
    """Relative frequency of each standard amino acid across all sequences."""
    counts = Counter("".join(sequences))
    total = sum(counts.values())
    return {aa: counts[aa] / total for aa in AA20}


class MarkovChain:
    """Order-k amino-acid chain with backoff to shorter contexts."""

    def __init__(self, order: int = 1, alpha: float = 0.1, rng: random.Random = None):
        if order < 0:
            raise ValueError("MarkovChain order must be >= 0")
        self.order = order
        self.alpha = alpha
        self.rng = rng if rng is not None else random.Random()
        self.alphabet = AA20 + END
        self.counts = {k: Counter() for k in range(order + 1)}
        self.lengths = None

    def fit(self, sequences):
        self.lengths = [len(sequence) for sequence in sequences]
        for sequence in sequences:
            symbols = START + sequence + END
            for i in range(1, len(symbols)):
                for k in range(self.order + 1):
                    if i - k >= 0:
                        self.counts[k][(symbols[i - k : i], symbols[i])] += 1
        return self

    def sample_length(self):
        """Target length drawn from the corpus's empirical length distribution."""
        if self.lengths is None:
            raise ValueError("MarkovChain: call fit() before sample_length()")
        return self.rng.choice(self.lengths)

    def distribution(self, sequence: str) -> dict:
        """Smoothed next-residue distribution, deepest available context first."""
        symbols = START + sequence
        counts, suffix = self.counts[0], ""
        for k in range(min(self.order, len(symbols)), 0, -1):
            candidate = symbols[len(symbols) - k :]
            if any(self.counts[k][(candidate, aa)] for aa in self.alphabet):
                counts, suffix = self.counts[k], candidate
                break
        weights = [counts[(suffix, aa)] + self.alpha for aa in self.alphabet]
        total = sum(weights)
        return dict(zip(self.alphabet, [w / total for w in weights]))

    def sample(self, start: str, max_length: int = 1000, rng=random,
               length: int = None):
        """Generate a sequence from `start`, ending at the END token.

        With `length=None` the chain ends stochastically at the END token.
        With an explicit `length` (e.g. from sample_length()) the END token
        is excluded and exactly `length` residues are emitted; the corpus
        length distribution then carries the stopping behavior.
        """
        sequence = start
        if length is not None:
            while len(sequence) < length:
                distribution = self.distribution(sequence)
                residue = rng.choices(
                    [aa for aa in distribution if aa != END],
                    weights=[w for aa, w in distribution.items() if aa != END],
                )[0]
                sequence += residue
            return sequence[:length]
        while len(sequence) < max_length:
            distribution = self.distribution(sequence)
            residue = rng.choices(
                list(distribution), weights=list(distribution.values())
            )[0]
            if residue == END:
                return sequence
            sequence += residue
        return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(LOADERS))
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--start", default="M")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--end-token",
        action="store_true",
        help="stop at the learned END token instead of drawing "
        "the length from the corpus's empirical length distribution",
    )
    arguments = parser.parse_args()

    corpus = load_sequences(arguments.dataset)
    print(
        f"{arguments.dataset}: {len(corpus)} sequences, "
        f"{sum(map(len, corpus))} residues, "
        f"mean length {sum(map(len, corpus)) / len(corpus):.1f}"
    )
    rng = random.Random(arguments.seed)
    chain = MarkovChain(order=arguments.order, rng=rng).fit(corpus)
    reference = amino_acid_composition(corpus)
    generated = []
    for _ in range(arguments.samples):
        length = None if arguments.end_token else chain.sample_length()
        generated.append(chain.sample(arguments.start, rng=rng, length=length))
    observed = amino_acid_composition(generated)
    print("AA  corpus  model")
    for residue in AA20:
        print(f"{residue}  {reference[residue]:.3f}   {observed[residue]:.3f}")
    for sequence in generated:
        print(len(sequence), sequence)


if __name__ == "__main__":
    main()
