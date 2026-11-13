"""Dump a raw M1_24 pickle to an Excel file.

Usage:
    python -m examples.calvados.pickle_to_excel 5_init_proteinsRgs.pkl
    python -m examples.calvados.pickle_to_excel 5_chi2.pkl
Requires: openpyxl  (pip install openpyxl)
"""

import sys

import pandas as pd
import numpy as np
import json

from examples.calvados.fetch_data import RAW


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    name = sys.argv[1]
    obj = pd.read_pickle(RAW / name)
    out = RAW.parent / name.replace(".pkl", ".xlsx")

    if isinstance(obj, pd.DataFrame):
        frame = obj.copy()
        # Excel has no list column type: pack list columns (e.g. rgarray)
        # into JSON strings so nothing is lost
        for col in frame.columns:
            if frame[col].map(lambda v: isinstance(v, (list, tuple, np.ndarray))).any():
                frame[col] = frame[col].map(
                    lambda v: v if isinstance(v, str) else json.dumps(list(v))
                )
    elif isinstance(obj, dict):
        frame = pd.DataFrame.from_dict(obj, orient="index")
    else:
        sys.exit(f"unsupported pickle type: {type(obj).__name__}")

    frame.to_excel(out, sheet_name=name[:28])
    print(f"wrote {out} ({frame.shape[0]} rows x {frame.shape[1]} columns)")


if __name__ == "__main__":
    main()
