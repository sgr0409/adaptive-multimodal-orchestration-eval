"""Convert official variable-length MIntRec2.0 features to fixed summaries."""

import argparse
import pickle
from pathlib import Path

import numpy as np


def summarize(value):
    rows = np.asarray(value, dtype=np.float32)
    if rows.ndim == 3 and rows.shape[1] == 1:
        rows = rows[:, 0, :]
    if rows.ndim == 1:
        rows = rows[None, :]
    finite = np.all(np.isfinite(rows), axis=1)
    rows = rows[finite]
    if len(rows) == 0:
        raise ValueError("feature sequence contains no finite frames")
    return np.concatenate([rows.mean(axis=0), rows.std(axis=0)]).astype(
        np.float32
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open("rb") as stream:
        features = pickle.load(stream)
    keys = sorted(features)
    summaries = np.stack([summarize(features[key]) for key in keys])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        keys=np.asarray(keys),
        features=summaries,
    )
    print({"examples": len(keys), "shape": summaries.shape})


if __name__ == "__main__":
    main()
