#!/usr/bin/env python3
"""Export the deterministic evaluation membership used by grouped validation.

The real-image random five-fold split is generated with NumPy's PCG64 default
generator at seed 0.  This file makes that membership inspectable rather than
leaving it implicit in the evaluator.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    eligible = [r for r in records if r.get("colour_training_eligible", True)]
    real = [r for r in eligible if r.get("real_image", False)]
    order = np.random.default_rng(0).permutation(len(real))
    fold_by_index = {}
    for fold, indices in enumerate(np.array_split(order, 5), start=1):
        for index in indices:
            fold_by_index[int(index)] = fold
    with args.output.open("w", newline="") as handle:
        fields = ["id", "species", "source_pmcid", "real_image", "colour_training_eligible", "random_5fold_seed0"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for i, record in enumerate(real):
            writer.writerow({
                "id": record["id"],
                "species": record.get("species", ""),
                "source_pmcid": (record.get("source") or {}).get("pmcid", ""),
                "real_image": True,
                "colour_training_eligible": True,
                "random_5fold_seed0": fold_by_index[i],
            })
    print(f"wrote {len(real)} real-image fold assignments to {args.output}")


if __name__ == "__main__":
    main()
