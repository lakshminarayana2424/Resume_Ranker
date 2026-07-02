#!/usr/bin/env python3
"""
08_build_lgbm_features.py
==========================
Build the LightGBM feature matrix for the full candidate pool.

What this script does
---------------------
This script creates the input feature CSV used by the LightGBM ranker. It reads
the candidate dataset, derives a stable reference date, and builds one feature
row per candidate using deterministic resume/profile logic.

Important behavior:
- This script does NOT create targets or labels.
  See 09_build_lgbm_targets.py for target generation.
- This script does NOT use any LLM ranking scores as input features.
  Those scores are used only when building targets, because they are not
  available for the full candidate pool or at timed inference.
- Features are derived only from:
  - the 23 hackathon-provided redrob_signals fields
  - structured candidate JSON fields such as career_history, education,
    skills, and certifications
  - deterministic keyword/date logic inside feature_engineering.py

Docker / path usage
-------------------
All paths are resolved from the current working directory based on the values
passed to the CLI. There is no hardcoded Resume_Ranker/ path in this script.
For Docker compatibility, run it from the repository root or any working
directory where the input and output paths you pass are valid.

Inputs
------
- candidates.jsonl / candidates.jsonl.gz / JSON array input
- Optional explicit --reference-date override

Output
------
- CSV file containing candidate_id plus FEATURE_COLUMNS

USAGE
-----
Run from the repo root or any directory where the provided paths are valid:

    python 08_build_lgbm_features.py         --candidates candidates.jsonl         --out outputs/lgbm_features_100k.csv

Notes
-----
- The script performs two passes:
  1) load candidates and derive the reference date
  2) build and write the final feature CSV
- If you rebuild features later, reuse the same reference date with
  --reference-date so recency features remain consistent between training and
  submission.
"""


import argparse
import csv
import gzip
import json
import time
from datetime import date
from typing import Iterable

from feature_engineering import (
    FEATURE_COLUMNS,
    build_feature_row,
    derive_reference_date,
)


def iter_candidates(path: str) -> Iterable[dict]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data:
            yield c


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--candidates", required=True,
                     help="Path to candidates.jsonl, candidates.jsonl.gz, or "
                          "a plain JSON array (e.g. sample_candidates.json).")
    ap.add_argument("--out", required=True,
                     help="Output CSV path, e.g. outputs/lgbm_features_100k.csv")
    ap.add_argument("--reference-date", default=None,
                     help="Override the auto-derived reference date "
                          "(YYYY-MM-DD). Leave unset to auto-derive from the "
                          "dataset itself (recommended -- see module "
                          "docstring in feature_engineering.py for why).")
    ap.add_argument("--limit", type=int, default=None,
                     help="Debug only: stop after N candidates.")
    args = ap.parse_args()

    t0 = time.time()

    print(f"Pass 1/2: scanning {args.candidates} to derive reference date "
          f"and candidate count ...")
    n_total = 0
    candidates_for_ref_date = []
    for c in iter_candidates(args.candidates):
        candidates_for_ref_date.append(c)
        n_total += 1
        if args.limit and n_total >= args.limit:
            break
    print(f"  {n_total} candidates loaded into memory.")

    if args.reference_date:
        ref_date = date.fromisoformat(args.reference_date)
        print(f"  Using explicit --reference-date={ref_date}")
    else:
        ref_date = derive_reference_date(candidates_for_ref_date)
        print(f"  Auto-derived reference date from dataset: {ref_date}")

    out_columns = ["candidate_id"] + FEATURE_COLUMNS

    print(f"\nPass 2/2: building {len(FEATURE_COLUMNS)} resume-derived "
          f"features per candidate ({len(out_columns)} output columns "
          f"total) ...")
    n_written = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_columns)
        writer.writeheader()
        for c in candidates_for_ref_date:
            row = build_feature_row(c, ref_date)
            writer.writerow(row)
            n_written += 1
            if n_written % 10000 == 0:
                elapsed = time.time() - t0
                rate = n_written / elapsed
                eta = (n_total - n_written) / rate if rate > 0 else float("nan")
                print(f"  {n_written}/{n_total} done "
                      f"({rate:.0f} rows/sec, ETA {eta:.0f}s) ...")

    elapsed = time.time() - t0
    print(f"\nWrote {n_written} feature rows to {args.out} in {elapsed:.1f}s "
          f"({n_written / elapsed:.0f} rows/sec).")
    print(f"\nReference date used for all recency features: {ref_date}")
    print("IMPORTANT: reuse this exact reference date (pass --reference-date "
          f"{ref_date} explicitly) if you ever rebuild features later or "
          "build inference-time features for rank.py, so recency features "
          "never drift between training and submission.")


if __name__ == "__main__":
    main()