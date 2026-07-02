#!/usr/bin/env python3
"""10_assemble_lgbm_training_set.py
=================================
Assemble the LightGBM learning-to-rank training data by combining:

1) the full-pool resume features from 08_build_lgbm_features.py, and
2) the labeled relevance targets from 09_build_lgbm_targets.py.

The output is a train/validation split that LightGBM's lambdarank
objective can consume directly, plus a metadata JSON file that captures
the feature columns, categorical columns, label gain mapping, split
sizes, and grade histograms.

Docker / path note
------------------
This script uses only the file paths passed through the CLI arguments.
There is no hardcoded Resume_Ranker path inside the code, so the safest
Docker setup is to run it from the repo root (the current working
directory inside the container) and pass relative paths such as:

    python "10_assemble_lgbm_training_set.py" \
        --features outputs/lgbm_features_100k.csv \
        --targets outputs/lgbm_targets_10000.csv \
        --out-dir outputs/lgbm_dataset_10000

What it writes
--------------
  <out-dir>/train_features.csv        feature columns only for train rows
  <out-dir>/train_labels.csv          candidate_id + relevance_grade
  <out-dir>/val_features.csv          feature columns only for val rows
  <out-dir>/val_labels.csv            candidate_id + relevance_grade
  <out-dir>/dataset_metadata.json     feature lists, gains, group sizes,
                                      split seed, and grade histograms

Key behavior
------------
- The script does not care whether 09_build_lgbm_targets.py produced
  10,000, 20,000, or any other labeled-set size.
- Use --max-rank here if you want to shrink the labeled set without
  regenerating the targets file.
- The train/val split is stratified by relevance grade so that the rare
  high-relevance rows are represented in both splits.
- Because this hackathon setup is a single job-description ranking task,
  the split is a practical document-level holdout rather than a
  query-level holdout.

USAGE
-----
    python "10_assemble_lgbm_training_set.py" \
        --features outputs/lgbm_features_100k.csv \
        --targets outputs/lgbm_targets_10000.csv \
        --out-dir outputs/lgbm_dataset_10000
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List

from feature_engineering import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


def load_features(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = {row["candidate_id"]: row for row in reader}
    print(f"Loaded {len(rows)} feature rows from {path}")
    return rows


def load_targets(path: str, max_rank: int = None) -> Dict[str, int]:
    n_dropped = 0
    rows: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_rank_col = "llm_rank" in (reader.fieldnames or [])
        if max_rank is not None and not has_rank_col:
            raise ValueError(
                f"--max-rank was given but {path} has no llm_rank column "
                f"to filter on (columns found: {reader.fieldnames})."
            )
        for row in reader:
            if max_rank is not None and int(row["llm_rank"]) > max_rank:
                n_dropped += 1
                continue
            rows[row["candidate_id"]] = int(row["relevance_grade"])
    print(f"Loaded {len(rows)} target rows from {path}"
          + (f" (--max-rank {max_rank} dropped {n_dropped})" if max_rank else ""))
    return rows


def stratified_split(
    ids_by_grade: Dict[int, List[str]], val_fraction: float, seed: int
) -> (List[str], List[str]):
    rng = random.Random(seed)
    train_ids, val_ids = [], []
    for grade, ids in ids_by_grade.items():
        ids = list(ids)
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_fraction)) if len(ids) > 1 else 0
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids


def write_split(
    out_dir: str, split_name: str, ids: List[str],
    features: Dict[str, Dict[str, str]], targets: Dict[str, int],
    feature_cols: List[str],
):
    feat_path = os.path.join(out_dir, f"{split_name}_features.csv")
    label_path = os.path.join(out_dir, f"{split_name}_labels.csv")

    with open(feat_path, "w", newline="", encoding="utf-8") as ff, \
         open(label_path, "w", newline="", encoding="utf-8") as lf:
        fwriter = csv.DictWriter(ff, fieldnames=["candidate_id"] + feature_cols)
        fwriter.writeheader()
        lwriter = csv.writer(lf)
        lwriter.writerow(["candidate_id", "relevance_grade"])

        for cid in ids:
            frow = features[cid]
            out_row = {"candidate_id": cid}
            for col in feature_cols:
                out_row[col] = frow.get(col, "")
            fwriter.writerow(out_row)
            lwriter.writerow([cid, targets[cid]])

    print(f"  Wrote {len(ids)} rows -> {feat_path} / {label_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features", required=True,
                     help="Output of 08_build_lgbm_features.py (full pool).")
    ap.add_argument("--targets", required=True,
                     help="Output of 09_build_lgbm_targets.py (labeled rows).")
    ap.add_argument("--out-dir", required=True,
                     help="Directory to write train/val files + metadata into.")
    ap.add_argument("--max-rank", type=int, default=None,
                     help="Custom training-set-size knob: only use labeled "
                          "candidates with llm_rank <= this value, even if "
                          "--targets contains a deeper-scored set. Leave "
                          "unset to use everything in --targets.")
    ap.add_argument("--val-fraction", type=float, default=0.15,
                     help="Fraction of the labeled candidates held out for "
                          "validation, stratified by relevance grade. "
                          "Default 0.15.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    features = load_features(args.features)
    targets = load_targets(args.targets, max_rank=args.max_rank)

    missing = [cid for cid in targets if cid not in features]
    if missing:
        print(f"WARNING: {len(missing)} candidate_ids have a target label "
              f"but no row in the feature file -- they will be dropped. "
              f"Example: {missing[:5]}")
    usable_ids = [cid for cid in targets if cid in features]
    print(f"{len(usable_ids)}/{len(targets)} labeled candidates have a "
          f"matching feature row and will be used for training.")

    ids_by_grade: Dict[int, List[str]] = defaultdict(list)
    for cid in usable_ids:
        ids_by_grade[targets[cid]].append(cid)

    train_ids, val_ids = stratified_split(ids_by_grade, args.val_fraction, args.seed)
    print(f"\nSplit: {len(train_ids)} train / {len(val_ids)} val "
          f"(val_fraction={args.val_fraction}, seed={args.seed})")

    feature_cols = FEATURE_COLUMNS

    print("\nWriting train split ...")
    write_split(args.out_dir, "train", train_ids, features, targets, feature_cols)
    print("Writing val split ...")
    write_split(args.out_dir, "val", val_ids, features, targets, feature_cols)

    def grade_hist(ids):
        h = defaultdict(int)
        for cid in ids:
            h[targets[cid]] += 1
        return dict(sorted(h.items()))

    metadata = {
        "feature_columns": feature_cols,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "label_gain": [2 ** g - 1 for g in range(7)],
        "num_relevance_grades": 7,
        "labeled_set_size": len(usable_ids),
        "max_rank_filter_applied": args.max_rank,
        "train_group_size": len(train_ids),
        "val_group_size": len(val_ids),
        "train_grade_histogram": grade_hist(train_ids),
        "val_grade_histogram": grade_hist(val_ids),
        "split_seed": args.seed,
        "val_fraction": args.val_fraction,
        "notes": (
            "Single-query (single-JD) ranking setup: pass "
            "group=[train_group_size] and group=[val_group_size] to "
            "lgb.Dataset for the respective split. Pass categorical_columns "
            "through categorical_feature, and note that feature_columns "
            "contains only resume-derived columns."
        ),
    }
    meta_path = os.path.join(args.out_dir, "dataset_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nWrote dataset metadata -> {meta_path}")

    print("\nTrain grade histogram:", metadata["train_grade_histogram"])
    print("Val grade histogram:  ", metadata["val_grade_histogram"])
    print(
        "\nNext step (not part of this script): load these with "
        "lgb.Dataset(train_features[feature_columns].values, "
        "label=train_labels, group=[train_group_size], "
        "categorical_feature=categorical_columns) and train with "
        "objective='lambdarank', eval_at=[10, 50] against the val set."
    )


if __name__ == "__main__":
    main()