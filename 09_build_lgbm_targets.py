#!/usr/bin/env python3
"""
09_build_lgbm_targets.py
========================

Purpose
-------
Build the LightGBM training targets for the ranked candidate pool by turning
the LLM pointwise scorer's rank output into graded relevance labels. This file
is part of the LightGBM ranking pipeline and is meant to be run after the LLM
pointwise ranking step has produced a CSV with candidate ranks.

What this script does
---------------------
1. Reads the ranked candidate CSV produced by the pointwise scorer.
2. Converts each candidate's rank into a discrete relevance grade.
3. Optionally applies a deterministic honeypot override so flagged candidates
   are forced to relevance_grade=0.
4. Optionally limits the labeled set with --max-rank.
5. Writes the final target CSV used to train LightGBM.
6. Optionally produces a pseudo-label file for the full candidate pool when
   --all-features-csv and --pseudo-out are both supplied.

Why rank is used instead of the raw LLM score
---------------------------------------------
The rank is the signal this pipeline trusts. The raw pointwise score is not used
as a feature, and it is intentionally not treated as the target here either.
Using rank keeps the target stable even if the scorer's absolute numeric scale
shifts slightly between runs. LightGBM's lambdarank objective also expects
small integer relevance grades, not floating-point scores.

How the grading scheme works
----------------------------
The rank is mapped into compact relevance buckets that emphasize the top of the
list, which is what the hackathon evaluation rewards most heavily. The default
boundaries are:

    rank 1-5      -> grade 6
    rank 6-10     -> grade 5
    rank 11-25    -> grade 4
    rank 26-50    -> grade 3
    rank 51-150   -> grade 2
    rank 151-1000  -> grade 1
    rank 1001+    -> grade 0

These boundaries are intentionally tied to the evaluation cutoffs and do not
need to change just because the labeled set grows from 10k to 20k or any other
size. As the pool grows, grade 0 simply absorbs more low-ranked candidates.

Honeypot override
-----------------
If a candidate is flagged by the deterministic honeypot rules, this script
forces relevance_grade=0 for that candidate regardless of LLM rank. That keeps
the label generation aligned with the hard disqualification constraint rather
than the softer LLM judgment.

Docker / repository usage
-------------------------
This script already uses relative file paths only. For Docker compatibility,
run it from the repository root or the current working directory inside the
container. There is no hardcoded "Resume_Ranker" path in the runtime logic.

Usage
-----
Run from the repo root (or the container working directory):

    python "./09_build_lgbm_targets.py"         --final-ranked-csv outputs/final_ranked.csv         --max-rank 10000         --out outputs/lgbm_targets_10000.csv

Optional full pseudo-label output:
    Add both --all-features-csv and --pseudo-out.

Notes
-----
- --final-ranked-csv must contain candidate_id and a rank column
  (final_rank or rank).
- --honeypot-csv is optional.
- --max-rank lets you rebuild a smaller labeled set from the same ranked CSV.
- The output keeps the original candidate order sorted by rank.
"""

import argparse
import csv
from typing import Dict, Optional


GRADE_BOUNDARIES = [
    (5, 6),       # rank 1-5      -> grade 6
    (10, 5),      # rank 6-10     -> grade 5
    (25, 4),      # rank 11-25    -> grade 4
    (50, 3),      # rank 26-50    -> grade 3
    (150, 2),     # rank 51-150   -> grade 2
    (1000, 1),    # rank 151-1000 -> grade 1
]
DEFAULT_GRADE = 0  # everything beyond the last boundary, up to N


def rank_to_grade(rank: int) -> int:
    for max_rank, grade in GRADE_BOUNDARIES:
        if rank <= max_rank:
            return grade
    return DEFAULT_GRADE


def load_honeypot_overrides(path: Optional[str]) -> Dict[str, bool]:
    """Return candidate IDs flagged by deterministic honeypot rules."""
    if not path:
        return {}

    flag_aliases = ["honeypot_flag", "is_honeypot", "flagged"]
    hits_aliases = ["honeypot_rule_hits", "rule_hits", "num_rule_hits"]

    flagged: Dict[str, bool] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        flag_col = next((a for a in flag_aliases if a in fieldnames), None)
        hits_col = next((a for a in hits_aliases if a in fieldnames), None)

        if not flag_col and not hits_col:
            print(f"WARNING: {path} has neither a honeypot flag column nor "
                  f"a rule-hits column among {fieldnames}; no honeypot "
                  f"override will be applied.")
            return {}

        print(f"Honeypot override CSV column mapping -> "
              f"flag={flag_col}, rule_hits={hits_col}")

        for row in reader:
            cid = row.get("candidate_id")
            if not cid:
                continue
            is_flagged = False
            if flag_col:
                is_flagged = str(row.get(flag_col, "")).strip().lower() in (
                    "true", "1", "yes",
                )
            if not is_flagged and hits_col:
                try:
                    is_flagged = int(row.get(hits_col, 0) or 0) > 0
                except ValueError:
                    is_flagged = False
            if is_flagged:
                flagged[cid] = True

    print(f"Loaded {len(flagged)} honeypot-flagged candidate_ids from {path}")
    return flagged


def load_pointwise_ranks(path: str, max_rank: Optional[int]) -> Dict[str, int]:
    """Read ranked candidates and return {candidate_id: final_rank}."""
    rank_aliases = ["final_rank", "rank"]

    ranks: Dict[str, int] = {}
    n_dropped_by_max_rank = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rank_col = next((a for a in rank_aliases if a in fieldnames), None)
        if not rank_col:
            raise ValueError(
                f"Could not find a rank column in {path} (looked for "
                f"{rank_aliases} among {fieldnames})."
            )
        for row in reader:
            cid = row.get("candidate_id")
            if not cid:
                continue
            try:
                r = int(row[rank_col])
            except (ValueError, KeyError):
                continue
            if max_rank is not None and r > max_rank:
                n_dropped_by_max_rank += 1
                continue
            ranks[cid] = r

    if max_rank is not None:
        print(f"--max-rank {max_rank}: dropped {n_dropped_by_max_rank} "
              f"candidates ranked beyond {max_rank}.")
    return ranks


def load_all_candidate_ids(path: str) -> list:
    """Read candidate_id values from a full feature CSV in file order."""
    candidate_ids = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "candidate_id" not in fieldnames:
            raise ValueError(
                f"Could not find candidate_id column in {path} (columns: {fieldnames})."
            )
        for row in reader:
            cid = row.get("candidate_id")
            if cid:
                candidate_ids.append(cid)
    return candidate_ids


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--final-ranked-csv", required=True,
                     help="Output of 06_llm_pointwise_score.py -- the CSV "
                          "with a final_rank column for the candidates "
                          "the LLM scored (10,000, 20,000, or any depth).")
    ap.add_argument("--honeypot-csv", default=None,
                     help="Optional CSV with deterministic honeypot rule "
                          "output (honeypot_flag and/or honeypot_rule_hits "
                          "columns) to force relevance_grade=0 regardless "
                          "of LLM rank. Can be the same file you pass as "
                          "--honeypot-csv to 08_build_lgbm_features.py.")
    ap.add_argument("--max-rank", type=int, default=None,
                     help="Only label candidates with llm_rank <= this "
                          "value. This is the 'custom training set size' "
                          "knob: leave unset to use every row in "
                          "--pointwise-csv (e.g. all 20,000), or set it to "
                          "e.g. 10000 to build a smaller labeled set from "
                          "the same, deeper-scored pointwise CSV.")
    ap.add_argument("--out", required=True,
                     help="Output CSV path, e.g. outputs/lgbm_targets_20000.csv")
    ap.add_argument("--all-features-csv", default=None,
                     help="Optional full feature CSV (e.g. 100k rows). If provided together with --pseudo-out, the script also writes a full pseudo-label file where unlabeled candidates get relevance_grade=1.")
    ap.add_argument("--pseudo-out", default=None,
                     help="Optional output CSV path for the full pseudo-label file produced from --all-features-csv.")
    args = ap.parse_args()

    ranks = load_pointwise_ranks(args.final_ranked_csv, args.max_rank)
    print(f"Loaded {len(ranks)} candidate ranks from {args.final_ranked_csv}"
          + (f" (capped at --max-rank {args.max_rank})" if args.max_rank else ""))

    overrides = load_honeypot_overrides(args.honeypot_csv)

    n_overridden = 0
    max_grade_seen = max((g for _, g in GRADE_BOUNDARIES), default=0)
    grade_counts: Dict[int, int] = {g: 0 for g in range(max_grade_seen + 1)}

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_id", "llm_rank", "relevance_grade",
            "honeypot_override_applied",
        ])
        for cid, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
            grade = rank_to_grade(rank)
            overridden = False
            if overrides.get(cid):
                if grade != 0:
                    n_overridden += 1
                grade = 0
                overridden = True
            grade_counts[grade] += 1
            writer.writerow([cid, rank, grade, int(overridden)])

    n_labeled = len(ranks)
    print(f"\nWrote {n_labeled} target rows to {args.out}")

    if args.all_features_csv and args.pseudo_out:
        all_candidate_ids = load_all_candidate_ids(args.all_features_csv)
        ranked_ids = set(ranks.keys())
        n_pseudo_written = 0

        with open(args.pseudo_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "candidate_id", "llm_rank", "relevance_grade",
                "honeypot_override_applied",
            ])

            for cid, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
                grade = rank_to_grade(rank)
                overridden = False
                if overrides.get(cid):
                    grade = 0
                    overridden = True
                writer.writerow([cid, rank, grade, int(overridden)])
                n_pseudo_written += 1

            for cid in all_candidate_ids:
                if cid in ranked_ids:
                    continue
                overridden = False
                grade = 1
                if overrides.get(cid):
                    grade = 0
                    overridden = True
                writer.writerow([cid, "", grade, int(overridden)])
                n_pseudo_written += 1

        print(
            f"Wrote {n_pseudo_written} full pseudo-label rows to {args.pseudo_out} "
            f"(ranked candidates kept as-is, unlabeled candidates assigned grade=1)."
        )
    elif args.all_features_csv or args.pseudo_out:
        print(
            "WARNING: to write the full pseudo-label file, pass BOTH --all-features-csv "
            "and --pseudo-out. Skipping pseudo-label output."
        )

    if overrides:
        print(f"Honeypot override forced grade=0 for {n_overridden} "
              f"candidates that the LLM had ranked above grade 0.")
    print("\nGrade distribution (grade: count, % of labeled set):")
    for g in range(max_grade_seen, -1, -1):
        pct = 100.0 * grade_counts[g] / n_labeled if n_labeled else 0.0
        bar = "#" * max(1, grade_counts[g] * 50 // max(n_labeled, 1))
        print(f"  grade {g}: {grade_counts[g]:6d}  ({pct:5.1f}%)  {bar}")
    print(
        "\nReminder: grade 0 will dominate by design, and its SHARE grows "
        "as your labeled-set size N grows (more candidates ranked beyond "
        "rank 1000) -- that's expected, not a bug. The model still spends "
        "its learning capacity on contrasts near the top of the list, "
        "which is where NDCG@10/NDCG@50 actually score you. Tune "
        "GRADE_BOUNDARIES above after your first training run if the "
        "grade-6/5/4 buckets look too thin or too crowded for LightGBM to "
        "learn from."
    )


if __name__ == "__main__":
    main()