#!/usr/bin/env python3
"""
05_apply_honeypot_to_ranking.py

Combines the cross-encoder ranking with the honeypot flags and moves every
flagged candidate below every clean candidate. The score order is preserved
inside each group, so this remains a deterministic post-processing step
before later training or submission stages.

Usage:
    python 05_apply_honeypot_to_ranking.py \
        --ranked outputs/cross_encoder_ranked.csv \
        --honeypots outputs/honeypot_flags.csv \
        --out outputs/cross_encoder_ranked_honeypot.csv
"""
import argparse
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_ranked(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "candidate_id": row["candidate_id"],
                "cross_encoder_score": float(row["cross_encoder_score"]),
            })
    return rows


def load_honeypot_flags(path):
    flags, reasons = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flags[row["candidate_id"]] = row["honeypot_flag"].strip().lower() == "true"
            reasons[row["candidate_id"]] = row.get("reasons", "")
    return flags, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked", default=os.path.join(BASE_DIR, "outputs", "cross_encoder_ranked.csv"))
    ap.add_argument("--honeypots", default=os.path.join(BASE_DIR, "outputs", "honeypot_flags.csv"))
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "outputs", "cross_encoder_ranked_honeypot.csv"))
    args = ap.parse_args()

    ranked = load_ranked(args.ranked)
    flags, reasons = load_honeypot_flags(args.honeypots)

    missing = [r["candidate_id"] for r in ranked if r["candidate_id"] not in flags]
    if missing:
        print(f"Warning: {len(missing)} ranked candidates have no honeypot flag "
              f"(treating as not-flagged). Example: {missing[:5]}")

    clean = [r for r in ranked if not flags.get(r["candidate_id"], False)]
    flagged = [r for r in ranked if flags.get(r["candidate_id"], False)]
    ordered = clean + flagged

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "cross_encoder_score", "honeypot_flag", "honeypot_reasons"])
        for i, r in enumerate(ordered, start=1):
            cid = r["candidate_id"]
            writer.writerow([
                cid, i, f"{r['cross_encoder_score']:.6f}",
                flags.get(cid, False),
                reasons.get(cid, ""),
            ])

    print(f"{len(clean)} clean candidates ranked 1-{len(clean)}.")
    print(f"{len(flagged)} honeypot candidates pushed to ranks "
          f"{len(clean) + 1}-{len(clean) + len(flagged)}.")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()