#!/usr/bin/env python3
"""
Step 3 — merge per-GPU cross-encoder score files into one ranked CSV.

This script:
- reads every matching scores_*.csv file,
- drops duplicate candidate_id rows,
- sorts by cross-encoder score descending with candidate_id as the tie-break,
- assigns a dense rank from 1..N.

The output is the cross-encoder stage result that feeds the later hackathon
pipeline stages.

Usage:
    python 03_merge_results.py --scores-glob "outputs/scores_*.csv" --out outputs/cross_encoder_ranked.csv
"""
import argparse
import csv
import glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-glob", default="outputs/scores_*.csv")
    ap.add_argument("--out", default="outputs/cross_encoder_ranked.csv")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(args.scores_glob)):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) == 2:
                    rows.append((row[0], float(row[1])))

    seen = set()
    deduped = []
    dupes = 0
    for cid, score in rows:
        if cid in seen:
            dupes += 1
            continue
        seen.add(cid)
        deduped.append((cid, score))

    deduped.sort(key=lambda r: (-r[1], r[0]))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "cross_encoder_score"])
        for i, (cid, score) in enumerate(deduped, start=1):
            writer.writerow([cid, i, f"{score:.6f}"])

    print(f"Merged {len(rows)} rows ({dupes} duplicate candidate_ids dropped).")
    print(f"Wrote {len(deduped)} ranked candidates to {args.out}")


if __name__ == "__main__":
    main()