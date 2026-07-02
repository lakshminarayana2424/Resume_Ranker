#!/usr/bin/env python3
"""Step 6b — calibrate honeypot thresholds from the full candidate pool.

This is the empirical calibration step that runs before 07. It scans the entire
candidate pool once, measures how the dataset behaves for each ambiguous signal
that 07 may treat as a honeypot indicator, and writes a JSON report containing
summary statistics plus suggested threshold values.

Why this script exists:
07 needs numeric cutoffs for signals such as:
- the gap between stated years_of_experience and the span of the career history
- a single role that appears longer than the candidate's total claimed experience
- expert skills with zero recorded duration
- invalid education date ranges
- unusually long overlapping full-time roles

Those cutoffs should come from the real population, not from hand-tuned guesses.
This script treats the full pool as the baseline for "normal" behavior and
pushes thresholds into the extreme tail, so ordinary candidates are not flagged
just because they sit in a common part of the distribution.

What it computes:
- reference_date: inferred automatically from the latest parseable ISO date in
  the dataset
- stated_years_of_experience distribution
- career_history entry-count distribution
- positive-side gap distribution for stated experience vs. career span
- positive-side excess distribution for a single longest role vs. stated experience
- zero-duration expert-skill counts
- invalid education-date counts
- overlap durations across career-history pairs
- a recommended_07_thresholds section translated from those distributions

Output:
Writes a JSON file with the summary statistics and recommended thresholds.
The JSON is meant to be consumed by 07 as CLI threshold input, not used as a
standalone honeypot detector.

Docker note:
The default output path is resolved relative to this script's location so the
same command works from the repo root, the parent directory, or inside Docker.

Usage:
    python 06b_honeypot_population_stats.py \
        --candidates candidates.jsonl \
        --reference-date 2025-05-27 \
        --out outputs/honeypot_population_stats.json

Runtime:
Pure Python, one pass over the full pool, no model calls, no GPU.
"""

import argparse
import gzip
import json
import math
import os
from datetime import date
from typing import Dict, List, Optional

# ----------------------------------------------------------------------------
# I/O (same iterator contract as 04/06/07 for consistency)
# ----------------------------------------------------------------------------
def iter_candidates(path: str):
    if path.endswith(".gz"):
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    elif path.endswith(".jsonl"):
        opener = lambda: open(path, "r", encoding="utf-8")
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data:
            yield c
        return
    with opener() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_date_safe(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return None


def months_between(d1: Optional[date], d2: Optional[date]) -> Optional[int]:
    if d1 is None or d2 is None:
        return None
    months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
    if d2.day < d1.day:
        months -= 1
    return max(months, 0)


def iter_string_values(obj):
    """Recursively yield all string values from a nested candidate object."""
    if isinstance(obj, dict):
        for value in obj.values():
            yield from iter_string_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_string_values(value)
    elif isinstance(obj, tuple):
        for value in obj:
            yield from iter_string_values(value)
    elif isinstance(obj, str):
        yield obj


def infer_reference_date(candidates_path: str) -> date:
    """Infer a fixed dataset snapshot date from the latest parseable date in the pool."""
    latest: Optional[date] = None
    for c in iter_candidates(candidates_path):
        for s in iter_string_values(c):
            d = parse_date_safe(s)
            if d is not None and (latest is None or d > latest):
                latest = d
    if latest is None:
        raise SystemExit(
            f"Could not infer a reference date from {candidates_path!r}: "
            "no parseable ISO dates were found anywhere in the dataset. "
            "Provide a dataset with explicit dates or add a fixed snapshot date."
        )
    return latest


# ----------------------------------------------------------------------------
# Stats helpers -- no numpy dependency required, kept deliberately simple
# and auditable since these numbers drive the rubric thresholds.
# ----------------------------------------------------------------------------
PERCENTILES = [50, 90, 95, 99, 99.5, 99.8, 99.9, 99.95, 100]


def summarize(values: List[float]) -> Dict:
    if not values:
        return {"count": 0}
    xs = sorted(values)
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n if n > 1 else 0.0
    std = math.sqrt(var)

    def pct(p):
        if n == 1:
            return xs[0]
        idx = (p / 100) * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        frac = idx - lo
        return xs[lo] + (xs[hi] - xs[lo]) * frac

    out = {
        "count": n,
        "mean": round(mean, 3),
        "std": round(std, 3),
        "min": round(xs[0], 3),
    }
    for p in PERCENTILES:
        out[f"p{p}"] = round(pct(p), 3)
    return out


# ----------------------------------------------------------------------------
# Per-candidate metric extraction -- mirrors compute_precomputed_analysis()
# in 07_llm_honeypot_check.py exactly, so the population stats are computed
# on the SAME definitions 07 will apply per-candidate. Keeping these two
# implementations in lockstep matters: if they drift, the calibrated
# thresholds stop being valid for what 07 actually measures.
# ----------------------------------------------------------------------------
def extract_metrics(c: dict, reference_date: date) -> dict:
    p = c.get("profile", {}) or {}
    history = c.get("career_history", []) or []
    education = c.get("education", []) or []
    skills = c.get("skills", []) or []

    stated_years = p.get("years_of_experience")

    total_months = 0
    max_months = -1
    starts, ends, parsed_roles = [], [], []
    for h in history:
        dm = h.get("duration_months")
        if isinstance(dm, (int, float)):
            total_months += dm
            if dm > max_months:
                max_months = dm
        sd = parse_date_safe(h.get("start_date"))
        if h.get("end_date"):
            ed = parse_date_safe(h.get("end_date"))
        elif h.get("is_current"):
            ed = reference_date
        else:
            ed = None
        if sd:
            starts.append(sd)
        if ed:
            ends.append(ed)
        parsed_roles.append({"start": sd, "end": ed})

    earliest_start = min(starts) if starts else None
    latest_end = max(ends) if ends else None
    span_months = (
        months_between(earliest_start, latest_end)
        if earliest_start and latest_end else None
    )
    span_years = span_months / 12 if span_months is not None else None

    gap_years = None
    if isinstance(stated_years, (int, float)) and span_years is not None:
        gap_years = stated_years - span_years

    single_role_years = max_months / 12 if max_months >= 0 else None
    single_role_excess_years = None
    if single_role_years is not None and isinstance(stated_years, (int, float)):
        single_role_excess_years = single_role_years - stated_years

    zero_dur_expert_count = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months") == 0
    )

    invalid_edu_count = sum(
        1 for e in education
        if isinstance(e.get("start_year"), int) and isinstance(e.get("end_year"), int)
        and e["end_year"] < e["start_year"]
    )

    overlap_months_list = []
    for i in range(len(parsed_roles)):
        for j in range(i + 1, len(parsed_roles)):
            a, b = parsed_roles[i], parsed_roles[j]
            if a["start"] and a["end"] and b["start"] and b["end"]:
                latest_start = max(a["start"], b["start"])
                earliest_end = min(a["end"], b["end"])
                if latest_start < earliest_end:
                    ov = months_between(latest_start, earliest_end)
                    if ov:
                        overlap_months_list.append(ov)

    return {
        "stated_years": stated_years,
        "span_years": span_years,
        "gap_years": gap_years,
        "single_role_years": single_role_years,
        "single_role_excess_years": single_role_excess_years,
        "zero_dur_expert_count": zero_dur_expert_count,
        "invalid_edu_count": invalid_edu_count,
        "overlap_months_list": overlap_months_list,
        "career_history_entries": len(history),
    }


# ----------------------------------------------------------------------------
# Recommended-threshold logic
# ----------------------------------------------------------------------------
def recommend_thresholds(agg: dict) -> dict:
    """
    Translate population percentiles into 07-ready CLI thresholds.

    Rule of thumb used throughout: pick the threshold at the percentile
    just above the honeypot base rate (~0.08% of the pool, i.e. ~p99.92).
    We use p99.9 as a slightly more conservative (higher) cut than the
    exact base rate, since the population still contains some legitimate
    high-variance candidates near the tail (long pre-listed-history
    careers, self-report rounding, etc.) and we want the FLOOR of "this is
    weird enough to investigate" to sit safely above ordinary tail noise,
    not exactly at the honeypot count. Round outward (more permissive) to
    the nearest sane unit so the threshold is human-defensible in a
    Stage-5 interview ("why 4.5 years and not 4.37") rather than a raw
    percentile readout.
    """
    rec = {}

    gap = agg["gap_years_dist"]
    if gap.get("count"):
        # gap_years > 0 means claimed more experience than career_history
        # geometrically supports. Use the p99.9 of the POSITIVE side only
        # (negative/zero gap is not a violation direction at all).
        raw = gap.get("p99.9", gap.get("p99", 3.0))
        rec["span_gap_tolerance_years"] = max(2.0, round(math.ceil(raw * 2) / 2, 1))

    excess = agg["single_role_excess_years_dist"]
    if excess.get("count"):
        raw = excess.get("p99.9", excess.get("p99", 1.0))
        rec["single_role_tolerance_years"] = max(1.0, round(math.ceil(raw * 2) / 2, 1))

    ov = agg["overlap_months_dist"]
    if ov.get("count"):
        raw = ov.get("p95", 3)
        rec["min_overlap_months"] = max(2, int(math.ceil(raw)))

    zd = agg["zero_dur_expert_count_dist"]
    if zd.get("count"):
        # If most candidates with ANY zero-duration-expert skill have
        # exactly 1, a count-based threshold (>=2) is far safer than a
        # binary "any" trigger -- single mislabels are common synthetic
        # noise; MULTIPLE zero-duration "expert" skills together is the
        # actual implausibility signal.
        rec["zero_duration_expert_min_count"] = 2 if zd.get("p99", 0) >= 1 else 1

    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True,
                     help="Full candidates.jsonl / .jsonl.gz pool (all 100,000 rows).")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "honeypot_population_stats.json"))
    args = ap.parse_args()

    reference_date = infer_reference_date(args.candidates)
    print(f"Auto-detected reference date: {reference_date.isoformat()}")

    gap_years, single_role_excess_years, overlap_months_all = [], [], []
    zero_dur_counts, invalid_edu_counts = [], []
    stated_years_all, history_entry_counts = [], []

    n_total = 0
    n_with_span = 0
    n_with_any_zero_dur_expert = 0

    print(f"Scanning {args.candidates} ...")
    for c in iter_candidates(args.candidates):
        n_total += 1
        m = extract_metrics(c, reference_date)

        if isinstance(m["stated_years"], (int, float)):
            stated_years_all.append(float(m["stated_years"]))
        history_entry_counts.append(m["career_history_entries"])

        if m["gap_years"] is not None:
            n_with_span += 1
            if m["gap_years"] > 0:  # only the violation direction
                gap_years.append(m["gap_years"])

        if m["single_role_excess_years"] is not None and m["single_role_excess_years"] > 0:
            single_role_excess_years.append(m["single_role_excess_years"])

        zero_dur_counts.append(m["zero_dur_expert_count"])
        if m["zero_dur_expert_count"] > 0:
            n_with_any_zero_dur_expert += 1

        invalid_edu_counts.append(m["invalid_edu_count"])
        overlap_months_all.extend(m["overlap_months_list"])

        if n_total % 20000 == 0:
            print(f"  ... {n_total} scanned")

    print(f"Done. {n_total} candidates scanned.")

    agg = {
        "reference_date": reference_date.isoformat(),
        "reference_date_source": "auto-detected from the latest parseable ISO date in the dataset",
        "n_total_candidates": n_total,
        "n_with_parseable_career_span": n_with_span,
        "n_with_any_zero_duration_expert_skill": n_with_any_zero_dur_expert,
        "pct_with_any_zero_duration_expert_skill": round(
            100 * n_with_any_zero_dur_expert / n_total, 4
        ) if n_total else 0,
        "stated_years_of_experience_dist": summarize(stated_years_all),
        "career_history_entry_count_dist": summarize([float(x) for x in history_entry_counts]),
        "gap_years_dist": summarize(gap_years),  # positive-direction only
        "single_role_excess_years_dist": summarize(single_role_excess_years),  # positive-direction only
        "overlap_months_dist": summarize([float(x) for x in overlap_months_all]),
        "zero_dur_expert_count_dist": summarize([float(x) for x in zero_dur_counts]),
        "invalid_edu_count_dist": summarize([float(x) for x in invalid_edu_counts]),
    }
    agg["recommended_07_thresholds"] = recommend_thresholds(agg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print(f"\nWrote {args.out}")
    print("\nRecommended 07 CLI thresholds (also in the JSON under "
          "'recommended_07_thresholds'):")
    for k, v in agg["recommended_07_thresholds"].items():
        print(f"  --{k.replace('_', '-')}  {v}")

    print(
        "\nSanity checks:"
        f"\n  invalid_edu_count_dist.p99.9 should be 0 or near-0 "
        f"(true logical impossibility, expected rare): "
        f"{agg['invalid_edu_count_dist'].get('p99.9')}"
        f"\n  pct of pool with ANY zero-duration expert skill: "
        f"{agg['pct_with_any_zero_duration_expert_skill']}% "
        f"(if this is >> 0.08%, a binary 'any' trigger would false-positive "
        f"heavily -- use the count-based recommendation instead)"
    )


if __name__ == "__main__":
    main()