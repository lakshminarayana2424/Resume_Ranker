#!/usr/bin/env python3
"""
04_honeypot_rules.py — deterministic honeypot detector.

Scans each candidate record and applies only hard, deterministic rules:
date consistency, impossible employment timelines, cross-field
contradictions, and a small schema sanity check. The output is a CSV of
candidate_id, honeypot_flag, hit_count, and rule reasons.

Usage:
    python 04_honeypot_rules.py --candidates candidates.jsonl --out outputs/honeypot_flags.csv
    python 04_honeypot_rules.py --candidates sample_candidates.json --out outputs/honeypot_flags.csv
"""
import argparse
import csv
import gzip
import json
import os
import re
from collections import Counter
from datetime import date, datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Absolute ceiling from the schema: years_of_experience max = 50, so skill duration is checked against a fixed human-career bound.
SCHEMA_MAX_EXPERIENCE_MONTHS = 50 * 12

FOUNDING_PATTERNS = [
    re.compile(r"founded(?: the company)? in (\d{4})", re.I),
    re.compile(r"since (?:its |the company'?s )?founding in (\d{4})", re.I),
    re.compile(r"established in (\d{4})", re.I),
    re.compile(r"launched in (\d{4})", re.I),
    re.compile(r"company was founded in (\d{4})", re.I),
    re.compile(r"co-founded in (\d{4})", re.I),
    re.compile(r"started the company in (\d{4})", re.I),
]


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def month_diff(d1, d2):
    """Whole months between two dates (d2 - d1)."""
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def iter_candidates(path):
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


def find_reference_now(path):
    """First pass: find the latest date anywhere in the dataset so open-ended roles can be bounded during overlap checks."""
    latest = date(1970, 1, 1)
    for c in iter_candidates(path):
        sig = c.get("redrob_signals", {}) or {}
        for key in ("signup_date", "last_active_date"):
            d = parse_date(sig.get(key))
            if d and d > latest:
                latest = d
        for h in c.get("career_history", []) or []:
            for key in ("start_date", "end_date"):
                d = parse_date(h.get(key))
                if d and d > latest:
                    latest = d
    return latest


def check_candidate(c, reference_now):
    """Returns hard reasons only. One hit is enough to flag the candidate."""
    hard = []
    p = c.get("profile", {}) or {}
    sig = c.get("redrob_signals", {}) or {}
    history = c.get("career_history", []) or []
    edu = c.get("education", []) or []
    skills = c.get("skills", []) or []

    yoe = p.get("years_of_experience")
    yoe_months = yoe * 12 if isinstance(yoe, (int, float)) else None

    # Temporal consistency and timeline sanity checks.
    current_count = 0
    intervals = []
    total_history_months = 0

    for h in history:
        sd, ed = parse_date(h.get("start_date")), parse_date(h.get("end_date"))
        is_current = h.get("is_current")
        dur = h.get("duration_months")

        if is_current:
            current_count += 1
            if ed is not None:
                hard.append("current_role_has_end_date")

        if sd and ed:
            if ed < sd:
                hard.append("end_date_before_start_date")
            elif isinstance(dur, (int, float)):
                expected = month_diff(sd, ed)
                if abs(expected - dur) > 2:
                    hard.append("duration_months_inconsistent_with_dates")

        # Only use a real end date for overlap checks; missing past end dates stay missing.
        end_for_overlap = reference_now if is_current else ed
        if sd and end_for_overlap:
            intervals.append((sd, end_for_overlap, h.get("company")))

        if isinstance(dur, (int, float)):
            total_history_months += dur

        desc = h.get("description") or ""
        for pat in FOUNDING_PATTERNS:
            m = pat.search(desc)
            if m:
                founding_year = int(m.group(1))
                if sd and sd.year < founding_year:
                    hard.append("career_predates_company_founding")
                break

    if current_count > 1:
        hard.append("multiple_current_roles")

    intervals.sort(key=lambda x: x[0])
    for (s1, e1, c1), (s2, e2, c2) in zip(intervals, intervals[1:]):
        if c1 != c2 and s2 < e1 and (e1 - s2).days > 60:
            hard.append("overlapping_employment_dates")

    for e in edu:
        sy, ey = e.get("start_year"), e.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and ey < sy:
            hard.append("education_end_before_start")

    # Cross-field contradictions.
    if yoe_months is not None and total_history_months > 0:
        if total_history_months > yoe_months * 1.4 + 12:
            hard.append("career_history_exceeds_stated_experience")

    if yoe_months is not None:
        for h in history:
            dur = h.get("duration_months")
            if isinstance(dur, (int, float)) and dur > yoe_months + 12:
                hard.append("single_role_exceeds_stated_experience")
                break

    for s in skills:
        prof = (s.get("proficiency") or "").lower()
        dur = s.get("duration_months")

        if prof == "expert" and isinstance(dur, (int, float)) and dur <= 1:
            hard.append("expert_skill_zero_duration")

        # Absolute ceiling from the schema's own definition of a human
        # career (years_of_experience max = 50), not the candidate's own
        # self-reported YOE -- see REPLACED note in the module docstring.
        if isinstance(dur, (int, float)) and dur > SCHEMA_MAX_EXPERIENCE_MONTHS:
            hard.append("skill_duration_exceeds_schema_max_experience")

    headline = (p.get("headline") or "").strip()
    summary = (p.get("summary") or "").strip()
    completeness = sig.get("profile_completeness_score")
    if isinstance(completeness, (int, float)) and completeness >= 95 and (not headline or len(summary) < 15):
        hard.append("high_completeness_but_empty_core_fields")

    return hard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "outputs", "honeypot_flags.csv"))
    args = ap.parse_args()

    print("Pass 1/2: scanning for the dataset reference date...")
    reference_now = find_reference_now(args.candidates)
    print(f"  reference 'now' = {reference_now.isoformat()}")

    print("Pass 2/2: applying hard rules to every candidate...")
    rows = []
    rule_counter = Counter()
    n = 0
    for c in iter_candidates(args.candidates):
        cid = c.get("candidate_id")
        if not cid:
            continue
        hard = check_candidate(c, reference_now)
        flagged = bool(hard)
        for r in hard:
            rule_counter[r] += 1
        rows.append((cid, flagged, len(hard), ";".join(hard)))
        n += 1
        if n % 20000 == 0:
            print(f"  checked {n} candidates...")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "honeypot_flag", "hit_count", "reasons"])
        for row in rows:
            w.writerow(row)

    flagged_count = sum(1 for r in rows if r[1])
    print(f"\nChecked {n} candidates. Flagged {flagged_count} as honeypots "
          f"({(flagged_count / n * 100) if n else 0:.2f}% of pool).")
    print("Hits per rule (a candidate can hit more than one):")
    for rule, count in rule_counter.most_common():
        print(f"  {rule}: {count}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()