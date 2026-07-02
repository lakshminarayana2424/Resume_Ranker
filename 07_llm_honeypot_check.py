#!/usr/bin/env python3
"""07_llm_honeypot_check.py — LLM-based honeypot audit of the top-K candidates.

This is the final integrity-check stage of the ranking pipeline. It takes the
pairwise-reranked top slice and the full cross-encoder ranking, then walks
down the merged candidate pool until it has collected exactly the requested
number of clean candidates for the final top slice. Any newly discovered
honeypots are pushed to the bottom of the output pool, while candidates that
were never reached by the audit keep their original placeholder state.

Why this script exists
----------------------
The earlier stages already ranked the candidates, but they cannot reliably
separate a genuinely strong profile from a profile that is logically or
temporally impossible. This script is the LLM-based audit layer that catches
those subtle, checkable contradictions. It is designed to be conservative:
if the evidence is unclear, it prefers to leave a candidate unflagged rather
than create a false positive.

What it checks
--------------
The rubric focuses on five narrow classes of impossibility:
- career span larger than the stated experience by more than the calibrated
  tolerance,
- one role longer than the candidate’s total claimed experience by more than
  the calibrated tolerance,
- multiple expert skills recorded with zero duration,
- invalid education dates,
- overlapping full-time roles beyond the calibrated overlap threshold.

Anything that is only "suspicious" is not enough. The model is told to use the
precomputed arithmetic block and to flag only concrete contradictions that can
be tied to specific numbers or dates.

How the merged pool is handled
------------------------------
1. The pairwise-ranked file supplies ranks 1..30000 in its exact order.
2. The cross-encoder file supplies the remaining rows in its exact order.
3. The script audits candidates from the top until exactly --top-k clean
   candidates have been found.
4. Honeypots found along the way are moved to the very end of the output.
5. Candidates never reached by the audit keep their original flag state:
   pairwise-origin rows remain NULL, and tail rows keep the carried-over
   rule-based honeypot flag from the cross-encoder file.

Why this is resilient
---------------------
The script is built to be restart-safe and Docker-friendly. It keeps a JSONL
cache of every honeypot decision so interrupted runs can resume without
re-checking the same candidates. The LLM output is parsed strictly as JSON,
and the script rejects malformed responses rather than trusting them. The
precomputed analysis block is computed in Python before the LLM sees the
profile, which keeps the arithmetic consistent and avoids repeated date math
inside the prompt.

Docker note
-----------
The model path is resolved relative to this script’s location, so the same
command works from the repo root, the parent directory, or inside Docker.
The command below is the intended launch command for the hackathon pipeline.

Usage:
CUDA_VISIBLE_DEVICES=1,0 python 07_llm_honeypot_check.py \
  --point-wise-ranked outputs/llm_pointwise_top30000.csv \
  --cross-encoder-ranked outputs/cross_encoder_ranked_honeypot.csv \
  --population-stats outputs/honeypot_population_stats.json \
  --candidates candidates.jsonl \
  --out outputs/final_ranked.csv \
  --honeypot-cache outputs/llm_honeypot_checks.jsonl \
  --model-name ./models/Qwen3-8B-AWQ \
  --num-gpus 2 \
  --gpu-ids 1,0 \
  --top-k 100 \
  --batch-size 1 \
  --max-model-len 4736 \
  --max-tokens 300 \
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 1 \
  --swap-space 4 \
  --secondary-preload 5000 \
  --max-checks 5000 \
  --max-retries 2 \
  --enable-prefix-caching \
  --enforce-eager
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time
from collections import Counter
from datetime import date
from typing import Dict, List, Optional, Tuple

# Reduce allocator fragmentation on small 8 GB GPUs (same as 06).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

RUBRIC_VERSION = "honeypot_v3"  # v3 uses calibrated thresholds from 06b and intentionally invalidates older cache entries.

# Tunable false-positive thresholds. Set once from CLI args at the top of
# main() (sourced from 06b's recommended_07_thresholds where available) and
# read directly by compute_precomputed_analysis() / render_analysis_block()
# / build_honeypot_rubric() below. Centralizing them here (instead of
# hardcoding numbers inline) is what lets you tune calibration without
# hunting through the prompt text. The defaults below are placeholders only
# -- main() always overwrites them from --span-gap-tolerance-years etc.,
# which you should set from 06b's output, not these defaults.
SPAN_GAP_TOLERANCE_YEARS = 3.0   # CAREER_SPAN_EXCEEDS_EXPERIENCE: only flag
                                 # if stated experience exceeds career_span
                                 # by MORE than this many years.
SINGLE_ROLE_TOLERANCE_YEARS = 1.0  # SINGLE_ROLE_EXCEEDS_EXPERIENCE: only
                                 # flag if one role's duration exceeds
                                 # stated years_of_experience by more than
                                 # this many years.
MIN_OVERLAP_MONTHS = 3          # OVERLAPPING_FULL_TIME_ROLES: overlaps
                                 # shorter than this are filtered out before
                                 # the LLM ever sees them (handover overlaps
                                 # of 1-2 months are completely normal).
ZERO_DURATION_EXPERT_MIN_COUNT = 2  # ZERO_DURATION_EXPERT_SKILL: only flag
                                 # if AT LEAST this many skills are both
                                 # proficiency=="expert" and
                                 # duration_months==0 on the SAME profile.
                                 # A single such skill is common synthetic-
                                 # data noise, not a honeypot signal on its
                                 # own (see 06b's population sanity check).
_RUBRIC_TEXT: Optional[str] = None  # rendered once in main() from the
                                     # template below, using the four
                                     # thresholds above.

# ============================================================================
# THE RUBRIC
# ============================================================================
# v3 changes vs. v2 (see RUBRIC_VERSION note above for *why*):
#   1. All four ambiguous thresholds (span gap, single-role excess, min
#      overlap, zero-duration-expert count) are now sourced from
#      06b_honeypot_population_stats.py's empirical scan of the full
#      100,000-candidate pool (roughly the p99.9 cut of each signal's real
#      distribution) instead of hand-picked numbers. Same base-rate logic
#      v2 already used ("honeypots are ~0.1% of the pool"), just applied
#      numerically instead of by feel.
#   2. ZERO_DURATION_EXPERT_SKILL is now COUNT-based
#      (zero_duration_expert_min_count, typically 1-2) instead of "any
#      single occurrence" -- 06b's population scan showed a nonzero
#      fraction of real candidates have one zero-duration "expert" skill
#      as ordinary data noise; a binary trigger on that would have been a
#      guaranteed false-positive source.
#   3. SINGLE_ROLE_EXCEEDS_EXPERIENCE's tolerance is now an explicit named
#      constant + CLI arg (SINGLE_ROLE_TOLERANCE_YEARS) instead of vague
#      prose ("an explicit small tolerance too") with no enforced number.
HONEYPOT_RUBRIC_TEMPLATE = """You are auditing ONE candidate profile for the Redrob "Senior AI Engineer"
hiring pipeline. This is a SECOND-pass integrity check, not a fit/quality
judgment -- this candidate already survived an upstream retrieval model and
an upstream pointwise LLM scorer, both of which judged them a good fit.
Your only job here is to catch profiles that are LOGICALLY OR TEMPORALLY
IMPOSSIBLE -- i.e. profiles that could not describe a real human career,
as opposed to profiles that are merely weak, generic, or a poor fit.

THE HACKATHON'S OWN DEFINITION (this is what you are actually looking for)
The organizers describe the planted honeypots as candidates with "subtly
impossible profiles" -- profiles that look plausible at a skim but contain
a concrete logical or temporal contradiction once you check the numbers.
Their own illustrative examples: claiming years of experience that don't
fit inside a company's founding date, or claiming "expert" proficiency in
many skills with zero recorded time spent on any of them. They are
explicit that this is a representative sample, not an exhaustive list --
any comparably concrete, checkable impossibility counts, even if it
doesn't match one of the five named categories below. The thing that makes
something a honeypot is that it is CHECKABLE AND FALSE, not that it
"feels off" or "feels too good."

BASE RATE -- READ THIS, BUT DO NOT LET IT OVERRIDE A CONFIRMED NUMBER
Roughly 80 out of 100,000 candidates in this dataset (about 0.08%) are
honeypots. That means in any batch you check, the prior for any given
candidate is is_honeypot=false. This is useful for category 6 below
(open-ended judgment, where evidence is genuinely ambiguous) -- when you
have no concrete checkable contradiction, default to false. It is NOT a
license to wave off a category 1-5 check that has already come back TRUE
with a specific number attached. A confirmed 8-year gap against a 2-year
threshold is not "borderline, probably a real person" -- it is a 4x
overshoot of a threshold that was itself calibrated to be generous. Do not
talk yourself out of a TRUE deterministic check because the rest of the
profile looks senior, well-written, or impressive. Impossibility detection
and seniority/quality assessment are unrelated questions; a fabricated
profile can still read as polished.

WHAT THIS CHECK IS NOT FOR
Do NOT flag a candidate for any of the following -- none of these are
honeypot signals, all of them describe completely normal real candidates
in this dataset:
  - Weak, generic, or poorly-written summaries/descriptions
  - A skills list that doesn't match the JD well
  - A title that sounds junior or unrelated
  - Career gaps, short tenures, or frequent job-hopping (when they do NOT
    trip a deterministic check below)
  - Low platform engagement (inactive, low response rate, long notice period)
  - "This profile feels too good" or "this profile feels too generic" --
    vibes are not evidence
  - Anything you cannot tie to a specific cited number, either from a
    DETERMINISTIC CHECK in the PRECOMPUTED ANALYSIS, or (for category 6
    only) a specific date/number you point to directly in the candidate's
    own profile text

CATEGORIES 1-5: DETERMINISTIC -- PRECOMPUTED ANALYSIS DOES THE MATH
For these five, Python has ALREADY compared the relevant numbers against a
calibrated threshold and tells you the result as
"DETERMINISTIC CHECK: TRUE" or "DETERMINISTIC CHECK: FALSE" in the
PRECOMPUTED ANALYSIS block in the user message. Do not re-derive or
second-guess that arithmetic -- it is correct. Your job is simply:
  - DETERMINISTIC CHECK: FALSE -> do not flag this category.
  - DETERMINISTIC CHECK: TRUE -> flag this category, UNLESS the
    candidate's own career_history/skills text gives an EXPLICIT, SPECIFIC,
    NAMED reason that fully explains the discrepancy (e.g. the role
    description literally states "advisory role, 5 hrs/week" for an
    overlap, or "career break 2019-2021, raising a family" for a gap).
    A generic-sounding strong description is NOT an explicit override --
    if you cannot point to a specific sentence that explains the
    discrepancy, the TRUE check stands and you must flag it.

1. CAREER_SPAN_EXCEEDS_EXPERIENCE -- stated years_of_experience exceeds the
   career_history span by more than {gap_tol} years (the threshold was
   calibrated from this dataset's real population, generous enough to
   absorb unlisted early-career/freelance time).
2. SINGLE_ROLE_EXCEEDS_EXPERIENCE -- one role's duration exceeds total
   stated years_of_experience by more than {single_role_tol} years.
3. ZERO_DURATION_EXPERT_SKILL -- {zero_dur_min_count} or more skills are
   simultaneously proficiency="expert" and duration_months=0.
4. INVALID_EDUCATION_DATES -- an education entry's end_year is before its
   start_year. Zero tolerance.
5. OVERLAPPING_FULL_TIME_ROLES -- two full-time roles at different
   companies overlap by more than {overlap_min} months with no stated
   legitimate reason.

CATEGORY 6: OTHER_LOGICAL_IMPOSSIBILITY -- OPEN, FOR PATTERNS NOT ABOVE
This is where the base-rate caution applies most: only use this category
when you can point to a SPECIFIC, CONCRETE, CHECKABLE contradiction
directly in the profile's own dates/numbers -- the same standard as the
hackathon's own examples (e.g. a role at a company that, based on dates
given elsewhere in the profile or obviously implausible founding context,
could not have existed yet; a credential or certification dated before the
candidate could plausibly have qualified for it; an internally
contradictory pair of facts within the SAME profile, like two different
stated total-experience figures that don't match each other). Do not use
this category for anything you'd describe as "suspicious," "too perfect,"
or "unusual" without a specific number/date you can cite. If you're not
confident enough to name the exact contradiction in the evidence field,
the answer is false.

OUTPUT FORMAT
Respond with ONLY a JSON object, no other text:
{{
  "is_honeypot": true | false,
  "violation_category": one of
    ["CAREER_SPAN_EXCEEDS_EXPERIENCE", "SINGLE_ROLE_EXCEEDS_EXPERIENCE",
     "ZERO_DURATION_EXPERT_SKILL", "INVALID_EDUCATION_DATES",
     "OVERLAPPING_FULL_TIME_ROLES", "OTHER_LOGICAL_IMPOSSIBILITY"]
    if is_honeypot is true, else null,
  "evidence": a one-sentence citation of the SPECIFIC numbers/dates that
    justify the flag (e.g. "stated 16.2 years experience vs 8.17 years of
    career_history span, an 8.03 year gap, threshold 2.0 -- DETERMINISTIC
    CHECK TRUE, no explicit override text found"), or null if is_honeypot
    is false,
  "brief_rationale": one short sentence summarizing your reasoning either way
}}

If is_honeypot is true, violation_category and evidence are REQUIRED and
must reference an actual number/date -- a flag without a specific cited
number will be rejected and re-asked. If is_honeypot is false for a
candidate where a DETERMINISTIC CHECK above was TRUE, your evidence-less
output will also be rejected: you must still explain, in brief_rationale,
the specific override text you found that explains the TRUE check."""


def build_honeypot_rubric(
    span_gap_tolerance_years: float,
    single_role_tolerance_years: float,
    min_overlap_months: int,
    zero_duration_expert_min_count: int,
) -> str:
    """Renders the rubric template with the active thresholds baked in, so
    the prompt text the LLM reads always matches the numbers Python is
    actually using to filter/compute (see render_analysis_block)."""
    return HONEYPOT_RUBRIC_TEMPLATE.format(
        gap_tol=span_gap_tolerance_years,
        single_role_tol=single_role_tolerance_years,
        overlap_min=min_overlap_months,
        zero_dur_min_count=zero_duration_expert_min_count,
    )


# ============================================================================
# GPU setup -- copied verbatim from 06_llm_pointwise_score.py
# ============================================================================
def parse_gpu_ids(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    return ids


def configure_visible_gpus(gpu_ids: List[int]) -> None:
    """Set CUDA_VISIBLE_DEVICES once, before any vLLM/torch CUDA init."""
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")


# ============================================================================
# Candidate data loading and rendering -- copied from 06 (same single source
# of truth for how a profile is turned into text, so the LLM sees an
# identical rendering style to the scoring pass it already trusts).
# ============================================================================
def iter_candidates(path: str):
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


def load_candidate_profiles(pool_path: str, ids_needed: set) -> Dict[str, dict]:
    found = {}
    remaining = set(ids_needed)
    for c in iter_candidates(pool_path):
        cid = c.get("candidate_id")
        if cid in remaining:
            found[cid] = c
            remaining.discard(cid)
            if not remaining:
                break
    if remaining:
        print(
            f"WARNING: {len(remaining)} candidate_ids were not found in "
            f"{pool_path} -- they will be skipped. "
            f"Example missing ids: {list(remaining)[:5]}"
        )
    return found


def build_candidate_block(c: dict) -> str:
    """Full, readable rendering of one candidate profile for the LLM.
    Identical to 06_llm_pointwise_score.py's build_candidate_block so the
    LLM sees the same profile presentation it already does in the scoring
    pass."""
    p = c.get("profile", {}) or {}
    sig = c.get("redrob_signals", {}) or {}
    lines = []

    lines.append(f"Headline: {p.get('headline', '')}")
    lines.append(
        f"Current role: {p.get('current_title', '')} at {p.get('current_company', '')} "
        f"({p.get('current_company_size', '')} employees, {p.get('current_industry', '')} industry)"
    )
    lines.append(
        f"Years of experience: {p.get('years_of_experience', '')}. "
        f"Location: {p.get('location', '')}, {p.get('country', '')}."
    )
    lines.append(f"Summary: {p.get('summary', '')}")

    history = c.get("career_history", []) or []
    if history:
        lines.append("Career history:")
        for h in history:
            lines.append(
                f"- {h.get('title', '')} at {h.get('company', '')} "
                f"({h.get('industry', '')}, {h.get('company_size', '')} employees), "
                f"{h.get('start_date', '')} to {h.get('end_date') or 'present'}, "
                f"{h.get('duration_months', 0)} months. {h.get('description', '')}"
            )

    edu = c.get("education", []) or []
    if edu:
        lines.append("Education:")
        for e in edu:
            lines.append(
                f"- {e.get('degree', '')} in {e.get('field_of_study', '')} "
                f"from {e.get('institution', '')} "
                f"({e.get('start_year', '')}-{e.get('end_year', '')})"
            )

    skills = c.get("skills", []) or []
    if skills:
        assess = sig.get("skill_assessment_scores", {}) or {}
        skill_strs = []
        for s in skills:
            name = s.get("name", "")
            extra = f", platform assessment score {assess[name]}" if name in assess else ""
            skill_strs.append(
                f"{name} ({s.get('proficiency', '')}, "
                f"{s.get('duration_months', 0)} months used{extra})"
            )
        lines.append("Skills: " + "; ".join(skill_strs))

    certs = c.get("certifications", []) or []
    if certs:
        lines.append(
            "Certifications: " + "; ".join(
                f"{x.get('name', '')} ({x.get('issuer', '')}, {x.get('year', '')})"
                for x in certs
            )
        )

    lines.append(
        f"Platform activity: open to work = {sig.get('open_to_work_flag', '')}, "
        f"last active = {sig.get('last_active_date', '')}, "
        f"recruiter response rate = {sig.get('recruiter_response_rate', '')}, "
        f"notice period = {sig.get('notice_period_days', '')} days, "
        f"preferred work mode = {sig.get('preferred_work_mode', '')}, "
        f"willing to relocate = {sig.get('willing_to_relocate', '')}."
    )

    return "\n".join(lines)


# ============================================================================
# Precomputed analysis -- the arithmetic the LLM is told to trust rather
# than redo. This is the main defense against false positives: every number
# the LLM reasons over is computed deterministically in Python first.
# ============================================================================
def parse_date_safe(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return None


def months_between(d1: Optional[date], d2: Optional[date]) -> Optional[int]:
    """Approximate whole months from d1 to d2 (d2 >= d1 assumed). Good
    enough for impossibilities that are measured in multiple months/years;
    not meant for day-level precision."""
    if d1 is None or d2 is None:
        return None
    months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
    if d2.day < d1.day:
        months -= 1
    return max(months, 0)


def compute_precomputed_analysis(c: dict, reference_date: date) -> dict:
    p = c.get("profile", {}) or {}
    history = c.get("career_history", []) or []
    education = c.get("education", []) or []
    skills = c.get("skills", []) or []

    stated_years = p.get("years_of_experience")

    total_months = 0
    max_role = None
    max_months = -1
    starts: List[date] = []
    ends: List[date] = []
    parsed_roles = []
    for h in history:
        dm = h.get("duration_months")
        if isinstance(dm, (int, float)):
            total_months += dm
            if dm > max_months:
                max_months = dm
                max_role = h
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
        parsed_roles.append({
            "title": h.get("title", ""), "company": h.get("company", ""),
            "start": sd, "end": ed, "description": h.get("description", "") or "",
        })

    earliest_start = min(starts) if starts else None
    latest_end = max(ends) if ends else None
    span_months = (
        months_between(earliest_start, latest_end)
        if earliest_start and latest_end else None
    )
    span_years = round(span_months / 12, 2) if span_months is not None else None

    gap_years = None
    if isinstance(stated_years, (int, float)) and span_years is not None:
        gap_years = round(stated_years - span_years, 2)

    zero_dur_expert = [
        s.get("name") for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months") == 0
    ]

    invalid_edu = [
        {
            "institution": e.get("institution"), "degree": e.get("degree"),
            "start_year": e.get("start_year"), "end_year": e.get("end_year"),
        }
        for e in education
        if isinstance(e.get("start_year"), int) and isinstance(e.get("end_year"), int)
        and e["end_year"] < e["start_year"]
    ]

    overlaps = []
    for i in range(len(parsed_roles)):
        for j in range(i + 1, len(parsed_roles)):
            a, b = parsed_roles[i], parsed_roles[j]
            if a["start"] and a["end"] and b["start"] and b["end"]:
                latest_start = max(a["start"], b["start"])
                earliest_end = min(a["end"], b["end"])
                if latest_start < earliest_end:
                    ov_months = months_between(latest_start, earliest_end)
                    # Only surface overlaps that clear MIN_OVERLAP_MONTHS.
                    # v1 surfaced ANY overlap > 0 (including a 1-month
                    # handover, which is completely normal) and then asked
                    # the LLM to eyeball whether it was "substantial" --
                    # that's a soft, easily-miscalibrated judgment call to
                    # hand an LLM. Filtering here makes the bar objective
                    # and removes a likely source of false positives.
                    if ov_months and ov_months >= MIN_OVERLAP_MONTHS:
                        overlaps.append({
                            "role_a": f'{a["title"]} at {a["company"]}',
                            "role_b": f'{b["title"]} at {b["company"]}',
                            "overlap_months": ov_months,
                            "role_a_desc": a["description"][:200],
                            "role_b_desc": b["description"][:200],
                        })

    single_role_max_years_val = round(max_months / 12, 2) if max_months >= 0 else None
    single_role_excess_years = None
    if (
        single_role_max_years_val is not None
        and isinstance(stated_years, (int, float))
    ):
        single_role_excess_years = round(single_role_max_years_val - stated_years, 2)

    # --------------------------------------------------------------------
    # DETERMINISTIC THRESHOLD CHECKS -- computed in Python, not left to the
    # LLM. This exists because of an observed failure mode: even when the
    # exact numbers are handed to the model, an 8B model under a heavily
    # false-positive-averse rubric can still talk itself out of acting on
    # a number that plainly exceeds the stated threshold (e.g. an 8-year
    # gap against a 2-year tolerance was rationalized away rather than
    # flagged). Arithmetic comparison is exactly the kind of step that
    # should never be re-derived by the model when Python can just do it
    # and hand over a TRUE/FALSE. The LLM's job for these five categories
    # is now narrowed to: confirm the deterministic flag, OR find an
    # explicit, specific, named reason in the candidate's own text that
    # fully explains the discrepancy (not vague reasoning) -- see rubric.
    # --------------------------------------------------------------------
    gap_exceeds_threshold = (
        gap_years is not None and gap_years > SPAN_GAP_TOLERANCE_YEARS
    )
    single_role_exceeds_threshold = (
        single_role_excess_years is not None
        and single_role_excess_years > SINGLE_ROLE_TOLERANCE_YEARS
    )
    zero_dur_meets_threshold = len(zero_dur_expert) >= ZERO_DURATION_EXPERT_MIN_COUNT
    invalid_edu_present = len(invalid_edu) > 0
    overlap_present = len(overlaps) > 0  # already filtered to >= MIN_OVERLAP_MONTHS

    return {
        "stated_years_of_experience": stated_years,
        "career_history_total_months": total_months,
        "career_history_total_years": round(total_months / 12, 2) if total_months else 0,
        "single_role_max_months": max_months if max_months >= 0 else None,
        "single_role_max_years": single_role_max_years_val,
        "single_role_max_title": (
            f'{max_role.get("title", "")} at {max_role.get("company", "")}'
            if max_role else None
        ),
        "single_role_excess_years": single_role_excess_years,
        "earliest_start_date": earliest_start.isoformat() if earliest_start else None,
        "latest_end_date": latest_end.isoformat() if latest_end else None,
        "career_span_months": span_months,
        "career_span_years": span_years,
        "experience_minus_span_gap_years": gap_years,
        "zero_duration_expert_skills": zero_dur_expert,
        "zero_duration_expert_count": len(zero_dur_expert),
        "invalid_education_entries": invalid_edu,
        "overlapping_role_pairs": overlaps,
        "gap_exceeds_threshold": gap_exceeds_threshold,
        "single_role_exceeds_threshold": single_role_exceeds_threshold,
        "zero_dur_meets_threshold": zero_dur_meets_threshold,
        "invalid_edu_present": invalid_edu_present,
        "overlap_present": overlap_present,
        "any_deterministic_flag": (
            gap_exceeds_threshold or single_role_exceeds_threshold
            or zero_dur_meets_threshold or invalid_edu_present or overlap_present
        ),
    }


def render_analysis_block(a: dict) -> str:
    lines = [
        "PRECOMPUTED ANALYSIS (verified arithmetic AND verified threshold "
        "comparisons -- the DETERMINISTIC CHECK lines below are computed by "
        "Python, not by you. Do not re-derive or second-guess the "
        "arithmetic. Your only job for categories 1, 2, 3, 4, 5 is: if the "
        "DETERMINISTIC CHECK says TRUE, flag it UNLESS the candidate's own "
        "career_history/skills text gives an explicit, specific, named "
        "reason that fully explains the discrepancy -- not vague "
        "plausibility, an actual stated fact, e.g. a description that says "
        '"on sabbatical 2019-2021" or "advisory role, 5 hrs/week" overlap. '
        "If you cannot point to that kind of explicit textual override, a "
        "TRUE deterministic check must be flagged, full stop -- an "
        "impressive-looking profile is not a valid reason to wave off a "
        "TRUE check.):"
    ]
    lines.append(f"  - stated years_of_experience: {a['stated_years_of_experience']}")
    lines.append(
        f"  - sum of all career_history duration_months: "
        f"{a['career_history_total_months']} months ({a['career_history_total_years']} years)"
    )
    if a["single_role_max_months"] is not None:
        lines.append(
            f"  - single_role_max_years (longest individual role): "
            f"{a['single_role_max_years']} years ({a['single_role_max_months']} months) "
            f"-- {a['single_role_max_title']}"
        )
    if a["single_role_excess_years"] is not None:
        lines.append(
            f"  - single_role_max_years minus stated years_of_experience: "
            f"{a['single_role_excess_years']} years "
            f"(threshold: more than {SINGLE_ROLE_TOLERANCE_YEARS:.1f} years) "
            f"-> DETERMINISTIC CHECK (SINGLE_ROLE_EXCEEDS_EXPERIENCE): "
            f"{'TRUE -- exceeds threshold, flag unless explicitly explained' if a['single_role_exceeds_threshold'] else 'FALSE -- within normal range, do not flag'}"
        )
    if a["earliest_start_date"] and a["latest_end_date"]:
        lines.append(
            f"  - career_span_years (earliest career_history start to latest "
            f"end / today): {a['career_span_years']} years "
            f"({a['career_span_months']} months), from {a['earliest_start_date']} "
            f"to {a['latest_end_date']}"
        )
    if a["experience_minus_span_gap_years"] is not None:
        gap = a["experience_minus_span_gap_years"]
        lines.append(
            f"  - gap between stated years_of_experience and career_span_years: "
            f"{gap} years (threshold: more than {SPAN_GAP_TOLERANCE_YEARS:.1f} "
            f"years) -> DETERMINISTIC CHECK (CAREER_SPAN_EXCEEDS_EXPERIENCE): "
            f"{'TRUE -- exceeds threshold, flag unless explicitly explained' if a['gap_exceeds_threshold'] else 'FALSE -- within normal range, do not flag'}"
        )
    lines.append(
        f"  - zero-duration 'expert' skills ({a['zero_duration_expert_count']} "
        f"total, threshold: {ZERO_DURATION_EXPERT_MIN_COUNT} or more) -> "
        f"DETERMINISTIC CHECK (ZERO_DURATION_EXPERT_SKILL): "
        f"{'TRUE -- flag unless explicitly explained' if a['zero_dur_meets_threshold'] else 'FALSE -- do not flag'}: "
        f"{a['zero_duration_expert_skills'] or 'none'}"
    )
    lines.append(
        f"  - education entries with end_year before start_year -> "
        f"DETERMINISTIC CHECK (INVALID_EDUCATION_DATES): "
        f"{'TRUE -- flag, this category has zero tolerance' if a['invalid_edu_present'] else 'FALSE -- do not flag'}: "
        f"{a['invalid_education_entries'] or 'none'}"
    )
    lines.append(
        f"  - overlapping career_history date ranges (only overlaps of "
        f"{MIN_OVERLAP_MONTHS}+ months are listed here -- shorter handovers "
        f"already filtered out as normal) -> DETERMINISTIC CHECK "
        f"(OVERLAPPING_FULL_TIME_ROLES): "
        f"{'TRUE -- flag unless explicitly explained' if a['overlap_present'] else 'FALSE -- do not flag'}: "
        f"{a['overlapping_role_pairs'] or 'none'}"
    )
    return "\n".join(lines)


def build_honeypot_user_content(candidate_id: str, c: dict, reference_date: date) -> str:
    block = build_candidate_block(c)
    analysis = compute_precomputed_analysis(c, reference_date)
    analysis_block = render_analysis_block(analysis)
    return (
        f"CANDIDATE PROFILE (id: {candidate_id}):\n\n{block}\n\n{analysis_block}\n\n"
        "Determine if this is a honeypot now. Output only the JSON object."
    )


def build_honeypot_messages(candidate_id: str, c: dict, reference_date: date) -> List[dict]:
    if _RUBRIC_TEXT is None:
        raise RuntimeError(
            "build_honeypot_rubric() was never called to initialize "
            "_RUBRIC_TEXT -- this should happen once near the top of "
            "main() before any candidate is processed."
        )
    return [
        {"role": "system", "content": _RUBRIC_TEXT},
        {"role": "user", "content": build_honeypot_user_content(candidate_id, c, reference_date)},
    ]


# ============================================================================
# Ranked CSV loading
# ============================================================================
def load_csv_rows(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_population_stats(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_flag(raw) -> str:
    """Normalizes a honeypot_flag cell from the cross-encoder CSV to one of
    'true' / 'false' / '' (empty stays empty -- that's a legitimate NULL,
    not something to coerce into false)."""
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes"):
        return "true"
    if s in ("false", "0", "no"):
        return "false"
    return ""  # empty string, or anything unrecognized -- treated as NULL


def load_pairwise_and_tail(
    pairwise_path: str, cross_encoder_path: str
) -> Tuple[List[str], List[str], Dict[str, dict], Dict[str, str]]:
    """Returns (pairwise_ids, tail_ids, cross_encoder_rows_by_id, initial_flag_by_id).

    pairwise_ids: candidate_ids from --pointwise-ranked, in its existing
                  final_rank order. This is ranks 1..30000 of the merged
                  pool. Their honeypot status starts out NULL -- the
                  pairwise pass reordered them and nothing has re-audited
                  them yet.
    tail_ids:     candidate_ids from --cross-encoder-ranked that are NOT in
                  pairwise_ids, in their existing cross-encoder rank order.
                  This is ranks 30001..100000 of the merged pool, and it
                  naturally ends with the rule-based honeypots, since 04/05
                  already pushed them to the bottom of that file.
    cross_encoder_rows_by_id: candidate_id -> full cross-encoder row dict,
                  for EVERY id in the cross-encoder file (used for
                  diagnostics / lookups if needed).
    initial_flag_by_id: candidate_id -> 'true' / 'false' / '' , the
                  honeypot_flag each candidate starts the merged pool with,
                  BEFORE the LLM gap-fill pass runs. Only tail_ids get a
                  non-empty value here (carried over from the cross-encoder
                  file's own honeypot_flag); pairwise_ids are deliberately
                  left out of this dict (i.e. default to '') even if they
                  also happen to appear in the cross-encoder file, per the
                  "leave ranks 1..30000 NULL until re-audited" rule.
    """
    pairwise_rows = load_csv_rows(pairwise_path)
    pairwise_rows.sort(key=lambda r: int(r.get("final_rank", 10**9)))
    pairwise_ids = [r["candidate_id"] for r in pairwise_rows]
    pairwise_id_set = set(pairwise_ids)

    cross_rows = load_csv_rows(cross_encoder_path)
    cross_rows.sort(key=lambda r: int(r.get("rank", 10**9)))
    cross_encoder_rows_by_id = {r["candidate_id"]: r for r in cross_rows}

    tail_ids: List[str] = []
    initial_flag_by_id: Dict[str, str] = {}
    for r in cross_rows:
        cid = r["candidate_id"]
        if cid in pairwise_id_set:
            continue
        tail_ids.append(cid)
        initial_flag_by_id[cid] = normalize_flag(r.get("honeypot_flag"))

    return pairwise_ids, tail_ids, cross_encoder_rows_by_id, initial_flag_by_id


# ============================================================================
# Honeypot check cache -- resumable, one JSON line per checked candidate.
# ============================================================================
def load_honeypot_cache(path: str) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("rubric_version") != RUBRIC_VERSION:
                    continue  # stale -- rubric changed since this was cached
                cid = entry.get("candidate_id")
                if cid:
                    cache[cid] = entry
        print(f"Loaded {len(cache)} cached honeypot checks (rubric {RUBRIC_VERSION}) from {path}")
    return cache


def append_honeypot_result(path: str, entry: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()


# ============================================================================
# LLM output parsing / validation
# ============================================================================
def extract_json_object(raw: Optional[str]) -> Optional[str]:
    """Strip <think>...</think>, markdown fences, and surrounding text."""
    if raw is None:
        return None
    text = raw
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.replace("```json", "").replace("```", "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start:end + 1]


VALID_CATEGORIES = {
    "CAREER_SPAN_EXCEEDS_EXPERIENCE",
    "SINGLE_ROLE_EXCEEDS_EXPERIENCE",
    "ZERO_DURATION_EXPERT_SKILL",
    "INVALID_EDUCATION_DATES",
    "OVERLAPPING_FULL_TIME_ROLES",
    "OTHER_LOGICAL_IMPOSSIBILITY",
}


def validate_and_normalize_honeypot(parsed) -> Optional[dict]:
    """Returns None if the parse is unusable (triggers a retry / fallback).
    Deliberately strict: a positive flag with no valid category or no
    evidence string is rejected rather than trusted, since that's exactly
    the kind of ungrounded "true" that would create a false positive."""
    if not isinstance(parsed, dict):
        return None

    is_hp = parsed.get("is_honeypot")
    if not isinstance(is_hp, bool):
        return None

    category = parsed.get("violation_category")
    evidence = parsed.get("evidence")

    if is_hp:
        if category not in VALID_CATEGORIES:
            return None
        if not evidence or not str(evidence).strip():
            return None
        evidence = str(evidence).strip()
    else:
        category = None
        evidence = None

    rationale = str(parsed.get("brief_rationale", "")).strip()

    return {
        "is_honeypot": is_hp,
        "violation_category": category,
        "evidence": evidence,
        "brief_rationale": rationale,
    }


# ============================================================================
# vLLM engine -- copied verbatim (singleton pattern + settings) from
# 06_llm_pointwise_score.py.
# ============================================================================
_ENGINE = None


def _get_engine(
    model_name, tensor_parallel_size, gpu_memory_utilization, max_model_len,
    max_num_seqs, enforce_eager, enable_prefix_caching, swap_space
):
    global _ENGINE
    if _ENGINE is None:
        from vllm import LLM
        _ENGINE = LLM(
            model=model_name,
            dtype="float16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            disable_custom_all_reduce=True,
            enforce_eager=enforce_eager,
            enable_flashinfer_autotune=False,
            # On by default: the rubric (system prompt) here is identical
            # for every call, same rationale as 06's pointwise scorer.
            enable_prefix_caching=enable_prefix_caching,
        )
    return _ENGINE


def call_llm_batch(
    message_batches, model_name, tensor_parallel_size, schema,
    temperature=0.0, max_tokens=300, gpu_memory_utilization=0.90,
    max_model_len=2048, max_num_seqs=1, enforce_eager=False,
    enable_prefix_caching=True, swap_space=4.0,
):
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    engine = _get_engine(
        model_name, tensor_parallel_size, gpu_memory_utilization, max_model_len,
        max_num_seqs, enforce_eager, enable_prefix_caching, swap_space,
    )
    try:
        outputs = engine.chat(
            message_batches, sp,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        outputs = engine.chat(message_batches, sp)
    except AttributeError:
        tok = engine.get_tokenizer()
        prompts = [
            tok.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
            for m in message_batches
        ]
        outputs = engine.generate(prompts, sp)
    return [o.outputs[0].text for o in outputs]


# ============================================================================
# Pre-flight checks -- same two-stage pattern as 06: token budget first
# (cheap, no GPU), then actual free VRAM on every GPU (right before the
# engine is created).
# ============================================================================
def preflight_check_honeypot_prompt_budget(
    ids: List[str], profiles: Dict[str, dict], reference_date: date,
    model_name: str, max_model_len: int, max_tokens: int,
) -> None:
    print(f"\nPre-flight check 1/2: worst-case honeypot-check prompt length for {len(ids)} candidates ...")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(
            f"  WARNING: could not load tokenizer to check prompt budget ({e}). "
            "Skipping -- vLLM will enforce max_model_len itself at request time."
        )
        return

    worst_id, worst_tokens, worst_prompt_tokens = None, -1, -1
    for cid in ids:
        if cid not in profiles:
            continue
        content = build_honeypot_user_content(cid, profiles[cid], reference_date)
        n = len(tok.encode(content, add_special_tokens=False))
        if n > worst_tokens:
            worst_tokens = n
            worst_id = cid

    if worst_id is None:
        print("  WARNING: no profiles available to check -- skipping.")
        return

    messages = build_honeypot_messages(worst_id, profiles[worst_id], reference_date)
    formatted = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompt_tokens = len(tok.encode(formatted, add_special_tokens=False))
    total_required = prompt_tokens + max_tokens

    print(f"  Largest candidate block: {worst_id} ({worst_tokens} tok of profile+analysis text)")
    print(f"  Worst-case full prompt (rubric + block + chat template): {prompt_tokens} tok")
    print(f"  + generation budget (--max-tokens): {max_tokens} tok")
    print(f"  = total required: {total_required} tok   (your --max-model-len is {max_model_len})")

    if total_required > max_model_len:
        deficit = total_required - max_model_len
        raise SystemExit(
            f"\nABORTING before touching the GPU: --max-model-len {max_model_len} is "
            f"{deficit} tokens too small. Rerun with at least "
            f"--max-model-len {total_required + 64}."
        )

    headroom = max_model_len - total_required
    print(f"  OK -- {headroom} tokens of headroom.")


def preflight_check_vram(
    gpu_memory_utilization: float, num_gpus: int = 1, safety_margin: float = 0.97
) -> None:
    """Checks every GPU that vLLM will use. Copied from 06."""
    print(f"\nPre-flight check 2/2: actual free VRAM on all {num_gpus} GPU(s) ...")
    try:
        import torch
        if not torch.cuda.is_available():
            print(
                "  WARNING: torch.cuda.is_available() is False -- skipping. "
                "vLLM will fail on its own if there's no GPU visible."
            )
            return
        visible_count = torch.cuda.device_count()
    except Exception as e:
        print(f"  WARNING: could not query GPU memory ({e}). Skipping this check.")
        return

    if visible_count < num_gpus:
        raise SystemExit(
            f"\nABORTING: --num-gpus {num_gpus} but only {visible_count} GPU(s) "
            f"are visible (check --gpu-ids / CUDA_VISIBLE_DEVICES)."
        )

    any_insufficient = False
    for idx in range(num_gpus):
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        total_gb = total_bytes / 1e9
        free_gb = free_bytes / 1e9
        already_used_gb = total_gb - free_gb
        requested_gb = gpu_memory_utilization * total_gb

        print(
            f"  GPU (logical index {idx}): total {total_gb:.2f} GB, "
            f"in use {already_used_gb:.2f} GB, free {free_gb:.2f} GB, "
            f"this run will request {requested_gb:.2f} GB"
        )

        if requested_gb > free_gb * safety_margin:
            max_safe_util = (free_gb * safety_margin) / total_gb
            print(f"  ^ INSUFFICIENT. Max safe --gpu-memory-utilization here is ~{max_safe_util:.3f}.")
            any_insufficient = True

    if any_insufficient:
        raise SystemExit(
            "\nABORTING before starting the engine: at least one GPU doesn't "
            "have enough free VRAM. Fix:\n"
            "  1) Kill stale python processes holding VRAM (nvidia-smi).\n"
            "  2) Lower --gpu-memory-utilization to the 'max safe' value above.\n"
            "  3) Check --gpu-ids excludes any thermal-throttling card."
        )

    print("  OK -- all GPUs have sufficient headroom.")


# ============================================================================
# Resolve one batch of candidate_ids to honeypot verdicts, using the cache
# first and the LLM (with retries) for anything not yet cached.
# ============================================================================
def resolve_honeypot_batch(
    cids: List[str],
    profiles: Dict[str, dict],
    reference_date: date,
    cache: Dict[str, dict],
    cache_path: str,
    args,
    stats: dict,
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    need_llm: List[str] = []

    for cid in cids:
        if cid in cache:
            results[cid] = cache[cid]
            stats["cache_hits"] += 1
        elif cid not in profiles:
            # Profile wasn't preloaded -- almost certainly means
            # --secondary-preload was too small for how deep this run had
            # to dig. Fail loudly with an actionable fix rather than
            # silently mis-scoring a candidate we never actually looked at.
            raise SystemExit(
                f"\nABORTING: candidate {cid} needs a honeypot check but its "
                f"full profile was never preloaded. This means the gap-fill "
                f"had to dig deeper into the cross-encoder tail than "
                f"--secondary-preload ({args.secondary_preload}) covered. "
                f"Rerun with a larger --secondary-preload (cached results "
                f"for already-checked candidates will be reused, so this "
                f"is cheap to retry)."
            )
        else:
            need_llm.append(cid)

    if not need_llm:
        return results

    pending = list(need_llm)
    resolved: Dict[str, dict] = {}
    last_raw: Dict[str, str] = {}

    for attempt in range(args.max_retries + 1):
        still_pending = [cid for cid in pending if cid not in resolved]
        if not still_pending:
            break
        msgs = [
            build_honeypot_messages(cid, profiles[cid], reference_date)
            for cid in still_pending
        ]
        raw_outputs = call_llm_batch(
            msgs,
            args.model_name,
            args.num_gpus,
            None,
            temperature=0.0,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=args.enable_prefix_caching,
            swap_space=args.swap_space,
        )
        for cid, raw in zip(still_pending, raw_outputs):
            last_raw[cid] = raw
            cleaned = extract_json_object(raw)
            try:
                parsed = json.loads(cleaned) if cleaned is not None else None
            except json.JSONDecodeError:
                parsed = None
            normalized = (
                validate_and_normalize_honeypot(parsed) if parsed is not None else None
            )
            if normalized is not None:
                resolved[cid] = normalized

    for cid in need_llm:
        if cid in resolved:
            entry = {
                "candidate_id": cid, "rubric_version": RUBRIC_VERSION,
                **resolved[cid],
            }
            stats["llm_calls"] += 1

            # ----------------------------------------------------------
            # Safety net: this exists because of an observed real miss --
            # an 8.03-year CAREER_SPAN_EXCEEDS_EXPERIENCE gap against a
            # 2.0-year threshold (4x overshoot) was returned as
            # is_honeypot=false by the LLM despite the deterministic
            # check being unambiguously TRUE. The rubric now instructs the
            # model not to do this, but we don't fully trust an 8B model
            # to follow that instruction 100% of the time, so we also
            # verify it here in Python and surface a loud, impossible-to-
            # miss warning (and a queryable flag in the cache) any time
            # the model disagrees with a TRUE deterministic check. This
            # does NOT auto-override the verdict -- a genuine textual
            # explanation (sabbatical, advisory role, etc.) is a valid
            # reason for the model to say false, and we don't want to
            # reintroduce false positives by blindly trusting Python's
            # arithmetic over the model's reading of the actual text.
            # It DOES mean every such disagreement is logged for a human
            # to spot-check before the final submission goes out.
            # ----------------------------------------------------------
            if not entry["is_honeypot"] and cid in profiles:
                det = compute_precomputed_analysis(profiles[cid], reference_date)
                if det["any_deterministic_flag"]:
                    tripped = [
                        name for name, flag in [
                            ("CAREER_SPAN_EXCEEDS_EXPERIENCE", det["gap_exceeds_threshold"]),
                            ("SINGLE_ROLE_EXCEEDS_EXPERIENCE", det["single_role_exceeds_threshold"]),
                            ("ZERO_DURATION_EXPERT_SKILL", det["zero_dur_meets_threshold"]),
                            ("INVALID_EDUCATION_DATES", det["invalid_edu_present"]),
                            ("OVERLAPPING_FULL_TIME_ROLES", det["overlap_present"]),
                        ] if flag
                    ]
                    entry["deterministic_override_disagreement"] = True
                    entry["deterministic_categories_tripped"] = tripped
                    stats["deterministic_disagreements"] = stats.get("deterministic_disagreements", 0) + 1
                    print(
                        f"  *** REVIEW FLAG *** {cid}: LLM said is_honeypot=False but "
                        f"deterministic check(s) {tripped} are TRUE "
                        f"(gap_years={det['experience_minus_span_gap_years']}, "
                        f"single_role_excess_years={det['single_role_excess_years']}, "
                        f"zero_dur_count={det['zero_duration_expert_count']}). "
                        f"brief_rationale was: {entry.get('brief_rationale', '')!r}. "
                        f"This candidate was kept as NOT a honeypot per the LLM verdict, "
                        f"but is logged here for manual spot-check before submission."
                    )
        else:
            # Parse failed on every retry. Fallback policy mirrors the
            # rubric's own "when uncertain, default false" instruction:
            # an unparseable response is NOT evidence of a honeypot, so we
            # do not disqualify the candidate on a plumbing failure. We do
            # flag it clearly for manual review.
            entry = {
                "candidate_id": cid, "rubric_version": RUBRIC_VERSION,
                "is_honeypot": False, "violation_category": None, "evidence": None,
                "brief_rationale": (
                    f"FALLBACK: no parseable LLM output after "
                    f"{args.max_retries + 1} attempts; defaulting to "
                    f"not-honeypot per false-positive-avoidance policy. "
                    f"MANUAL REVIEW RECOMMENDED. Last raw output: "
                    f"{(last_raw.get(cid) or '')[:150]!r}"
                ),
                "fallback": True,
            }
            stats["fallbacks"] += 1
            print(f"  WARNING: honeypot check for {cid} fell back to is_honeypot=False after parse failures.")
        cache[cid] = entry
        append_honeypot_result(cache_path, entry)
        results[cid] = entry

    return results


# ============================================================================
# THE GAP-FILL ALGORITHM
# ============================================================================
def run_honeypot_gap_fill(
    primary_ids: List[str],
    secondary_ids: List[str],
    profiles: Dict[str, dict],
    reference_date: date,
    cache: Dict[str, dict],
    cache_path: str,
    args,
    stats: dict,
) -> Tuple[List[str], List[str], List[str], List[dict], int]:
    """Walks down primary_ids (then secondary_ids if primary is exhausted),
    checking candidates in batches, until exactly args.top_k clean
    candidates are collected.

    Returns:
      clean_top:            list of candidate_id, length == args.top_k,
                             best-first, confirmed not honeypots.
      remainder_primary:    primary_ids not used above and not flagged,
                             in original relative order.
      remainder_secondary:  secondary_ids not used above and not flagged,
                             in original relative order.
      honeypot_records:     list of dicts for every NEWLY found honeypot
                             (candidate_id, source, violation_category,
                             evidence, brief_rationale).
      total_checked:        number of candidates the LLM/cache was asked
                             about during this run (for reporting).
    """
    combined: List[Tuple[str, str]] = (
        [("primary", cid) for cid in primary_ids]
        + [("secondary", cid) for cid in secondary_ids]
    )
    primary_n = len(primary_ids)

    clean_top: List[str] = []
    honeypot_set = set()
    honeypot_records: List[dict] = []
    cursor = 0

    while len(clean_top) < args.top_k:
        if cursor >= len(combined):
            raise SystemExit(
                f"\nABORTING: ran out of candidates (checked all "
                f"{len(combined)}) while trying to fill a clean top-"
                f"{args.top_k}. Either the honeypot rate is implausibly "
                f"high, or --pointwise-ranked / --cross-encoder-ranked don't "
                f"cover the full pool. Found {len(clean_top)} clean and "
                f"{len(honeypot_records)} honeypots so far."
            )
        if cursor >= args.max_checks:
            raise SystemExit(
                f"\nABORTING: exceeded --max-checks ({args.max_checks}) "
                f"while only {len(clean_top)}/{args.top_k} clean candidates "
                f"found. This safety cap exists to stop a runaway loop -- "
                f"raise --max-checks if you've confirmed this is expected."
            )

        batch = combined[cursor: cursor + args.batch_size]
        cursor += len(batch)

        batch_cids = [cid for _, cid in batch]
        results = resolve_honeypot_batch(
            batch_cids, profiles, reference_date, cache, cache_path, args, stats
        )

        for source, cid in batch:
            res = results[cid]
            if res["is_honeypot"]:
                honeypot_set.add(cid)
                honeypot_records.append({
                    "candidate_id": cid,
                    "source": source,
                    "violation_category": res.get("violation_category"),
                    "evidence": res.get("evidence"),
                    "brief_rationale": res.get("brief_rationale"),
                })
            elif len(clean_top) < args.top_k:
                clean_top.append(cid)
            # else: clean but found in an "overshoot" batch after top_k was
            # already filled -- leave it to fall through to the remainder
            # list below, in its normal relative position.

        stats["total_evaluated"] = cursor
        elapsed = time.time() - stats["t0"]
        print(
            f"  checked {cursor} candidates so far "
            f"({len(clean_top)}/{args.top_k} clean, "
            f"{len(honeypot_records)} honeypot(s) found) -- {elapsed:.1f}s elapsed"
        )

    cursor_primary_end = min(cursor, primary_n)
    cursor_secondary_end = max(0, cursor - primary_n)

    checked_primary = primary_ids[:cursor_primary_end]
    checked_secondary = secondary_ids[:cursor_secondary_end]
    clean_top_set = set(clean_top)

    remainder_primary = (
        [cid for cid in checked_primary if cid not in clean_top_set and cid not in honeypot_set]
        + primary_ids[cursor_primary_end:]
    )
    remainder_secondary = (
        [cid for cid in checked_secondary if cid not in clean_top_set and cid not in honeypot_set]
        + secondary_ids[cursor_secondary_end:]
    )

    return clean_top, remainder_primary, remainder_secondary, honeypot_records, cursor


# ============================================================================
# Output assembly
# ============================================================================
OUT_COLUMNS = [
    "final_rank",
    "candidate_id",
    "honeypot_flag",
]


def build_checked_id_set(
    pairwise_ids: List[str], tail_ids: List[str], total_checked: int,
) -> set:
    """Recomputes exactly which candidate_ids the gap-fill cursor actually
    reached (checked_primary + checked_secondary inside
    run_honeypot_gap_fill), from the single `total_checked` (cursor) value
    that function returns. Mirrors that function's own
    cursor_primary_end / cursor_secondary_end split exactly."""
    pairwise_n = len(pairwise_ids)
    cursor_pairwise_end = min(total_checked, pairwise_n)
    cursor_tail_end = max(0, total_checked - pairwise_n)
    return set(pairwise_ids[:cursor_pairwise_end]) | set(tail_ids[:cursor_tail_end])


def row_for_clean_or_remainder(cid: str, checked_ids: set, initial_flag_by_id: Dict[str, str]) -> dict:
    """A candidate that the gap-fill cursor actually checked (and found
    clean) is LLM-confirmed clean -> 'false'. A candidate never reached by
    the cursor keeps whatever flag the merged pool started it with: NULL
    for pairwise-origin candidates, the carried-over rule-based flag for
    cross-encoder-tail-origin candidates."""
    if cid in checked_ids:
        flag = "false"
    else:
        flag = initial_flag_by_id.get(cid, "")
    return {"candidate_id": cid, "honeypot_flag": flag}


def row_for_honeypot(record: dict) -> dict:
    return {"candidate_id": record["candidate_id"], "honeypot_flag": "true"}


def assemble_final_rows(
    clean_top: List[str],
    remainder_primary: List[str],
    remainder_secondary: List[str],
    honeypot_records: List[dict],
    checked_ids: set,
    initial_flag_by_id: Dict[str, str],
) -> List[dict]:
    """Reassembles the merged pool in its original relative order --
    pairwise ids first (1..30000), then cross-encoder tail ids
    (30001..100000) -- with ONLY the newly LLM-confirmed honeypots pulled
    out and moved to the very bottom. Nothing else is reshuffled: a
    checked-clean candidate keeps its spot and gets 'false'; an
    un-reached candidate keeps its spot and keeps its starting flag."""
    rows: List[dict] = []

    for cid in clean_top:
        rows.append(row_for_clean_or_remainder(cid, checked_ids, initial_flag_by_id))
    for cid in remainder_primary:
        rows.append(row_for_clean_or_remainder(cid, checked_ids, initial_flag_by_id))
    for cid in remainder_secondary:
        rows.append(row_for_clean_or_remainder(cid, checked_ids, initial_flag_by_id))
    for record in honeypot_records:
        rows.append(row_for_honeypot(record))

    for i, row in enumerate(rows, start=1):
        row["final_rank"] = i

    return rows


def write_final_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in OUT_COLUMNS})


# ============================================================================
# Main
# ============================================================================
def main():
    global SPAN_GAP_TOLERANCE_YEARS, SINGLE_ROLE_TOLERANCE_YEARS, MIN_OVERLAP_MONTHS, ZERO_DURATION_EXPERT_MIN_COUNT, _RUBRIC_TEXT

    ap = argparse.ArgumentParser(
        description="LLM honeypot audit of the merged pairwise+cross-encoder pool's top-K, with automatic gap-fill."
    )
    ap.add_argument("--point-wise-ranked", default="outputs/llm_pointwise_top30000.csv",
                    help="Pairwise-reranked top 30,000. Columns: final_rank,candidate_id,sort_method,previous_pointwise_rank. Authoritative order for ranks 1..30000.")
    ap.add_argument("--cross-encoder-ranked", default="outputs/cross_encoder_ranked_honeypot.csv",
                    help="Full 100,000-row upstream ranking (includes rule-based honeypots at the bottom). Rows not in --pointwise-ranked supply ranks 30001..100000.")
    ap.add_argument("--population-stats", default="outputs/honeypot_population_stats.json",
                    help="JSON produced by 06b_honeypot_population_stats.py; used by default to load the reference date and calibrated thresholds.")
    ap.add_argument("--candidates", default="candidates.jsonl",
                    help="Full candidate pool (.jsonl.gz / .jsonl / .json)")
    ap.add_argument("--reference-date", default=None,
                    help="Optional override for the fixed reference date (YYYY-MM-DD). If omitted, the script loads it from --population-stats.")
    ap.add_argument("--top-k", type=int, default=100,
                    help="Size of the clean top slice to produce (100 for the hackathon).")
    ap.add_argument("--out", default="outputs/final_ranked.csv",
                    help="Output CSV: final_rank,candidate_id,honeypot_flag for all 100,000 candidates.")
    ap.add_argument("--honeypot-cache", default="outputs/llm_honeypot_checks.jsonl",
                    help="Resumable JSONL audit trail of every honeypot check performed.")
    ap.add_argument("--secondary-preload", type=int, default=5000,
                    help="How many candidates beyond the pairwise list (i.e. from the "
                         "cross-encoder tail) to preload full profiles for, in "
                         "case the gap-fill has to dig past the end of the pairwise "
                         "list. Only matters in the rare case of extreme "
                         "honeypot contamination near the top.")
    ap.add_argument("--max-checks", type=int, default=5000,
                    help="Safety cap on total candidates checked before aborting.")
    ap.add_argument("--span-gap-tolerance-years", type=float, default=None,
                    help="Optional override for CAREER_SPAN_EXCEEDS_EXPERIENCE. If omitted, the calibrated value is loaded from --population-stats.")
    ap.add_argument("--single-role-tolerance-years", type=float, default=None,
                    help="Optional override for SINGLE_ROLE_EXCEEDS_EXPERIENCE. If omitted, the calibrated value is loaded from --population-stats.")
    ap.add_argument("--min-overlap-months", type=int, default=None,
                    help="Optional override for OVERLAPPING_FULL_TIME_ROLES. If omitted, the calibrated value is loaded from --population-stats.")
    ap.add_argument("--zero-duration-expert-min-count", type=int, default=None,
                    help="Optional override for ZERO_DURATION_EXPERT_SKILL. If omitted, the calibrated value is loaded from --population-stats.")

    ap.add_argument("--model-name", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3-8B-AWQ"))
    ap.add_argument("--gpu-ids", default="1,0",
                    help="Comma-separated physical GPU ids, e.g. '0,1'. Leave "
                         "empty to respect the shell's CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--num-gpus", type=int, default=2,
                    help="Tensor-parallel size inside vLLM.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    ap.add_argument("--max-model-len", type=int, default=4736)
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    ap.add_argument("--enable-prefix-caching", action="store_true", default=True)
    ap.add_argument("--no-enable-prefix-caching", dest="enable_prefix_caching", action="store_false")
    ap.add_argument("--max-tokens", type=int, default=300,
                    help="Generation budget per check. The 4-field JSON response "
                         "typically needs 80-250 tokens.")
    ap.add_argument("--swap-space", type=float, default=4.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Candidates per vLLM call. NOTE: once the clean-top-K "
                         "quota is filled, the script still finishes processing "
                         "the batch already in flight, so the total checked can "
                         "overshoot the strict minimum by up to (batch_size - 1). "
                         "Use --batch-size 1 if you need the exact minimal check "
                         "count (slower).")
    args = ap.parse_args()

    stats = load_population_stats(args.population_stats)

    reference_date_raw = args.reference_date or stats.get("reference_date")
    if not reference_date_raw:
        raise SystemExit(
            f"Could not determine a reference date. Provide --reference-date or "
            f"ensure {args.population_stats!r} contains a 'reference_date' field."
        )
    try:
        reference_date = date.fromisoformat(str(reference_date_raw))
    except ValueError:
        raise SystemExit(f"reference date must be YYYY-MM-DD, got {reference_date_raw!r}")

    recommended = stats.get("recommended_07_thresholds", {}) or {}

    # Wire the tunable false-positive thresholds (and the rubric text that's
    # rendered from them) into the module globals that compute_precomputed_
    # analysis() / render_analysis_block() / build_honeypot_messages() read.
    # Must happen before any preflight check or candidate is processed.
    SPAN_GAP_TOLERANCE_YEARS = (
        args.span_gap_tolerance_years
        if args.span_gap_tolerance_years is not None
        else float(recommended.get("span_gap_tolerance_years", 2.0))
    )
    SINGLE_ROLE_TOLERANCE_YEARS = (
        args.single_role_tolerance_years
        if args.single_role_tolerance_years is not None
        else float(recommended.get("single_role_tolerance_years", 4.0))
    )
    MIN_OVERLAP_MONTHS = (
        args.min_overlap_months
        if args.min_overlap_months is not None
        else int(recommended.get("min_overlap_months", 3))
    )
    ZERO_DURATION_EXPERT_MIN_COUNT = (
        args.zero_duration_expert_min_count
        if args.zero_duration_expert_min_count is not None
        else int(recommended.get("zero_duration_expert_min_count", 1))
    )
    _RUBRIC_TEXT = build_honeypot_rubric(
        SPAN_GAP_TOLERANCE_YEARS, SINGLE_ROLE_TOLERANCE_YEARS,
        MIN_OVERLAP_MONTHS, ZERO_DURATION_EXPERT_MIN_COUNT,
    )
    print(
        f"Using population stats from {args.population_stats} "
        f"(reference_date={reference_date.isoformat()}, "
        f"source={stats.get('reference_date_source', 'unknown')})"
    )
    print(
        f"Honeypot rubric {RUBRIC_VERSION} thresholds: "
        f"span-gap tolerance = {SPAN_GAP_TOLERANCE_YEARS:.1f} years, "
        f"single-role tolerance = {SINGLE_ROLE_TOLERANCE_YEARS:.1f} years, "
        f"min overlap = {MIN_OVERLAP_MONTHS} months, "
        f"zero-duration-expert min count = {ZERO_DURATION_EXPERT_MIN_COUNT}"
    )

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    configure_visible_gpus(gpu_ids)
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be >= 1")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.honeypot_cache) or ".", exist_ok=True)

    print(f"Loading {args.point_wise_ranked} and {args.cross_encoder_ranked} ...")
    pairwise_ids, tail_ids, cross_encoder_rows_by_id, initial_flag_by_id = (
        load_pairwise_and_tail(args.point_wise_ranked, args.cross_encoder_ranked)
    )
    print(f"  pairwise-ranked list: {len(pairwise_ids)} candidates (ranks 1..{len(pairwise_ids)})")
    print(f"  cross-encoder tail:   {len(tail_ids)} candidates (ranks {len(pairwise_ids)+1}..{len(pairwise_ids)+len(tail_ids)})")
    print(f"  total pool covered:   {len(pairwise_ids) + len(tail_ids)} candidates")

    preload_ids = set(pairwise_ids) | set(tail_ids[:args.secondary_preload])
    print(f"\nLooking up full profiles for {len(preload_ids)} candidates in {args.candidates} ...")
    profiles = load_candidate_profiles(args.candidates, preload_ids)
    print(f"{len(profiles)} profiles loaded.")

    preflight_check_honeypot_prompt_budget(
        list(preload_ids), profiles, reference_date,
        args.model_name, args.max_model_len, args.max_tokens,
    )
    preflight_check_vram(args.gpu_memory_utilization, num_gpus=args.num_gpus)

    cache = load_honeypot_cache(args.honeypot_cache)

    stats = {
        "llm_calls": 0, "cache_hits": 0, "fallbacks": 0,
        "total_evaluated": 0, "t0": time.time(),
    }

    print(f"\nFilling a clean top-{args.top_k} (checking down from rank 1 until "
          f"{args.top_k} non-honeypots are found) ...")
    clean_top, remainder_primary, remainder_secondary, honeypot_records, total_checked = (
        run_honeypot_gap_fill(
            pairwise_ids, tail_ids, profiles, reference_date,
            cache, args.honeypot_cache, args, stats,
        )
    )

    elapsed = time.time() - stats["t0"]
    print(f"\nDone. Checked {total_checked} candidates in {elapsed:.1f}s.")
    print(f"  New LLM calls:  {stats['llm_calls']}")
    print(f"  Cache hits:     {stats['cache_hits']}")
    print(f"  Fallbacks:      {stats['fallbacks']}")
    n_disagree = stats.get("deterministic_disagreements", 0)
    print(f"  Deterministic-check disagreements (LLM said False, math said True): {n_disagree}")
    if n_disagree:
        print(
            "    ^ REVIEW THESE BEFORE SUBMITTING -- see '*** REVIEW FLAG ***' lines "
            "above and the 'deterministic_override_disagreement' / "
            "'deterministic_categories_tripped' fields in the honeypot cache JSONL."
        )
    print(f"  Honeypots found in the top slice: {len(honeypot_records)}")
    for rec in honeypot_records:
        print(
            f"    - {rec['candidate_id']} ({rec['source']}): "
            f"{rec['violation_category']} -- {rec['evidence']}"
        )

    # Diagnostic breakdown by category -- if one category dominates the
    # flag count, that's exactly the signal that category's threshold (or
    # the data feeding it) needs another look. This is what would have
    # made the v1 false-positive blowout immediately diagnosable.
    if honeypot_records:
        cat_counts = Counter(rec["violation_category"] for rec in honeypot_records)
        print("  Breakdown by category:")
        for cat, n in cat_counts.most_common():
            print(f"    {cat}: {n}")

    flag_rate = len(honeypot_records) / total_checked if total_checked else 0.0
    print(f"  Flag rate: {flag_rate:.1%} of candidates checked")
    if total_checked >= 20 and flag_rate > 0.25:
        print(
            f"  ⚠ This flag rate is implausibly high for a rare, deliberately-"
            f"planted signal (~80/100,000 ≈ 0.08% pool-wide). If you're seeing "
            f"this, one category is very likely still over-triggering -- check "
            f"the breakdown above for whichever category dominates, and "
            f"consider raising --span-gap-tolerance-years and/or "
            f"--min-overlap-months further before trusting this output."
        )

    checked_ids = build_checked_id_set(pairwise_ids, tail_ids, total_checked)
    rows = assemble_final_rows(
        clean_top, remainder_primary, remainder_secondary, honeypot_records,
        checked_ids, initial_flag_by_id,
    )
    write_final_csv(args.out, rows)

    total_in = len(pairwise_ids) + len(tail_ids)
    total_out = len(rows)
    print(f"\nWrote {total_out} rows to {args.out}")
    print(f"  clean_top_k:              {len(clean_top)}")
    print(f"  pairwise_remainder:       {len(remainder_primary)}")
    print(f"  cross_encoder_remainder:  {len(remainder_secondary)}")
    print(f"  llm_detected_honeypot:    {len(honeypot_records)}")
    print(f"  ----------------------------------------")
    print(f"  total in (input pool):    {total_in}")
    print(f"  total out (final pool):  {total_out}")
    if total_in != total_out:
        print(
            f"  ⚠ MISMATCH -- input and output totals differ. This should "
            f"never happen; please report this as a bug before trusting "
            f"the output."
        )
    else:
        print("  ✓ totals match -- nothing lost, nothing duplicated.")
    print(f"\nAudit trail: {args.honeypot_cache}")


if __name__ == "__main__":
    main()