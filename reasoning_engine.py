#!/usr/bin/env python3
"""
reasoning_engine.py
=====================
Fallback template renderer for the final ranking pipeline.

In the normal submission flow, the LLM-based reasoning path is primary.
This module is only used when that path is unavailable or fails. It takes
the already-verified TreeSHAP feature contributions from the trained
LightGBM ranker, combines them with the candidate's raw profile facts,
and returns the short human-readable `reasoning` string required by the
submission format.

What this fallback does
-----------------------
- uses the model's own per-row TreeSHAP contributions, not hand-written rules
- turns the strongest positive and negative drivers into a short note
- keeps every statement grounded in the candidate's raw facts or engineered features
- preserves rank consistency by narrating the same contributions that produced the score
- stays deterministic and CPU-only, with no network calls or external LLM dependency

How it fits into the repo
-------------------------
- 11_train_lgbm_ranker.py trains the LightGBM ranker
- 12_predict_lgbm_cpu.py scores the full pool and can call this module
- rank.py / the final ranking script should use the LLM-based reasoning path first
- this file is a safe fallback only, useful when the LLM-based realizer is missing or fails

Why TreeSHAP is used
--------------------
LightGBM's `Booster.predict(X, pred_contrib=True)` computes exact TreeSHAP
values natively in C++. For every row it returns one value per feature
column plus a trailing bias term such that:

    predicted_score == bias + sum(contribution[f] for f in features)

That makes the reasoning trace tied directly to the score that produced
the rank. The text should explain why a candidate ranked well or poorly
without inventing facts, names, numbers, or unsupported claims.

Design
------
The module keeps curated renderers for the features that matter most to
the job description and a generic fallback for everything else. It can
also extract raw profile facts for the final top-N rows so the wording
can mention titles, locations, companies, education, and skills when
those facts are actually present.

Never fabricate
---------------
Every number and claim in the rendered text comes from either the raw
candidate JSON or the engineered feature row. If raw facts are missing,
the fallback uses only the engineered features rather than guessing.
"""

import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from feature_engineering import KEYWORD_FAMILIES

# ============================================================================
# Tunables
# ============================================================================
MAX_POSITIVE_CLAUSES = 3
MAX_NEGATIVE_CLAUSES = 1
# A contribution must be at least this fraction of the row's own largest
# |contribution| to be considered a "driver" worth narrating -- relative,
# not absolute, since the raw score scale isn't comparable row to row.
RELATIVE_DRIVER_THRESHOLD = 0.10
MAX_REASONING_CHARS = 320

CONNECTORS = ["; ", "; ", "; "]  # kept simple/consistent on purpose -- see
# module docstring "VARIATION IS STRUCTURAL" -- variety comes from WHICH
# facts are chosen, not from randomizing punctuation, which would read as
# noise rather than substance under Stage-4 manual review.


# ============================================================================
# Human labels for the 18 keyword families (feature_engineering.py is the
# source of truth for the family list itself -- KEYWORD_FAMILIES.keys()).
# ============================================================================
FAMILY_LABELS: Dict[str, str] = {
    "retrieval_embeddings": "retrieval/embeddings work (RAG, dense retrieval, sentence-transformers)",
    "vector_db_hybrid_search": "vector database / hybrid search infrastructure",
    "evaluation_ranking_metrics": "ranking-evaluation rigor (NDCG/MRR/MAP, offline-online correlation)",
    "llm_finetuning": "LLM fine-tuning (LoRA/QLoRA/PEFT)",
    "learning_to_rank_models": "learning-to-rank modeling (LambdaMART, neural/XGBoost rankers)",
    "distributed_inference": "distributed/inference-serving systems (Kubernetes, vLLM, multi-GPU)",
    "open_source_validation": "open-source contributions, talks, or publications",
    "nlp_ir_domain": "NLP / information-retrieval domain work",
    "cv_speech_robotics_domain": "computer-vision/speech/robotics domain work",
    "production_deployment": "shipping to real production traffic at scale",
    "research_only": "academic/research-only framing",
    "pre_llm_ml_background": "pre-LLM-era ML/IR background (recommenders, classifiers, search ranking)",
    "llm_only_recent": "recent LLM-API-only tooling (LangChain/GPT/ChatGPT)",
    "code_quality_signals": "code-quality practice (code review, CI/CD, test coverage)",
    "hr_recruiting_marketplace_domain": "HR-tech / recruiting-marketplace domain exposure",
    "marketing_nontech_titles": "non-technical (marketing/sales/HR/finance) title language",
    "management_architect_titles": "management/architect-track title language",
    "title_chasing_seniority_words": "rapid seniority-title escalation language",
}

# JD's own framing for the 4 "absolutely need" families and the families
# that map directly onto a named disqualifier -- used to decide whether a
# curated clause should explicitly say "this is one of the JD's must-haves"
# / "this is one of the JD's named disqualifiers" rather than just describing
# the count.
JD_MUST_HAVE_FAMILIES = {
    "retrieval_embeddings", "vector_db_hybrid_search",
    "evaluation_ranking_metrics",
}
JD_NICE_TO_HAVE_FAMILIES = {
    "llm_finetuning", "learning_to_rank_models", "distributed_inference",
    "open_source_validation",
}


def _safe_get(d: Optional[Mapping[str, Any]], key: str, default: Any = None) -> Any:
    if not d:
        return default
    v = d.get(key, default)
    return default if v is None else v


def _fmt_num(x: Any, decimals: int = 1) -> str:
    try:
        return f"{float(x):.{decimals}f}"
    except (TypeError, ValueError):
        return "unknown"


def _stable_choice(seed_text: str, options: Sequence[str]) -> str:
    """Deterministic, not random -- picks the same option for the same
    candidate_id every run (reproducibility matters here; see model
    training script's own `deterministic=True` philosophy), while still
    varying phrasing across different candidates."""
    if not options:
        return ""
    h = int(hashlib.md5(seed_text.encode("utf-8")).hexdigest(), 16)
    return options[h % len(options)]


# ============================================================================
# Raw-fact extraction from ONE raw candidate JSON record (profile +
# redrob_signals + skills + education + career_history). Called only for
# the handful of candidates actually being narrated (the top-N), never for
# the full 100k pool -- see 12_predict_lgbm_cpu.py's
# load_raw_facts_for_ids().
# ============================================================================
def extract_raw_facts(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    p = candidate.get("profile", {}) or {}
    sig = candidate.get("redrob_signals", {}) or {}
    skills = candidate.get("skills", []) or []
    edu = candidate.get("education", []) or []
    history = candidate.get("career_history", []) or []

    top_skills = sorted(
        skills,
        key=lambda s: ({"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
                       .get((s.get("proficiency") or "").lower(), 0),
                       s.get("endorsements", 0) or 0),
        reverse=True,
    )[:3]

    cur_role = next((h for h in history if h.get("is_current")), None)
    top_edu = edu[0] if edu else None

    return {
        "years_of_experience": p.get("years_of_experience"),
        "location": p.get("location"),
        "country": p.get("country"),
        "current_title": p.get("current_title"),
        "current_company": p.get("current_company"),
        "current_industry": p.get("current_industry"),
        "headline": p.get("headline"),
        "notice_period_days": sig.get("notice_period_days"),
        "recruiter_response_rate": sig.get("recruiter_response_rate"),
        "last_active_date": sig.get("last_active_date"),
        "open_to_work_flag": sig.get("open_to_work_flag"),
        "willing_to_relocate": sig.get("willing_to_relocate"),
        "preferred_work_mode": sig.get("preferred_work_mode"),
        "github_activity_score": sig.get("github_activity_score"),
        "top_skill_names": [s.get("name") for s in top_skills if s.get("name")],
        "current_role_tenure_months": (cur_role or {}).get("duration_months"),
        "education_degree": (top_edu or {}).get("degree"),
        "education_field": (top_edu or {}).get("field_of_study"),
        "education_institution": (top_edu or {}).get("institution"),
    }


# ============================================================================
# Per-feature clause renderers. Each takes (feature_row, raw_facts,
# contribution_value) and returns a short clause fragment (no leading
# capital, no trailing period -- assembled later) or None to skip.
# `feature_row` is a Mapping[str, float] of THIS candidate's engineered
# feature values (already coerced, pre-categorical-shift values are fine
# here since we read straight from the features CSV, not the shifted
# training matrix).
# ============================================================================
FeatureRow = Mapping[str, Any]
RawFacts = Mapping[str, Any]
ClauseFn = "Callable[[FeatureRow, RawFacts, float], Optional[str]]"


def _experience_band(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    fit = row.get("experience_band_fit")
    yoe = raw.get("years_of_experience")
    if yoe is None:
        yoe = row.get("years_of_experience")
    if yoe is None or fit is None:
        return None
    score10 = max(0, min(10, round(float(fit) * 10)))
    if fit >= 0.85:
        desc = "squarely in the JD's 5-9yr target band"
    elif fit >= 0.6:
        desc = "close to the JD's 5-9yr target band"
    elif fit >= 0.35:
        desc = "noticeably outside the JD's 5-9yr target band"
    else:
        desc = "well outside the JD's 5-9yr target band"
    return f"{_fmt_num(yoe)} years of experience, {desc} (fit {score10}/10)"


def _location(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    tier = row.get("location_tier")
    if tier is None:
        return None
    tier = int(tier)
    loc = raw.get("location")
    loc_txt = f"based in {loc}" if loc else "their stated location"
    score10 = {3: 10, 2: 8, 1: 6, 0: 4, -1: 2}.get(tier, 4)
    if tier == 3:
        return f"{loc_txt}, one of the JD's two explicitly preferred cities (location fit {score10}/10)"
    if tier == 2:
        return f"{loc_txt}, a city the JD explicitly welcomes (location fit {score10}/10)"
    if tier == 1:
        return f"{loc_txt}, a Tier-1 Indian city but not one the JD names specifically (location fit {score10}/10)"
    if tier == 0:
        return f"{loc_txt} in India, outside the JD's named preferred/welcome cities (location fit {score10}/10)"
    relocate = raw.get("willing_to_relocate")
    base = f"{loc_txt}, outside India; the JD doesn't sponsor work visas and is case-by-case here"
    if relocate:
        base += ", though they've marked willingness to relocate"
    return base


def _availability(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    score = row.get("availability_tilt_score")
    if score is None:
        return None
    notice = raw.get("notice_period_days")
    if notice is None:
        notice = row.get("notice_period_days")
    resp = raw.get("recruiter_response_rate")
    if resp is None:
        resp = row.get("recruiter_response_rate")
    parts = []
    if notice is not None:
        parts.append(f"{int(notice)}-day notice period" + (" (within the JD's preferred window)" if notice <= 30 else " (above the JD's preferred 30-day window)"))
    if resp is not None:
        parts.append(f"responds to recruiters {round(float(resp) * 100)}% of the time")
    detail = ", ".join(parts) if parts else "behavioral signals"
    return f"availability/engagement score {round(float(score))}/10 ({detail})"


def _notice_period_only(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    notice = raw.get("notice_period_days", row.get("notice_period_days"))
    if notice is None:
        return None
    notice = int(notice)
    if notice > 30:
        return f"notice period of {notice} days exceeds the JD's preferred ≤30-day window"
    return f"a {notice}-day notice period, within the JD's preferred window"


def _recent_llm_only(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    if not row.get("recent_llm_only_experience_flag"):
        return None
    return "recent AI experience reads as LangChain/GPT-API-only without a clear pre-LLM ML background -- matches the JD's named disqualifier"


def _pre_llm_background(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    if not row.get("has_pre_llm_ml_background"):
        return None
    return "shows pre-LLM-era ML/IR background, which the JD explicitly values over recent-only LangChain experience"


def _pure_research(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    if not row.get("pure_research_flag"):
        return None
    return "profile reads as academic/research-only with no production-deployment language found -- the JD explicitly rules this out"


def _management_pivot(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    if not row.get("management_pivot_18mo_plus_flag"):
        return None
    title = raw.get("current_title")
    title_txt = f" ({title})" if title else ""
    return f"current role{title_txt} is management/architect-track and held 18+ months -- the JD is explicit this role writes code"


def _wrong_domain(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    if not row.get("wrong_domain_flag"):
        return None
    return "background reads as computer-vision/speech/robotics with no NLP or retrieval signal -- the JD flags this as a re-learning-fundamentals risk"


def _title_skill_mismatch(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    if not row.get("title_skill_mismatch_flag"):
        return None
    title = raw.get("current_title")
    title_txt = f"non-technical title ({title})" if title else "a non-technical current title"
    return f"{title_txt} paired with several AI/ML keywords in skills but no supporting narrative -- a possible keyword-stuffing pattern"


def _marketing_title(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    if row.get("title_skill_mismatch_flag") or not row.get("marketing_nontech_title_flag"):
        return None  # avoid double-narrating the same fact two ways
    title = raw.get("current_title")
    return f"current title ({title}) reads non-technical" if title else "current title reads non-technical"


def _pure_services(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    if not row.get("pure_services_career_flag"):
        return None
    return "entire career history is at IT-services/consulting firms -- the JD flags pure-services-only careers as a fit concern"


def _closed_source(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    if not row.get("closed_source_only_flag"):
        return None
    return "5+ years of experience with no GitHub activity and no open-source/publication signal anywhere -- the JD wants external validation of how someone thinks"


def _github(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    if row.get("closed_source_only_flag") or not row.get("has_github"):
        return None
    gh = raw.get("github_activity_score", row.get("github_activity_score"))
    if gh is None or gh < 0:
        return None
    return f"active GitHub presence (activity score {round(float(gh))}/100)"


def _title_chasing(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    if not row.get("title_chasing_flag"):
        return None
    n = row.get("title_chasing_jump_count")
    n_txt = f"{int(n)} " if n is not None else "multiple "
    return f"{n_txt}short-tenure (<18mo) seniority jumps in the last 5 years -- reads like the title-chasing pattern the JD screens out"


def _python(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    has_py = row.get("has_python_skill")
    if has_py is None:
        return None
    if not has_py:
        return "no Python listed in skills at all, despite the JD calling out Python specifically"
    prof = int(row.get("python_proficiency", 0) or 0)
    months = row.get("python_duration_months")
    prof_label = {4: "expert", 3: "advanced", 2: "intermediate", 1: "beginner"}.get(prof, "unspecified")
    months_txt = f", {int(months)} months" if months else ""
    return f"Python listed at {prof_label} proficiency{months_txt} -- the JD calls this out specifically"


def _education(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    degree = raw.get("education_degree")
    field = raw.get("education_field")
    tier = row.get("highest_education_tier_ord")
    if not degree and tier is None:
        return None
    tier_label = {4: "tier-1", 3: "tier-2", 2: "tier-3", 1: "tier-4", 0: "unranked"}.get(
        int(tier) if tier is not None else -1, "unranked"
    )
    if degree:
        field_txt = f" in {field}" if field else ""
        return f"{degree}{field_txt} from a {tier_label} institution"
    return f"highest education tier: {tier_label}"


def _assessment(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    v = row.get("assessment_score_jd_relevant_mean")
    n = row.get("num_assessments_taken")
    if not v or not n:
        return None
    return f"Redrob assessment scores for JD-relevant skills (retrieval/ranking/eval) average {round(float(v))}/100 across {int(n)} assessment(s)"


def _honeypot(row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    flag = row.get("honeypot_flag")
    if not flag:
        return None
    return "flagged by the deterministic honeypot rules (an internal profile inconsistency) -- treat this ranking with caution"


CURATED_RULES: Dict[str, ClauseFn] = {
    "experience_band_fit": _experience_band,
    "location_tier": _location,
    "availability_tilt_score": _availability,
    "notice_period_days": _notice_period_only,
    "recent_llm_only_experience_flag": _recent_llm_only,
    "has_pre_llm_ml_background": _pre_llm_background,
    "pure_research_flag": _pure_research,
    "management_pivot_18mo_plus_flag": _management_pivot,
    "wrong_domain_flag": _wrong_domain,
    "title_skill_mismatch_flag": _title_skill_mismatch,
    "marketing_nontech_title_flag": _marketing_title,
    "pure_services_career_flag": _pure_services,
    "closed_source_only_flag": _closed_source,
    "has_github": _github,
    "title_chasing_flag": _title_chasing,
    "has_python_skill": _python,
    "highest_education_tier_ord": _education,
    "assessment_score_jd_relevant_mean": _assessment,
    "honeypot_flag": _honeypot,
}

# Features that are an honest-disclosure signal regardless of how large
# their SHAP contribution is -- always eligible for the single "caution"
# slot if present, even if its |contribution| doesn't clear the relative
# driver threshold. Currently just the honeypot flag; see module docstring
# "NEVER FABRICATE" for why this is force-included rather than left to
# SHAP magnitude alone (a >10%-honeypots-in-top-100 disqualification is a
# harder failure mode than a slightly-off reasoning tone).
FORCE_INCLUDE_IF_TRUE = {"honeypot_flag"}


# ============================================================================
# Generic fallback for any keyword-family column not in CURATED_RULES.
# ============================================================================
def _family_clause(feature_name: str, row: FeatureRow) -> Optional[str]:
    for family, label in FAMILY_LABELS.items():
        if feature_name == f"{family}_narrative_hits":
            n = row.get(feature_name)
            if not n:
                return None
            tag = " (one of the JD's must-haves)" if family in JD_MUST_HAVE_FAMILIES else (
                " (a JD nice-to-have)" if family in JD_NICE_TO_HAVE_FAMILIES else "")
            return f"career narrative shows {int(n)} mention(s) of {label}{tag}"
        if feature_name == f"{family}_skills_hits":
            n = row.get(feature_name)
            if not n:
                return None
            return f"lists {int(n)} skills-list mention(s) of {label}"
        if feature_name == f"{family}_inflation_gap":
            n = row.get(feature_name)
            if not n:
                return None
            return f"claims {label} in the skills list with no supporting narrative ({int(n)}-mention gap) -- a possible keyword-stuffing signal"
    return None


def _fmt_value(x: Any) -> str:
    """Format a raw engineered-feature value for insertion into a fallback
    clause. Engineered features arrive here as whatever dtype pandas (or a
    plain dict) hands back -- including numpy.float64 with full native
    precision (e.g. 15.734824531905788) for anything that wasn't a clean
    integer count. Dumping that straight into reasoning text reads as
    unreviewed/unprofessional under Stage 4 manual review even when the
    number itself is perfectly legitimate -- this rounds it the way a
    human writing this sentence by hand would."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(f):
        return "unknown"
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.2f}"


def _generic_fallback(feature_name: str, row: FeatureRow, contrib: float) -> str:
    label = feature_name.replace("_flag", "").replace("_", " ")
    val = _fmt_value(row.get(feature_name))
    direction = "supports" if contrib >= 0 else "weighs against"
    return f"{label} ({val}) {direction} this ranking"


def render_clause(feature_name: str, row: FeatureRow, raw: RawFacts, contrib: float) -> Optional[str]:
    if feature_name in CURATED_RULES:
        clause = CURATED_RULES[feature_name](row, raw, contrib)
        if clause:
            return clause
        # Curated rule declined (flag not set / data missing) -- don't
        # fall through to the generic renderer for a feature we have a
        # specific opinion about; just skip it.
        return None
    fam_clause = _family_clause(feature_name, row)
    if fam_clause:
        return fam_clause
    return _generic_fallback(feature_name, row, contrib)


# ============================================================================
# Driver selection + assembly. Tracks (name, clause) pairs throughout (not
# bare strings) so the honeypot caution clause can be phrased differently
# from an ordinary trade-off caveat, and so length control can drop whole
# clauses rather than ever slicing mid-word.
# ============================================================================
def _select_clauses(
    feature_columns: Sequence[str],
    contributions: Sequence[float],
    row: FeatureRow,
    raw: RawFacts,
    max_pos: int,
    max_neg: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    pairs = list(zip(feature_columns, contributions))
    max_abs = max((abs(c) for _, c in pairs), default=0.0)
    threshold = max_abs * RELATIVE_DRIVER_THRESHOLD

    forced_names = [n for n in FORCE_INCLUDE_IF_TRUE if row.get(n)]
    pos_sorted = sorted((p for p in pairs if p[1] > threshold), key=lambda p: -p[1])
    neg_sorted = sorted((p for p in pairs if p[1] < -threshold), key=lambda p: p[1])

    used: set = set()
    pos_clauses: List[Tuple[str, str]] = []
    for name, c in pos_sorted:
        if name in used:
            continue
        clause = render_clause(name, row, raw, c)
        if clause:
            pos_clauses.append((name, clause))
            used.add(name)
        if len(pos_clauses) >= max_pos:
            break

    # Forced (honeypot-style) clauses always take the first negative
    # slot(s); ordinary negative drivers fill whatever's left.
    neg_candidates = [(n, 0.0) for n in forced_names if n not in used] + [
        p for p in neg_sorted if p[0] not in used
    ]
    neg_clauses: List[Tuple[str, str]] = []
    for name, c in neg_candidates:
        if name in used:
            continue
        clause = render_clause(name, row, raw, c)
        if clause:
            neg_clauses.append((name, clause))
            used.add(name)
        if len(neg_clauses) >= max_neg:
            break

    return pos_clauses, neg_clauses


def _assemble(
    pos_clauses: List[Tuple[str, str]], neg_clauses: List[Tuple[str, str]],
    raw: RawFacts, row: FeatureRow, candidate_id: str,
) -> str:
    if not pos_clauses and not neg_clauses:
        yoe = raw.get("years_of_experience", row.get("years_of_experience"))
        title = raw.get("current_title")
        bits = []
        if yoe is not None:
            bits.append(f"{_fmt_num(yoe)} years of experience")
        if title:
            bits.append(f"currently {title}")
        base = ", ".join(bits) if bits else "no single feature stood out strongly"
        return (
            f"Adjacent fit at best: {base}; no clearly dominant strength or "
            f"weakness in the engineered signals for this profile."
        )

    honeypot_clause = next((c for n, c in neg_clauses if n == "honeypot_flag"), None)
    other_neg = [c for n, c in neg_clauses if n != "honeypot_flag"]

    if not pos_clauses:
        body = "; ".join(c for _, c in neg_clauses)
        return f"Below the JD's core profile: {body}."

    sep = _stable_choice(candidate_id, CONNECTORS)
    sentence1 = sep.join(c for _, c in pos_clauses)
    sentence1 = sentence1[0].upper() + sentence1[1:] + "."
    text = sentence1

    if other_neg:
        lead_in = _stable_choice(candidate_id, ["However, ", "That said, ", "On the downside, "])
        text += f" {lead_in}{'; '.join(other_neg)}."
    if honeypot_clause:
        text += f" Note: {honeypot_clause}."

    return text


def build_reasoning(
    feature_columns: Sequence[str],
    contributions: Sequence[float],
    row: FeatureRow,
    raw_facts: Optional[RawFacts],
    candidate_id: str,
) -> str:
    """Main entry point. `row` is this candidate's engineered-feature
    values (dict-like); `raw_facts` is the output of extract_raw_facts()
    for this candidate, or None if no --candidates-for-reasoning file was
    supplied (degrades gracefully to engineered-feature-only phrasing).
    `contributions` must be aligned 1:1 with `feature_columns` (the
    non-bias slice of a pred_contrib=True row, averaged across ensemble
    members if applicable).

    Length control never slices the final string (that risks cutting a
    clause mid-word) -- if the fullest version overflows
    MAX_REASONING_CHARS, the function retries with progressively fewer
    clauses, always keeping the honeypot caution (if any) and at least
    one positive clause.
    """
    raw = raw_facts or {}

    for max_pos, max_neg in (
        (MAX_POSITIVE_CLAUSES, MAX_NEGATIVE_CLAUSES + 1),
        (2, MAX_NEGATIVE_CLAUSES + 1),
        (1, 1),
    ):
        pos_clauses, neg_clauses = _select_clauses(
            feature_columns, contributions, row, raw, max_pos, max_neg
        )
        text = _assemble(pos_clauses, neg_clauses, raw, row, candidate_id)
        if len(text) <= MAX_REASONING_CHARS:
            return text

    return text  # last attempt is the most compact; accept it even if long