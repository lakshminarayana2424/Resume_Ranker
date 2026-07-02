#!/usr/bin/env python3
"""
feature_engineering.py
=======================

OVERVIEW:
This module provides shared, deterministic feature computation for the LightGBM student ranker. 
It relies entirely on standard Python libraries (NO LLM, NO GPU, NO NETWORK) for high-speed, 
CPU-bound execution suitable for processing large datasets within strict time limits (e.g., hackathons).

KEY FUNCTIONALITY:
1. Train/Inference Consistency: By isolating feature logic here, we prevent train/inference skew. 
   Both the dataset builder and the final ranking script (e.g., rank.py) must import 
   `build_feature_row` and `FEATURE_COLUMNS` from this file.
2. Signal Extraction: Instead of hand-coding a composite score, this script extracts granular 
   signals (keyword counts, tenure lengths, location tiers, behavioral patterns) and lets the 
   gradient boosting model learn the optimal weights.
3. Keyword-Stuffer Countermeasure: It explicitly calculates the gap between self-reported skills 
   and the free-text career narrative. This exposes "claims-without-narrative-support" to penalize 
   candidates who stuff their profiles with keywords.
4. Deterministic Time Features: Time-based features use a dataset-derived reference date 
   (`derive_reference_date`) rather than `datetime.now()`. This ensures that processing the same 
   static snapshot always yields the exact same features regardless of the wall-clock time it runs.
5. Docker/Path Compatibility: This file acts purely as an imported module. It has zero hardcoded 
   paths or I/O steps. Keep it in your root execution directory for seamless Docker integration.
"""

import re
import statistics
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

LLM_WAVE_CUTOFF = date(2022, 11, 1)
RECENT_WINDOW_DAYS = 365

COMPANY_SIZE_ORD: Dict[str, int] = {
    "1-10": 0, "11-50": 1, "51-200": 2, "201-500": 3,
    "501-1000": 4, "1001-5000": 5, "5001-10000": 6, "10001+": 7,
}

PROFICIENCY_ORD: Dict[str, int] = {
    "beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4,
}

EDU_TIER_ORD: Dict[str, int] = {
    "tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0,
}

WORK_MODE_ORD: Dict[str, int] = {
    "remote": 0, "hybrid": 1, "onsite": 2, "flexible": 3,
}

CATEGORICAL_COLUMNS = [
    "current_company_size_ord",
    "preferred_work_mode_ord",
    "highest_education_tier_ord",
    "current_title_seniority_level",
    "location_tier",
]

def _compiled(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]

KEYWORD_FAMILIES: Dict[str, List[str]] = {
    "retrieval_embeddings": [
        r"\bembeddings?\b", r"\bsentence[\s\-]?transformers?\b", r"\bbge\b",
        r"\be5\b", r"\bdense retrieval\b", r"\bsemantic search\b",
        r"\bbi[\s\-]?encoders?\b", r"\bcross[\s\-]?encoders?\b", r"\brag\b",
        r"retrieval[\s\-]?augmented", r"\bvector search\b",
        r"openai embeddings?", r"\bword2vec\b", r"\bdoc2vec\b",
        r"\bann\b.{0,15}\bsearch\b", r"approximate nearest neighbou?r",
    ],
    "vector_db_hybrid_search": [
        r"\bpinecone\b", r"\bweaviate\b", r"\bqdrant\b", r"\bmilvus\b",
        r"\bfaiss\b", r"\bopensearch\b", r"\belasticsearch\b",
        r"\bhybrid search\b", r"vector database", r"\bann index\b",
    ],
    "evaluation_ranking_metrics": [
        r"\bndcg\b", r"\bmrr\b", r"mean average precision", r"\bmap@\d+",
        r"precision@\d+", r"\bp@\d+\b", r"\ba/b test", r"\bab test",
        r"offline.{0,20}online correlation", r"offline evaluation",
        r"recall@\d+", r"learning[\s\-]to[\s\-]rank",
    ],
    "llm_finetuning": [
        r"\block?ra\b", r"\bq?lora\b", r"\bpeft\b", r"adapter tuning",
        r"fine[\s\-]?tun(e|ing|ed)",
    ],
    "learning_to_rank_models": [
        r"lambdamart", r"\bxgboost\b.{0,20}rank", r"listwise rank",
        r"pairwise rank", r"neural ranker", r"learning[\s\-]to[\s\-]rank",
    ],
    "distributed_inference": [
        r"\bkubernetes\b", r"\bk8s\b", r"multi[\s\-]?gpu", r"\btriton\b",
        r"\bvllm\b", r"\btensorrt\b", r"model serving", r"inference optimi[sz]ation",
        r"\bqps\b", r"latency.{0,15}(slo|sla|optimi[sz])",
    ],
    "open_source_validation": [
        r"open[\s\-]?source", r"\bgithub\b", r"published a paper",
        r"\bconference talk\b", r"tech talk", r"\bblog post\b",
        r"contributed to", r"maintainer of",
    ],
    "nlp_ir_domain": [
        r"\bnlp\b", r"natural language processing", r"information retrieval",
        r"\bsearch ranking\b", r"\btext classification\b", r"\btopic model",
        r"\bnamed entity\b", r"\bner\b",
    ],
    "cv_speech_robotics_domain": [
        r"computer vision", r"\bopencv\b", r"image classification",
        r"object detection", r"speech recognition", r"\basr\b",
        r"\brobotics?\b", r"autonomous (vehicle|driving|navigation)",
        r"\blidar\b", r"\bslam\b",
    ],
    "production_deployment": [
        r"\bproduction\b", r"\bdeployed?\b", r"shipped to", r"real users",
        r"at scale", r"\bmillions of (users|queries|requests)\b",
        r"\blive system\b", r"\btraffic\b",
    ],
    "research_only": [
        r"\bpostdoc\b", r"\bph\.?d\.?\b", r"research scientist",
        r"academic lab", r"\bacademia\b", r"published.{0,15}paper",
    ],
    "pre_llm_ml_background": [
        r"machine learning", r"deep learning", r"recommendation system",
        r"\bclassifier\b", r"\bregression model\b", r"search ranking",
        r"\bnlp\b", r"computer vision",
    ],
    "llm_only_recent": [
        r"\blangchain\b", r"\bopenai api\b", r"\bgpt[\s\-]?[34]\b",
        r"\bchatgpt\b", r"\banthropic\b", r"\bllm\b",
    ],
    "code_quality_signals": [
        r"code review", r"\bci/cd\b", r"continuous integration",
        r"unit test", r"\btest coverage\b", r"\bopen[\s\-]?source\b",
    ],
    "hr_recruiting_marketplace_domain": [
        r"\bhr tech\b", r"recruit(ing|ment)", r"\btalent\b",
        r"\bmarketplace\b", r"two[\s\-]?sided", r"\bats\b",
        r"job board", r"candidate matching",
    ],
    "marketing_nontech_titles": [
        r"\bmarketing\b", r"\bsales\b", r"account manager",
        r"\bhuman resources\b", r"\bhr\b generalist", r"\bfinance\b",
        r"business analyst",
    ],
    "management_architect_titles": [
        r"\barchitect\b", r"\btech lead\b", r"\bhead of\b", r"\bdirector\b",
        r"\bvp\b", r"\bmanager\b", r"\bengineering manager\b",
    ],
    "title_chasing_seniority_words": [
        r"\bsenior\b", r"\bstaff\b", r"\bprincipal\b", r"\blead\b",
    ],
}

COMPILED_KEYWORDS: Dict[str, List[re.Pattern]] = {
    k: _compiled(v) for k, v in KEYWORD_FAMILIES.items()
}

SERVICE_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hexaware", "mphasis", "hcl", "tech mahindra", "mindtree",
    "ltimindtree", "l&t infotech",
]

TECH_INDUSTRY_KEYWORDS = [
    "software", "internet", "saas", "information technology", "ai", "ml",
    "fintech", "edtech", "e-commerce", "ecommerce", "technology",
]
SERVICES_INDUSTRY_KEYWORDS = [
    "it services", "consulting", "staffing", "outsourcing", "bpo",
]

PREFERRED_CITIES = {"pune", "noida"}
WELCOME_CITIES = {
    "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "ncr",
    "delhi ncr", "new delhi",
}
OTHER_TIER1_CITIES = {"bangalore", "bengaluru", "chennai", "kolkata"}

def parse_date(s: Optional[str]) -> Optional[date]:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None

def safe_lower(s: Any) -> str:
    return str(s).lower() if s is not None else ""

def count_hits(text: str, patterns: List[re.Pattern]) -> int:
    if not text:
        return 0
    return sum(1 for p in patterns if p.search(text))

def any_hit(text: str, patterns: List[re.Pattern]) -> bool:
    return count_hits(text, patterns) > 0

def days_between(d1: Optional[date], d2: Optional[date]) -> Optional[int]:
    if d1 is None or d2 is None:
        return None
    return (d1 - d2).days

def derive_reference_date(candidates_iterable) -> date:
    max_d: Optional[date] = None

    def consider(d: Optional[date]):
        nonlocal max_d
        if d is not None and (max_d is None or d > max_d):
            max_d = d

    for c in candidates_iterable:
        sig = c.get("redrob_signals", {}) or {}
        consider(parse_date(sig.get("signup_date")))
        consider(parse_date(sig.get("last_active_date")))
        for h in (c.get("career_history") or []):
            consider(parse_date(h.get("start_date")))
            consider(parse_date(h.get("end_date")))

    if max_d is None:
        max_d = date(2026, 1, 1)
    return max_d

class CandidateText:
    def __init__(self, c: Dict[str, Any]):
        p = c.get("profile", {}) or {}
        history = c.get("career_history") or []
        skills = c.get("skills") or []

        self.skills_text = " ; ".join(safe_lower(s.get("name", "")) for s in skills)
        self.skill_names = {safe_lower(s.get("name", "")) for s in skills}

        narrative_parts = [
            safe_lower(p.get("headline", "")),
            safe_lower(p.get("summary", "")),
        ]
        for h in history:
            narrative_parts.append(safe_lower(h.get("description", "")))
        self.narrative_text = " ; ".join(narrative_parts)

        title_parts = [safe_lower(p.get("current_title", ""))]
        for h in history:
            title_parts.append(safe_lower(h.get("title", "")))
        self.titles_text = " ; ".join(title_parts)

        industry_parts = [safe_lower(p.get("current_industry", ""))]
        for h in history:
            industry_parts.append(safe_lower(h.get("industry", "")))
        self.industries_text = " ; ".join(industry_parts)

        company_parts = [safe_lower(p.get("current_company", ""))]
        for h in history:
            company_parts.append(safe_lower(h.get("company", "")))
        self.companies_text = " ; ".join(company_parts)

        self.full_text = " ; ".join([
            self.skills_text, self.narrative_text, self.titles_text,
            self.industries_text,
        ])

def keyword_family_features(ct: CandidateText, family: str) -> Dict[str, int]:
    patterns = COMPILED_KEYWORDS[family]
    narrative_hits = count_hits(ct.narrative_text, patterns)
    skills_hits = count_hits(ct.skills_text, patterns)
    inflation_gap = max(0, skills_hits - narrative_hits) if skills_hits else 0
    return {
        f"{family}_narrative_hits": narrative_hits,
        f"{family}_skills_hits": skills_hits,
        f"{family}_inflation_gap": inflation_gap,
    }

_SENIORITY_LEVELS: List[Tuple[int, List[str]]] = [
    (1, [r"\bintern\b", r"\bjunior\b", r"\bassociate\b", r"\btrainee\b"]),
    (3, [r"\bsenior\b", r"\bsr\.?\b"]),
    (4, [r"\bstaff\b", r"\blead\b", r"\bprincipal\b"]),
    (5, [r"\barchitect\b", r"\bdirector\b", r"\bhead of\b", r"\bvp\b",
         r"\bmanager\b", r"\bengineering manager\b"]),
]
_SENIORITY_PATTERNS = [(lvl, _compiled(pats)) for lvl, pats in _SENIORITY_LEVELS]

def extract_seniority_level(title: str) -> int:
    if not title:
        return 2
    best = 2
    found_any = False
    for lvl, patterns in _SENIORITY_PATTERNS:
        if any(p.search(title) for p in patterns):
            best = max(best, lvl) if found_any else lvl
            found_any = True
    return best if found_any else 2

def location_tier(location: str, country: str) -> int:
    loc = safe_lower(location)
    ctry = safe_lower(country)
    if any(c in loc for c in PREFERRED_CITIES):
        return 3
    if any(c in loc for c in WELCOME_CITIES):
        return 2
    if any(c in loc for c in OTHER_TIER1_CITIES):
        return 1
    if "india" in ctry or ctry == "in":
        return 0
    return -1

def experience_band_fit(yoe: Optional[float]) -> float:
    if yoe is None:
        return 0.0
    if 5 <= yoe <= 9:
        return 1.0
    if yoe < 5:
        return max(0.0, 1.0 - (5 - yoe) / 5.0)
    return max(0.0, 1.0 - (yoe - 9) / 10.0)

def redrob_signal_features(c: Dict[str, Any], reference_date: date) -> Dict[str, Any]:
    sig = c.get("redrob_signals", {}) or {}
    out: Dict[str, Any] = {}

    out["profile_completeness_score"] = sig.get("profile_completeness_score", 0.0)

    signup_d = parse_date(sig.get("signup_date"))
    active_d = parse_date(sig.get("last_active_date"))
    out["days_on_platform"] = days_between(reference_date, signup_d) or 0
    out["days_since_active"] = days_between(reference_date, active_d)
    if out["days_since_active"] is None:
        out["days_since_active"] = 99999

    out["open_to_work_flag"] = int(bool(sig.get("open_to_work_flag", False)))
    out["profile_views_received_30d"] = sig.get("profile_views_received_30d", 0)
    out["applications_submitted_30d"] = sig.get("applications_submitted_30d", 0)
    out["recruiter_response_rate"] = sig.get("recruiter_response_rate", 0.0)
    out["avg_response_time_hours"] = sig.get("avg_response_time_hours", 0.0)

    assess = sig.get("skill_assessment_scores", {}) or {}
    out["num_assessments_taken"] = len(assess)
    out["assessment_score_mean"] = (
        sum(assess.values()) / len(assess) if assess else 0.0
    )
    out["assessment_score_max"] = max(assess.values()) if assess else 0.0
    
    relevant_terms = (
        COMPILED_KEYWORDS["retrieval_embeddings"]
        + COMPILED_KEYWORDS["vector_db_hybrid_search"]
        + COMPILED_KEYWORDS["evaluation_ranking_metrics"]
        + COMPILED_KEYWORDS["learning_to_rank_models"]
    )
    relevant_assess_scores = [
        v for k, v in assess.items() if any_hit(safe_lower(k), relevant_terms)
    ]
    out["assessment_score_jd_relevant_mean"] = (
        sum(relevant_assess_scores) / len(relevant_assess_scores)
        if relevant_assess_scores else 0.0
    )

    out["connection_count"] = sig.get("connection_count", 0)
    out["endorsements_received"] = sig.get("endorsements_received", 0)
    out["notice_period_days"] = sig.get("notice_period_days", 0)

    sal = sig.get("expected_salary_range_inr_lpa", {}) or {}
    sal_min = sal.get("min", 0.0) or 0.0
    sal_max = sal.get("max", 0.0) or 0.0
    out["expected_salary_min_lpa"] = sal_min
    out["expected_salary_max_lpa"] = sal_max
    out["expected_salary_mid_lpa"] = (sal_min + sal_max) / 2.0
    out["expected_salary_width_lpa"] = max(0.0, sal_max - sal_min)

    out["preferred_work_mode_ord"] = WORK_MODE_ORD.get(
        sig.get("preferred_work_mode", ""), -1
    )
    out["willing_to_relocate"] = int(bool(sig.get("willing_to_relocate", False)))

    gh = sig.get("github_activity_score", -1)
    out["has_github"] = int(gh is not None and gh >= 0)
    out["github_activity_score"] = gh if gh is not None else -1

    out["search_appearance_30d"] = sig.get("search_appearance_30d", 0)
    out["saved_by_recruiters_30d"] = sig.get("saved_by_recruiters_30d", 0)
    out["interview_completion_rate"] = sig.get("interview_completion_rate", 0.0)

    oar = sig.get("offer_acceptance_rate", -1)
    out["has_offer_history"] = int(oar is not None and oar >= 0)
    out["offer_acceptance_rate"] = oar if oar is not None else -1

    out["verified_email"] = int(bool(sig.get("verified_email", False)))
    out["verified_phone"] = int(bool(sig.get("verified_phone", False)))
    out["linkedin_connected"] = int(bool(sig.get("linkedin_connected", False)))

    avail = 0
    if out["notice_period_days"] <= 30 and out["open_to_work_flag"]:
        avail += 2
    if out["recruiter_response_rate"] >= 0.6:
        avail += 1
    if out["days_since_active"] <= 30:
        avail += 1
    if out["notice_period_days"] > 60:
        avail -= 1
    if (out["days_since_active"] > 180 and not out["open_to_work_flag"]
            and out["recruiter_response_rate"] < 0.2):
        avail -= 2
    out["availability_tilt_score"] = max(0, min(10, avail + 5)) 

    return out

def career_features(c: Dict[str, Any], ct: CandidateText, reference_date: date) -> Dict[str, Any]:
    history = c.get("career_history") or []
    out: Dict[str, Any] = {}

    out["num_career_entries"] = len(history)

    durations = [h.get("duration_months", 0) or 0 for h in history]
    out["avg_tenure_months"] = sum(durations) / len(durations) if durations else 0.0
    out["min_tenure_months"] = min(durations) if durations else 0.0
    out["max_tenure_months"] = max(durations) if durations else 0.0
    out["tenure_stdev_months"] = (
        statistics.pstdev(durations) if len(durations) > 1 else 0.0
    )

    sizes = [COMPANY_SIZE_ORD.get(h.get("company_size", ""), -1) for h in history]
    sizes = [s for s in sizes if s >= 0]
    out["avg_company_size_ord"] = sum(sizes) / len(sizes) if sizes else -1.0

    service_months = 0
    total_months = 0
    all_service = True if history else False
    
    for h in history:
        dur = h.get("duration_months", 0) or 0
        total_months += dur
        company = safe_lower(h.get("company", ""))
        industry = safe_lower(h.get("industry", ""))
        is_service = (
            any(f in company for f in SERVICE_FIRMS)
            or any(k in industry for k in SERVICES_INDUSTRY_KEYWORDS)
        )
        if is_service:
            service_months += dur
        else:
            all_service = False
            
    out["service_firm_months"] = service_months
    out["service_firm_month_fraction"] = (
        service_months / total_months if total_months else 0.0
    )
    out["pure_services_career_flag"] = int(all_service)

    cur_industry = safe_lower((c.get("profile") or {}).get("current_industry", ""))
    out["current_industry_is_tech"] = int(
        any(k in cur_industry for k in TECH_INDUSTRY_KEYWORDS)
    )
    out["current_industry_is_services"] = int(
        any(k in cur_industry for k in SERVICES_INDUSTRY_KEYWORDS)
    )

    dated = []
    for h in history:
        sd = parse_date(h.get("start_date"))
        if sd is not None:
            dated.append((sd, h))
    dated.sort(key=lambda x: x[0])

    jump_count = 0
    five_years_ago = reference_date.replace(year=reference_date.year - 5)
    
    for i in range(len(dated) - 1):
        sd_i, h_i = dated[i]
        sd_j, h_j = dated[i + 1]
        if sd_j < five_years_ago:
            continue
        tenure_i = h_i.get("duration_months", 0) or 0
        lvl_i = extract_seniority_level(safe_lower(h_i.get("title", "")))
        lvl_j = extract_seniority_level(safe_lower(h_j.get("title", "")))
        if tenure_i < 18 and lvl_j > lvl_i:
            jump_count += 1
            
    out["title_chasing_jump_count"] = jump_count
    out["title_chasing_flag"] = int(jump_count >= 3)

    if dated:
        earliest = dated[0][0]
        span_years = (reference_date - earliest).days / 365.25
    else:
        span_years = 0.0
        
    out["computed_career_span_years"] = span_years
    stated_yoe = (c.get("profile") or {}).get("years_of_experience", 0.0) or 0.0
    out["years_experience_consistency_gap"] = abs(stated_yoe - span_years)

    has_recent_llm_only_role = False
    has_pre_llm_ml_role = False
    
    for sd, h in dated:
        desc = safe_lower(h.get("description", "")) + " " + safe_lower(h.get("title", ""))
        is_recent = (reference_date - sd).days <= (RECENT_WINDOW_DAYS)
        if is_recent and any_hit(desc, COMPILED_KEYWORDS["llm_only_recent"]):
            has_recent_llm_only_role = True
        if sd < LLM_WAVE_CUTOFF and any_hit(desc, COMPILED_KEYWORDS["pre_llm_ml_background"]):
            has_pre_llm_ml_role = True
            
    out["recent_llm_only_experience_flag"] = int(
        has_recent_llm_only_role and not has_pre_llm_ml_role
    )
    out["has_pre_llm_ml_background"] = int(has_pre_llm_ml_role)

    research_hits = count_hits(ct.titles_text + " " + ct.industries_text, COMPILED_KEYWORDS["research_only"])
    production_hits = count_hits(ct.narrative_text, COMPILED_KEYWORDS["production_deployment"])
    out["pure_research_flag"] = int(research_hits > 0 and production_hits == 0)

    cur_title = safe_lower((c.get("profile") or {}).get("current_title", ""))
    cur_role = next((h for h in history if h.get("is_current")), None)
    cur_tenure = cur_role.get("duration_months", 0) if cur_role else 0
    is_mgmt_title = any_hit(cur_title, COMPILED_KEYWORDS["management_architect_titles"])
    out["management_pivot_18mo_plus_flag"] = int(is_mgmt_title and cur_tenure >= 18)

    cv_hits = count_hits(ct.full_text, COMPILED_KEYWORDS["cv_speech_robotics_domain"])
    nlp_hits = count_hits(ct.full_text, COMPILED_KEYWORDS["nlp_ir_domain"])
    retrieval_hits = count_hits(ct.full_text, COMPILED_KEYWORDS["retrieval_embeddings"])
    out["wrong_domain_flag"] = int(cv_hits > 0 and nlp_hits == 0 and retrieval_hits == 0)

    return out

def skill_features(c: Dict[str, Any]) -> Dict[str, Any]:
    skills = c.get("skills") or []
    out: Dict[str, Any] = {}
    
    out["num_skills"] = len(skills)
    profs = [PROFICIENCY_ORD.get(s.get("proficiency", ""), 0) for s in skills]
    out["avg_skill_proficiency"] = sum(profs) / len(profs) if profs else 0.0
    out["max_skill_proficiency"] = max(profs) if profs else 0.0

    total_endorsements = sum(s.get("endorsements", 0) or 0 for s in skills)
    out["total_skill_endorsements"] = total_endorsements

    suspicious = 0
    for s in skills:
        prof = PROFICIENCY_ORD.get(s.get("proficiency", ""), 0)
        dur = s.get("duration_months", 0) or 0
        if prof >= 3 and dur < 6: 
            suspicious += 1
    out["high_proficiency_low_duration_count"] = suspicious

    python_entry = next(
        (s for s in skills if safe_lower(s.get("name", "")) == "python"), None
    )
    if python_entry:
        out["has_python_skill"] = 1
        out["python_proficiency"] = PROFICIENCY_ORD.get(
            python_entry.get("proficiency", ""), 0
        )
        out["python_duration_months"] = python_entry.get("duration_months", 0) or 0
    else:
        out["has_python_skill"] = 0
        out["python_proficiency"] = 0
        out["python_duration_months"] = 0

    return out

def education_features(c: Dict[str, Any]) -> Dict[str, Any]:
    edu = c.get("education") or []
    out: Dict[str, Any] = {}
    
    out["num_education_entries"] = len(edu)
    tiers = [EDU_TIER_ORD.get(e.get("tier", "unknown"), 0) for e in edu]
    out["highest_education_tier_ord"] = max(tiers) if tiers else 0

    relevant_fields = ["computer science", "information technology", "data science",
                        "electronics", "mathematics", "statistics", "artificial intelligence"]
    out["has_relevant_degree_field"] = int(any(
        any(rf in safe_lower(e.get("field_of_study", "")) for rf in relevant_fields)
        for e in edu
    ))

    certs = c.get("certifications") or []
    out["num_certifications"] = len(certs)
    cert_text = " ".join(
        safe_lower(x.get("name", "")) + " " + safe_lower(x.get("issuer", ""))
        for x in certs
    )
    out["has_relevant_certification"] = int(any_hit(
        cert_text, COMPILED_KEYWORDS["retrieval_embeddings"]
        + COMPILED_KEYWORDS["learning_to_rank_models"]
    ) or any(k in cert_text for k in ["aws", "gcp", "google cloud", "azure", "deep learning"]))

    return out

def jd_fit_features(c: Dict[str, Any], ct: CandidateText) -> Dict[str, Any]:
    p = c.get("profile") or {}
    out: Dict[str, Any] = {}

    yoe = p.get("years_of_experience", 0.0) or 0.0
    out["years_of_experience"] = yoe
    out["experience_band_fit"] = experience_band_fit(yoe)

    out["location_tier"] = location_tier(p.get("location", ""), p.get("country", ""))
    out["current_company_size_ord"] = COMPANY_SIZE_ORD.get(
        p.get("current_company_size", ""), -1
    )

    out["current_title_seniority_level"] = extract_seniority_level(
        safe_lower(p.get("current_title", ""))
    )

    out["marketing_nontech_title_flag"] = int(any_hit(
        safe_lower(p.get("current_title", "")),
        COMPILED_KEYWORDS["marketing_nontech_titles"],
    ))

    nontech_title = out["marketing_nontech_title_flag"]
    ai_skill_hits = sum(
        count_hits(ct.skills_text, COMPILED_KEYWORDS[fam])
        for fam in ("retrieval_embeddings", "vector_db_hybrid_search",
                    "learning_to_rank_models", "llm_finetuning")
    )
    ai_narrative_hits = sum(
        count_hits(ct.narrative_text, COMPILED_KEYWORDS[fam])
        for fam in ("retrieval_embeddings", "vector_db_hybrid_search",
                    "learning_to_rank_models", "llm_finetuning")
    )
    out["title_skill_mismatch_flag"] = int(
        nontech_title and ai_skill_hits >= 3 and ai_narrative_hits == 0
    )

    for fam in KEYWORD_FAMILIES:
        out.update(keyword_family_features(ct, fam))

    return out

def build_feature_row(c: Dict[str, Any], reference_date: date) -> Dict[str, Any]:
    candidate_id = c.get("candidate_id", "")
    ct = CandidateText(c)

    row: Dict[str, Any] = {"candidate_id": candidate_id}
    row.update(redrob_signal_features(c, reference_date))
    row.update(career_features(c, ct, reference_date))
    row.update(skill_features(c))
    row.update(education_features(c))
    row.update(jd_fit_features(c, ct))

    oss_hits = count_hits(ct.full_text, COMPILED_KEYWORDS["open_source_validation"])
    row["closed_source_only_flag"] = int(
        row["years_of_experience"] >= 5
        and row["has_github"] == 0
        and oss_hits == 0
    )

    return row

def _dummy_candidate() -> Dict[str, Any]:
    return {
        "candidate_id": "CAND_0000000",
        "profile": {
            "headline": "", "summary": "", "location": "", "country": "",
            "years_of_experience": 0, "current_title": "", "current_company": "",
            "current_company_size": "1-10", "current_industry": "",
        },
        "career_history": [{
            "company": "", "title": "", "start_date": "2020-01-01",
            "end_date": None, "duration_months": 12, "is_current": True,
            "industry": "", "company_size": "1-10", "description": "",
        }],
        "education": [],
        "skills": [],
        "certifications": [],
        "redrob_signals": {
            "profile_completeness_score": 0, "signup_date": "2020-01-01",
            "last_active_date": "2020-01-01", "open_to_work_flag": False,
            "profile_views_received_30d": 0, "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.0, "avg_response_time_hours": 0.0,
            "skill_assessment_scores": {}, "connection_count": 0,
            "endorsements_received": 0, "notice_period_days": 0,
            "expected_salary_range_inr_lpa": {"min": 0, "max": 0},
            "preferred_work_mode": "remote", "willing_to_relocate": False,
            "github_activity_score": -1, "search_appearance_30d": 0,
            "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.0,
            "offer_acceptance_rate": -1, "verified_email": False,
            "verified_phone": False, "linkedin_connected": False,
        },
    }

FEATURE_COLUMNS: List[str] = [
    k for k in build_feature_row(_dummy_candidate(), date(2026, 1, 1)).keys()
    if k != "candidate_id"
]