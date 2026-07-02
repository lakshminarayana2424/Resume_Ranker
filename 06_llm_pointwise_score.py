#!/usr/bin/env python3
"""Step 6 — pointwise LLM scoring for the top N candidates.

This stage is the LLM-based reranking step in the pipeline. It takes the already
ranked candidate list from the cross-encoder + honeypot filters, loads the full
candidate profiles, and scores only the top N candidates with a structured
pointwise rubric. The goal is to produce a more human-like ranking signal than
the cross-encoder alone, while still keeping the scoring deterministic,
resumable, and easy to audit.

What this script does:

* Reads the ranked candidate IDs from the previous stage.
* Loads the corresponding full candidate profiles from the source pool.
* Renders each candidate into a single text block that the LLM can score.
* Scores each candidate using a five-part rubric instead of one vague overall score.
* Applies a hard-disqualifier check before normal scoring.
* Saves every completed scoring pass to a cache so interrupted runs can resume.
* Writes the final reranked output after averaging any repeated passes.

Why this stage is different:
A single holistic 0–100 LLM score is unstable because it gives the model too much
freedom. Two similar candidates can drift apart simply because the prompt is long
or the response is slightly inconsistent. This script reduces that variance by
splitting the decision into narrow dimensions with clear anchors:
retrieval depth, evaluation rigor, applied ML engineering, credibility, and
bonus/availability signals.

The scoring rubric is also designed to behave like a real recruiter:

* It checks hard disqualifiers first.
* It caps obvious mismatches even if other areas look strong.
* It requires a short rationale grounded in actual profile facts.
* It outputs JSON only, so the result can be parsed and reused downstream.

Resilience and resume behavior:
The script writes each successful pass into a JSONL cache. If the run stops part
way through, the next run skips candidates that were already scored. If the LLM
returns malformed text, the script extracts the JSON object, validates the
fields, clamps scores to the allowed ranges, and falls back safely if needed.
That makes the stage robust enough for a hackathon pipeline where reruns are
common.

Docker and path behavior:
The default model path is resolved relative to the script location so the same
command works from the repo root, the parent directory, or inside Docker. The
documented command in the usage block is the intended way to run the script.

Usage:
python 06_llm_pointwise_score.py 
--ranked  outputs/cross_encoder_ranked_honeypot.csv 
--candidates candidates.jsonl 
--top-n 30000 
--out outputs/llm_pointwise_top30000.csv 
--model-name ./models/Qwen3-8B-AWQ 
--num-gpus 1 
--max-model-len 6144 
--gpu-memory-utilization 0.75 
--enforce-eager 
--batch-size 1
"""


import argparse
import csv
import gzip
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCORING_RUBRIC = """\
You are an expert technical recruiter scoring a single candidate profile for a \
Senior AI Engineer position at Redrob AI (Series A, Pune/Noida). Your output \
must be a single JSON object — nothing else, no markdown, no preamble.

═══════════════════════════════════════════════════════════════
STEP 0 — DISQUALIFIER CHECK (run this before scoring anything)
═══════════════════════════════════════════════════════════════
Examine the profile for any of these HARD DISQUALIFIERS.
If ANY apply: set hard_disqualifier=true and disqualifier_reason to the first
one that matches. The total_score must then be at most 20 regardless of
subscores. If none apply: set hard_disqualifier=false and disqualifier_reason=null.

HARD DISQUALIFIERS:
  A. PURE RESEARCH ONLY — entire career in academic labs or research-only roles
     with zero production deployments to real users at any point.
  B. LLM-ONLY AI EXPERIENCE — all AI/ML work is under 12 months old AND
     consists of calling OpenAI/Anthropic/LangChain APIs with no prior
     production ML background predating the LLM wave (pre-2022 ML work at
     product companies is the counterevidence that clears this).
  C. WRONG DOMAIN — primary technical background is computer vision, speech
     recognition, or robotics with no significant NLP / information-retrieval /
     search experience visible in career history or descriptions.
  D. FABRICATED/HOLLOW PROFILE — job titles and career descriptions clearly do
     not support the AI/ML skills listed (e.g., a Marketing Manager, Sales
     Engineer, or Business Analyst whose skills section reads like an ML
     engineer's resume). Skills keywords do NOT override the actual job history.

SOFT DISQUALIFIERS (reduce credibility_score, do NOT set hard_disqualifier):
  E. Entire career at pure consulting / outsourcing firms (TCS, Infosys, Wipro,
     Accenture, Cognizant, Capgemini, Hexaware, Mphasis, HCL) with no product-
     company experience at any point.
  F. Senior engineer who has not written production code in 18+ months because
     they moved fully into architecture, tech-lead, or management roles.
  G. Title-chasing pattern: 3 or more company switches in under 5 years where
     each move was primarily a title upgrade (Engineer → Senior → Staff).
  H. 5+ consecutive years on closed-source proprietary systems with no external
     validation at all (no open-source contributions, no papers, no tech talks).

═══════════════════════════════════════════════════════════════════
STEP 1 — SCORE DIMENSION 1: RETRIEVAL & SEARCH SYSTEMS DEPTH (0–30)
═══════════════════════════════════════════════════════════════════
Measures production experience with embeddings-based retrieval AND vector
search / hybrid search infrastructure.

BAND 0–5 — Minimal or irrelevant:
  • No semantic / dense retrieval work; only BM25, SQL, or keyword search.
  • Embeddings appear only in tutorials, blog posts, or Colab notebooks.
  • All ML work is in unrelated domains (tabular, CV, speech, time-series).
  • "RAG" or "vector search" listed as a skill but no supporting project or
    job description mentions it in a production context.

BAND 6–14 — Prototype / limited exposure:
  • Used sentence-transformers or OpenAI embeddings in an internal tool, demo,
    or side project but NOT deployed to external users at scale.
  • Touched Pinecone, Qdrant, FAISS, Weaviate, or Milvus in proof-of-concept
    work; no evidence of running it in production under load.
  • Search work is primarily keyword-based with a semantic layer added on top
    but not owned end-to-end.

BAND 15–22 — Solid production experience (meets the must-have bar):
  • Has deployed at least ONE embeddings-based retrieval system to real users
    at a product company (not a demo or internal tool).
  • Operational experience with a vector database in production: handled index
    refresh schedules, latency SLAs, or retrieval-quality monitoring.
  • Evidence of understanding dense-vs-hybrid tradeoffs from actual production
    (not just having read the papers).
  • Built or maintained a two-stage pipeline (ANN retrieval + re-ranking) or a
    hybrid search system (BM25 + dense) in a live product.

BAND 23–30 — Deep multi-system expertise (exceeds the must-have bar):
  • Owned the full retrieval stack (indexing pipeline, serving layer, quality
    monitoring) across multiple systems or at meaningful scale (millions of
    documents / queries per day).
  • Has first-hand experience with embedding drift, retrieval-quality regression
    in production, and corrective action (re-indexing strategies, model swaps).
  • Strong, defensible opinions on dense vs hybrid retrieval grounded in
    specific systems they built — not opinions borrowed from papers.
  • Has worked with multiple embedding model families (BGE, E5, OpenAI,
    fine-tuned bi-encoders) and can articulate concrete tradeoffs from use.

SCORE: integer 0–30

═══════════════════════════════════════════════════════════════════════
STEP 2 — SCORE DIMENSION 2: EVALUATION & RANKING RIGOR (0–20)
═══════════════════════════════════════════════════════════════════════
Measures hands-on experience designing and running evaluation frameworks
specifically for ranking and retrieval systems.

BAND 0–4 — No evidence:
  • No mention of NDCG, MRR, MAP, Precision@K, or other IR metrics anywhere
    in the profile.
  • "Evaluation" in their history means accuracy, F1, or AUC on classification
    tasks, not ranking-quality metrics.
  • Has run experiments but never measured retrieval quality rigorously.

BAND 5–9 — Basic awareness:
  • Mentions ranking metrics by name (NDCG, MRR) but the career descriptions
    don't show ownership of measurement infrastructure.
  • Has participated in A/B tests as a consumer (read results, made decisions)
    but did not design or own the experiment framework.
  • Academic exposure to IR evaluation (course projects, papers) without
    production application.

BAND 10–15 — Practiced experience (meets the must-have bar):
  • Has designed offline evaluation benchmarks for at least one production
    ranking or retrieval system (built annotation sets, computed NDCG/MRR).
  • Has run A/B tests on a live ranking system and can interpret results
    (statistical significance, novelty effects, long-term vs short-term lift).
  • Shows awareness of the offline-to-online correlation problem (knows that
    good offline metrics don't always translate to online gains).
  • Some experience with user-feedback loops or implicit signals (clicks,
    dwell time, recruiter accept/reject actions) feeding back into evaluation.

BAND 16–20 — Framework ownership (exceeds the must-have bar):
  • Has built and owned the full evaluation pipeline: ground-truth annotation,
    offline benchmarks, online A/B experiments, user-feedback loops.
  • Track record of improving ranking quality with before/after metric evidence
    (can quote numbers: "improved NDCG@10 from X to Y after re-ranking").
  • Has navigated the offline-to-online correlation failure case in practice
    (shipped something that looked great offline, diagnosed the gap, fixed it).
  • Has designed evaluation specifically for retrieval / search (not just
    adapted classification-evaluation tooling).

SCORE: integer 0–20

═══════════════════════════════════════════════════════════════════════════
STEP 3 — SCORE DIMENSION 3: APPLIED ML PRODUCT ENGINEERING (0–25)
═══════════════════════════════════════════════════════════════════════════
Measures the "shipper" vs. "researcher" balance the JD explicitly asks for.
This role writes production code. It does not need a researcher who has
never deployed, nor a senior architect who stopped coding 2 years ago.

BAND 0–6 — Pure research, services, or pre-ML career:
  • Entire career in academic labs, research-only roles, or large
    consulting/outsourcing firms (TCS, Infosys, Wipro, Accenture, Cognizant,
    Capgemini, Hexaware, Mphasis) with zero product-company experience.
  • ML models exist only in notebooks or internal demos — never reached real
    users.
  • AI/ML experience begins in 2023 or later and is primarily API calls to
    OpenAI / Anthropic / Cohere with no pre-LLM production ML background.

BAND 7–13 — Some product exposure, limited ML ownership:
  • At least one product-company role but primarily in supporting/specialist
    capacity (data analyst, MLOps support, junior research role).
  • 1–2 production ML deployments but small-scale, not in retrieval/ranking/
    recommendation domains, or inherited rather than built from scratch.
  • Pre-LLM ML background exists but is shallow: no end-to-end ranking or
    search system shipped.
  • Python is present but no evidence of production code quality (no GitHub,
    no system design, no description of code review or architecture ownership).

BAND 14–20 — Strong applied ML at product companies (meets the target bar):
  • 4+ years total at product companies (non-services) in ML/AI engineering.
  • Has shipped at least one end-to-end ranking, search, or recommendation
    system to real users with meaningful traffic.
  • Pre-LLM retrieval/ranking understanding: worked on search or recommendation
    before 2023 and can demonstrate fundamentals that predate the LLM wave.
  • Writes production-quality Python: descriptions mention code review, CI/CD,
    testing, or open-source work consistent with high code quality.

BAND 21–25 — Ideal applied ML engineer (exceeds the target bar):
  • 5+ years at product companies building ML systems that real users depend on,
    not just support or research roles.
  • Has owned a ranking, search, or recommendation system through its full
    lifecycle: initial design → production deployment → ongoing improvement.
  • Strong Python evidenced by open-source contributions, described system
    architecture, or technical blog/talk content about production systems.
  • Additional domain bonus: prior experience in HR tech, recruiting tech,
    two-sided marketplace products, or large-scale inference optimization.

SCORE: integer 0–25

═══════════════════════════════════════════════════════════════════════════════
STEP 4 — SCORE DIMENSION 4: PROFILE CREDIBILITY & DISQUALIFIERS (0–15)
═══════════════════════════════════════════════════════════════════════════════
Assesses whether the profile is substantively real. A keyword-stuffed shell
profile scores low here even if every other dimension looks strong.

IMPORTANT: If a HARD DISQUALIFIER was found in Step 0, score this 0.

15 — Highly credible, no disqualifiers:
  Career history clearly matches stated skills. Technical depth is visible in
  project/role descriptions (not just a skills list). Evidence of longevity at
  companies (not constant switching). No soft disqualifiers apply.

10–14 — Credible with minor issues:
  Profile is clearly real and technically substantive, but 1 soft disqualifier
  applies (e.g., one stint at a consulting firm among otherwise product-company
  roles, or a recent architectural / management pivot).

5–9 — Credibility concerns:
  Multiple soft disqualifiers OR profile is thin (job descriptions are 1-line
  summaries with no technical substance) OR skill list feels inflated relative
  to what the career descriptions actually support.

0–4 — Serious credibility problems or hard disqualifier applies:
  Hard disqualifier found → score 0.
  OR: Profile descriptions are almost entirely absent; skills section is the
  only content; strong mismatch between job titles and listed AI/ML skills.

SCORE: integer 0–15

═══════════════════════════════════════════════════════════════════
STEP 5 — SCORE DIMENSION 5: NICE-TO-HAVES & AVAILABILITY (0–10)
═══════════════════════════════════════════════════════════════════
Nice-to-haves are tiebreakers — they should not compensate for weak must-haves.
Availability signals are minor tilts, not hard criteria.

NICE-TO-HAVES (up to 6 points, 1–2 per item):
  • LLM fine-tuning experience: LoRA, QLoRA, PEFT, adapter training (not just
    "used a fine-tuned model").
  • Learning-to-rank models: LambdaMART, XGBoost-based L2R, neural rankers
    (listwise or pairwise), beyond just using them from a library.
  • HR tech, recruiting tech, or two-sided marketplace product experience
    (directly relevant domain).
  • Distributed systems or large-scale inference optimization: served models at
    >100 QPS, managed multi-GPU serving clusters, optimized inference latency.
  • Open-source AI/ML contributions: meaningful PRs to notable repos, a library
    of their own with real users — NOT just tutorial repos or forks.

AVAILABILITY SIGNALS (up to 4 points):
  • +2: notice period ≤ 30 days (or buyout mentioned) AND open_to_work = true
  • +1: recruiter response rate ≥ 60%
  • +1: last active within the last 30 days
  • −1 (applied as reduction from above): notice period > 60 days
  • −2 (applied as reduction from above): inactive > 6 months AND not open
    to work AND response rate < 20% (this candidate is practically unavailable)

Sum nice-to-haves + availability, clamp to [0, 10].

SCORE: integer 0–10

══════════════════════════════════
TOTAL SCORE AND OUTPUT FORMAT
══════════════════════════════════
total_score = retrieval_score + evaluation_score + applied_ml_score +
              credibility_score + bonus_score

IF hard_disqualifier is true: total_score = min(total_score, 20)

Output ONLY this JSON object, no other text:
{
  "retrieval_score":   <integer 0–30>,
  "evaluation_score":  <integer 0–20>,
  "applied_ml_score":  <integer 0–25>,
  "credibility_score": <integer 0–15>,
  "bonus_score":       <integer 0–10>,
  "total_score":       <integer 0–100>,
  "hard_disqualifier": <true or false>,
  "disqualifier_reason": <"string describing which disqualifier (A/B/C/D) and why" or null>,
  "brief_rationale": "<2-3 sentences citing specific concrete facts from THIS profile that drove the scores>"
}

CRITICAL CONSISTENCY RULES:
- Use the band anchors above literally — do not invent a score outside the
  described band for a given level of evidence.
- brief_rationale MUST reference at least one specific fact from the profile
  (company name, tool name, metric mentioned, year). Generic statements like
  "the candidate has relevant experience" are not acceptable.
- Do not reward keyword presence in a skills list. Only reward evidence in
  career history and project descriptions.
- total_score must equal the arithmetic sum of the five subscores, subject to
  the hard_disqualifier cap.\
"""

SCORING_SCHEMA = {
    "type": "object",
    "properties": {
        "retrieval_score":    {"type": "integer", "minimum": 0, "maximum": 30},
        "evaluation_score":   {"type": "integer", "minimum": 0, "maximum": 20},
        "applied_ml_score":   {"type": "integer", "minimum": 0, "maximum": 25},
        "credibility_score":  {"type": "integer", "minimum": 0, "maximum": 15},
        "bonus_score":        {"type": "integer", "minimum": 0, "maximum": 10},
        "total_score":        {"type": "integer", "minimum": 0, "maximum": 100},
        "hard_disqualifier":  {"type": "boolean"},
        "disqualifier_reason": {"type": ["string", "null"]},
        "brief_rationale":    {"type": "string"},
    },
    "required": [
        "retrieval_score", "evaluation_score", "applied_ml_score",
        "credibility_score", "bonus_score", "total_score",
        "hard_disqualifier", "disqualifier_reason", "brief_rationale",
    ],
}

_ENGINE = None

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

def load_top_n(ranked_path: str, top_n: int) -> List[dict]:
    rows = []
    with open(ranked_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= top_n:
                break
    return rows

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
            f"WARNING: {len(remaining)} candidate_ids from the ranked CSV were not "
            f"found in {pool_path} — they will be dropped. "
            f"Example missing ids: {list(remaining)[:5]}"
        )
    return found

def build_candidate_block(c: dict) -> str:
    """Full, readable rendering of one candidate profile for the LLM."""
    p   = c.get("profile", {}) or {}
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
            name  = s.get("name", "")
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

def load_scores_cache(path: str) -> Dict[str, List[dict]]:
    """Returns {candidate_id: [scored_entry, ...]} — list because of --num-passes."""
    cache: Dict[str, List[dict]] = {}
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
                cid = entry.get("candidate_id")
                if cid:
                    cache.setdefault(cid, []).append(entry)
        total_entries = sum(len(v) for v in cache.values())
        print(f"Loaded {total_entries} cached score entries for "
              f"{len(cache)} candidates from {path}")
    return cache

def append_score(path: str, entry: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()

def average_scores(entries: List[dict]) -> dict:
    """Average the numeric sub-scores across multiple passes.
    hard_disqualifier is True if ANY pass flagged it (conservative).
    brief_rationale comes from the first pass.
    total_score is recomputed from averaged sub-scores (with the cap applied).
    """
    dims = [
        "retrieval_score", "evaluation_score", "applied_ml_score",
        "credibility_score", "bonus_score",
    ]
    averaged = {}
    for d in dims:
        vals = [e[d] for e in entries if d in e]
        averaged[d] = round(sum(vals) / len(vals)) if vals else 0

    hard_dq = any(e.get("hard_disqualifier", False) for e in entries)
    averaged["hard_disqualifier"] = hard_dq

    dq_reason = None
    for e in entries:
        if e.get("hard_disqualifier") and e.get("disqualifier_reason"):
            dq_reason = e["disqualifier_reason"]
            break
    averaged["disqualifier_reason"] = dq_reason

    averaged["brief_rationale"] = entries[0].get("brief_rationale", "")

    raw_total = sum(averaged[d] for d in dims)
    averaged["total_score"] = min(raw_total, 20) if hard_dq else raw_total
    return averaged

def compute_pointwise_prompt_budget(blocks: Dict[str, str], tokenizer, max_tokens: int) -> dict:
    """Tokenizes the actual worst-case scorer prompt: rubric + largest
    single candidate block in THIS top-N set."""
    token_counts = {
        cid: len(tokenizer.encode(text, add_special_tokens=False))
        for cid, text in blocks.items()
    }
    largest_id, largest_n = max(token_counts.items(), key=lambda kv: kv[1])

    messages = build_score_messages(largest_id, blocks[largest_id])
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompt_tokens = len(tokenizer.encode(formatted, add_special_tokens=False))

    return {
        "largest_id":            largest_id,
        "largest_block_tokens":  largest_n,
        "worst_case_prompt_tokens": prompt_tokens,
        "generation_budget":     max_tokens,
        "total_required":        prompt_tokens + max_tokens,
    }

def preflight_check_prompt_budget(
    blocks: Dict[str, str], model_name: str, max_model_len: int, max_tokens: int
) -> None:
    print(
        f"\nPre-flight check 1/2: worst-case scorer prompt length "
        f"for these {len(blocks)} candidates ..."
    )
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(
            f"  WARNING: could not load tokenizer to check prompt budget ({e}). "
            "Skipping — vLLM will enforce max_model_len itself at request time."
        )
        return

    info = compute_pointwise_prompt_budget(blocks, tok, max_tokens)
    print(f"  Largest candidate block in this set: {info['largest_id']} "
          f"({info['largest_block_tokens']} tok)")
    print(f"  Worst-case full prompt (rubric + block + chat template): "
          f"{info['worst_case_prompt_tokens']} tok")
    print(f"  + generation budget (--max-tokens): {max_tokens} tok")
    print(f"  = total required: {info['total_required']} tok   "
          f"(your --max-model-len is {max_model_len})")

    if info["total_required"] > max_model_len:
        deficit = info["total_required"] - max_model_len
        raise SystemExit(
            f"\nABORTING before touching the GPU: --max-model-len {max_model_len} is "
            f"{deficit} tokens too small. Rerun with at least "
            f"--max-model-len {info['total_required'] + 64}."
        )

    headroom = max_model_len - info["total_required"]
    print(f"  OK — {headroom} tokens of headroom.")
    if headroom > 512:
        suggested = info["total_required"] + 128
        print(
            f"  NOTE: large headroom. You could reclaim VRAM by setting "
            f"--max-model-len {suggested}."
        )

def preflight_check_vram(
    gpu_memory_utilization: float, num_gpus: int = 1, safety_margin: float = 0.97
) -> None:
    """Checks every GPU that vLLM will use. See 06 for the full rationale."""
    print(
        f"\nPre-flight check 2/2: actual free VRAM on all {num_gpus} GPU(s) ..."
    )
    try:
        import torch
        if not torch.cuda.is_available():
            print(
                "  WARNING: torch.cuda.is_available() is False — skipping. "
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
        total_gb         = total_bytes / 1e9
        free_gb          = free_bytes  / 1e9
        already_used_gb  = total_gb - free_gb
        requested_gb     = gpu_memory_utilization * total_gb

        print(
            f"  GPU (logical index {idx}): total {total_gb:.2f} GB, "
            f"in use {already_used_gb:.2f} GB, free {free_gb:.2f} GB, "
            f"this run will request {requested_gb:.2f} GB"
        )

        if requested_gb > free_gb * safety_margin:
            max_safe_util = (free_gb * safety_margin) / total_gb
            print(
                f"  ^ INSUFFICIENT. Max safe --gpu-memory-utilization "
                f"here is ~{max_safe_util:.3f}."
            )
            any_insufficient = True

    if any_insufficient:
        raise SystemExit(
            "\nABORTING before starting the engine: at least one GPU doesn't "
            "have enough free VRAM. Fix:\n"
            "  1) Kill stale python processes holding VRAM (nvidia-smi).\n"
            "  2) Lower --gpu-memory-utilization to the 'max safe' value above.\n"
            "  3) Check --gpu-ids excludes any thermal-throttling card."
        )

    print("  OK — all GPUs have sufficient headroom.")

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

DIMS = ["retrieval_score", "evaluation_score", "applied_ml_score",
        "credibility_score", "bonus_score"]
DIM_MAXES = {"retrieval_score": 30, "evaluation_score": 20,
             "applied_ml_score": 25, "credibility_score": 15, "bonus_score": 10}

def validate_and_normalize_score(parsed) -> Optional[dict]:
    """Validate the parsed JSON, clamp sub-scores to declared maxima, and
    recompute total_score from sub-scores so arithmetic errors by the LLM
    don't affect the final ranking."""
    if not isinstance(parsed, dict):
        return None

    result = {}
    for dim, max_val in DIM_MAXES.items():
        val = parsed.get(dim)
        if not isinstance(val, (int, float)):
            return None
        result[dim] = max(0, min(int(round(val)), max_val))

    hard_dq = bool(parsed.get("hard_disqualifier", False))
    result["hard_disqualifier"] = hard_dq

    dq_reason = parsed.get("disqualifier_reason")
    result["disqualifier_reason"] = str(dq_reason).strip() if dq_reason else None

    rationale = parsed.get("brief_rationale", "")
    result["brief_rationale"] = str(rationale).strip()

    raw_total = sum(result[d] for d in DIMS)
    result["total_score"] = min(raw_total, 20) if hard_dq else min(raw_total, 100)

    return result

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
            max_cudagraph_capture_size=0,
            enable_flashinfer_autotune=False,
            enable_prefix_caching=enable_prefix_caching,
        )
    return _ENGINE

def call_llm_batch(
    message_batches, model_name, tensor_parallel_size, schema,
    temperature=0.0, max_tokens=400, gpu_memory_utilization=0.90,
    max_model_len=2560, max_num_seqs=1, enforce_eager=False,
    enable_prefix_caching=True, swap_space=8.0,
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

def build_score_messages(candidate_id: str, block: str) -> List[dict]:
    return [
        {"role": "system", "content": SCORING_RUBRIC},
        {
            "role": "user",
            "content": (
                f"CANDIDATE PROFILE (id: {candidate_id}):\n\n{block}\n\n"
                "Score this candidate now. Output only the JSON object."
            ),
        },
    ]

def score_candidates(
    ids: List[str],
    blocks: Dict[str, str],
    scores_cache: Dict[str, List[dict]],
    cache_path: str,
    args,
    stats: dict,
) -> Dict[str, dict]:
    """Score all candidates in `ids`, respecting the cache and --num-passes.

    Returns {candidate_id: averaged_score_dict} for every id in ids.
    """
    results: Dict[str, dict] = {}

    need_passes: Dict[str, int] = {}
    for cid in ids:
        completed = len(scores_cache.get(cid, []))
        remaining = max(0, args.num_passes - completed)
        if remaining > 0:
            need_passes[cid] = remaining
        else:
            stats["cache_hits"] += 1

    for cid in ids:
        if cid not in need_passes:
            results[cid] = average_scores(scores_cache[cid])

    if not need_passes:
        print("  All candidates loaded from cache — no LLM calls needed.")
        return results

    total_remaining = sum(need_passes.values())
    print(f"  Candidates needing at least one more scoring pass: "
          f"{len(need_passes)} ({total_remaining} total LLM calls).")

    pass_number = 0
    while need_passes:
        pass_number += 1
        this_round = list(need_passes.keys())

        for chunk_start in range(0, len(this_round), args.batch_size):
            chunk = this_round[chunk_start:chunk_start + args.batch_size]

            msgs = [build_score_messages(cid, blocks[cid]) for cid in chunk]
            pending = list(chunk)
            resolved: Dict[str, dict] = {}
            last_raw: Dict[str, str] = {}

            for attempt in range(args.max_retries + 1):
                still_pending = [cid for cid in pending if cid not in resolved]
                if not still_pending:
                    break
                attempt_msgs = [
                    build_score_messages(cid, blocks[cid]) for cid in still_pending
                ]
                raw_outputs = call_llm_batch(
                    attempt_msgs,
                    args.model_name,
                    args.num_gpus,
                    SCORING_SCHEMA,
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
                        validate_and_normalize_score(parsed)
                        if parsed is not None else None
                    )
                    if normalized is not None:
                        resolved[cid] = normalized

            for cid in chunk:
                if cid in resolved:
                    entry = {"candidate_id": cid, "pass": pass_number,
                             **resolved[cid]}
                    scores_cache.setdefault(cid, []).append(entry)
                    append_score(cache_path, entry)
                    stats["llm_calls"] += 1
                    need_passes[cid] -= 1
                    if need_passes[cid] <= 0:
                        del need_passes[cid]
                        results[cid] = average_scores(scores_cache[cid])
                else:
                    fallback_entry = {
                        "candidate_id": cid, "pass": pass_number,
                        "retrieval_score": 0, "evaluation_score": 0,
                        "applied_ml_score": 0, "credibility_score": 0,
                        "bonus_score": 0, "total_score": 0,
                        "hard_disqualifier": False,
                        "disqualifier_reason": None,
                        "brief_rationale": (
                            f"FALLBACK: LLM produced no parseable output after "
                            f"{args.max_retries + 1} attempts. "
                            f"Last raw output: {(last_raw.get(cid) or '')[:150]!r}"
                        ),
                        "fallback": True,
                    }
                    scores_cache.setdefault(cid, []).append(fallback_entry)
                    append_score(cache_path, fallback_entry)
                    stats["fallbacks"] += 1
                    need_passes[cid] -= 1
                    if need_passes[cid] <= 0:
                        del need_passes[cid]
                        results[cid] = average_scores(scores_cache[cid])
                    print(
                        f"  WARNING: scoring {cid} fell back to 0 after "
                        f"{args.max_retries + 1} failed parse attempts."
                    )

            stats["total_done"] += len(chunk)
            if stats["total_done"] % max(args.batch_size, 10) < args.batch_size:
                elapsed = time.time() - stats["t0"]
                rate = stats["total_done"] / elapsed if elapsed > 0 else 0
                print(
                    f"  scored: {stats['total_done']} "
                    f"(new LLM calls: {stats['llm_calls']}, "
                    f"cache hits: {stats['cache_hits']}, "
                    f"fallbacks: {stats['fallbacks']}) — {rate:.2f}/sec"
                )

    return results

def main():
    ap = argparse.ArgumentParser(
        description="Pointwise LLM scorer — replaces the pairwise quicksort."
    )
    ap.add_argument("--ranked", default="outputs/cross_encoder_ranked_honeypot.csv")
    ap.add_argument("--candidates", default="candidates.jsonl",
                    help="Full candidate pool (.jsonl.gz / .jsonl / .json)")
    ap.add_argument("--top-n", type=int, default=30000)
    ap.add_argument("--out", default="outputs/llm_pointwise_top30000.csv")
    ap.add_argument("--scores-cache", default="outputs/llm_pointwise_scores.jsonl",
                    help="JSONL file where individual scoring passes are appended. "
                         "Safe to resume — candidates already in this file at the "
                         "required number of passes will be skipped.")

    ap.add_argument("--model-name", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3-8B-AWQ"))
    ap.add_argument("--gpu-ids", default="",
                    help="Comma-separated physical GPU ids, e.g. '0,1'. Leave "
                         "empty to respect the shell's CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--num-gpus", type=int, default=1,
                    help="Tensor-parallel size inside vLLM.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    ap.add_argument("--max-model-len", type=int, default=6144,
                    help="Single-candidate prompts are ~half the size of the "
                         "pairwise comparator prompts, so this can be set lower "
                         "than in 06 to reclaim KV cache VRAM.")
    ap.add_argument("--max-num-seqs", type=int, default=1,
                    help="Keep at 1 for fragile 8 GB cards.")
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    ap.add_argument("--enable-prefix-caching", action="store_true", default=True,
                    help="ON by default (unlike 06) because the system-prompt "
                         "rubric is identical for every call, making it an ideal "
                         "prefix-caching workload. Pass --no-enable-prefix-caching "
                         "to disable if you observe instability.")
    ap.add_argument("--no-enable-prefix-caching", dest="enable_prefix_caching",
                    action="store_false")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="Generation budget per scoring call. The 9-field JSON "
                         "response (with a rationale sentence) typically needs "
                         "200-350 tokens.")
    ap.add_argument("--swap-space", type=float, default=4.0,
                    help="GiB of CPU RAM vLLM can use to offload KV cache blocks.")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="Re-attempts per candidate if the LLM output fails to parse.")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Candidates per vLLM call. True GPU concurrency is "
                         "governed by --max-num-seqs; this controls checkpoint "
                         "granularity and the size of each engine.chat() call.")
    ap.add_argument("--num-passes", type=int, default=1,
                    help="Number of independent scoring passes per candidate. "
                         "Sub-scores are averaged across passes before final "
                         "ranking. N=1 (default) is already very consistent at "
                         "temperature=0; use N=2 or N=3 if you want extra "
                         "variance guarantees at the cost of N× LLM calls.")
    args = ap.parse_args()

    if args.num_passes < 1:
        raise ValueError("--num-passes must be >= 1")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    configure_visible_gpus(gpu_ids)
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be >= 1")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.scores_cache) or ".", exist_ok=True)

    print(f"Loading top {args.top_n} from {args.ranked} ...")
    top_rows = load_top_n(args.ranked, args.top_n)
    ids = [r["candidate_id"] for r in top_rows]
    original_rank = {r["candidate_id"]: int(r["rank"]) for r in top_rows}
    n_flagged = sum(
        1 for r in top_rows
        if r.get("honeypot_flag", "").strip().lower() == "true"
    )
    if n_flagged:
        print(
            f"NOTE: {n_flagged} of the top {args.top_n} are honeypot-flagged "
            f"(unexpected — 05 should push them to the bottom; double-check inputs)."
        )
    print(f"Loaded {len(ids)} candidate ids.")

    print(f"Looking up full profiles in {args.candidates} ...")
    profiles = load_candidate_profiles(args.candidates, set(ids))
    ids = [i for i in ids if i in profiles]
    print(f"{len(ids)} candidates have full profiles and will be scored.")

    print("Rendering candidate text blocks ...")
    blocks = {cid: build_candidate_block(profiles[cid]) for cid in ids}

    preflight_check_prompt_budget(blocks, args.model_name,
                                  args.max_model_len, args.max_tokens)
    preflight_check_vram(args.gpu_memory_utilization, num_gpus=args.num_gpus)

    scores_cache = load_scores_cache(args.scores_cache)

    print(
        f"\nStarting pointwise scoring: {len(ids)} candidates, "
        f"{args.num_passes} pass(es) each ..."
    )
    stats = {
        "llm_calls": 0, "cache_hits": 0, "fallbacks": 0,
        "total_done": 0, "t0": time.time(),
    }

    final_scores = score_candidates(ids, blocks, scores_cache,
                                    args.scores_cache, args, stats)

    elapsed = time.time() - stats["t0"]
    print(f"\nDone scoring {len(final_scores)} candidates in {elapsed:.1f}s.")
    print(f"  New LLM calls:   {stats['llm_calls']}")
    print(f"  Cache hits:      {stats['cache_hits']}")
    print(f"  Fallbacks (0 score): {stats['fallbacks']}")

    sorted_ids = sorted(
        final_scores.keys(),
        key=lambda cid: (
            -final_scores[cid]["total_score"],
            original_rank.get(cid, 10**9),
        ),
    )

    rows_by_id = {r["candidate_id"]: r for r in top_rows}
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "final_rank",
            "candidate_id",
            "total_score",
            "retrieval_score",
            "evaluation_score",
            "applied_ml_score",
            "credibility_score",
            "bonus_score",
            "hard_disqualifier",
            "disqualifier_reason",
            "brief_rationale",
            "original_cross_encoder_rank",
            "original_cross_encoder_score",
        ])
        for i, cid in enumerate(sorted_ids, start=1):
            sc   = final_scores[cid]
            orig = rows_by_id.get(cid)
            writer.writerow([
                i,
                cid,
                sc["total_score"],
                sc["retrieval_score"],
                sc["evaluation_score"],
                sc["applied_ml_score"],
                sc["credibility_score"],
                sc["bonus_score"],
                sc["hard_disqualifier"],
                sc.get("disqualifier_reason", ""),
                sc.get("brief_rationale", ""),
                original_rank.get(cid, ""),
                orig.get("cross_encoder_score", "") if orig else "",
            ])

    print(f"\nWrote ranked output:  {args.out}")
    print(f"Scoring audit trail:  {args.scores_cache}")
    print()

    all_totals = [final_scores[cid]["total_score"] for cid in sorted_ids]
    n_dq = sum(1 for cid in sorted_ids if final_scores[cid]["hard_disqualifier"])
    bands = {"80-100": 0, "60-79": 0, "40-59": 0, "20-39": 0, "0-19": 0}
    for t in all_totals:
        if t >= 80:   bands["80-100"] += 1
        elif t >= 60: bands["60-79"]  += 1
        elif t >= 40: bands["40-59"]  += 1
        elif t >= 20: bands["20-39"]  += 1
        else:         bands["0-19"]   += 1
    print("Score distribution:")
    for band, count in bands.items():
        bar = "█" * (count * 40 // max(bands.values(), 1))
        print(f"  {band:>7}: {count:4d}  {bar}")
    print(f"\nHard-disqualified candidates (score capped at 20): {n_dq}")
    if all_totals:
        print(
            f"Score range: {min(all_totals)}–{max(all_totals)}  "
            f"mean: {sum(all_totals)/len(all_totals):.1f}"
        )

if __name__ == "__main__":
    main()