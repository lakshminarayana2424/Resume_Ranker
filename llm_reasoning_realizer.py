#!/usr/bin/env python3
"""
llm_reasoning_realizer.py  (v2 -- single-instance, watchdog-timed rewrite)
============================================================================
This is the main reasoning-realization stage used for final ranking.
It turns the already-selected facts from reasoning_engine.py into the
final human-readable reasoning text for the top-100 rows.

Primary path:
  - local LLM on CPU via llama-cpp-python
  - one persistent model instance
  - watchdog per row
  - grounding verification before accepting text

Fallback path:
  - reasoning_engine.py's deterministic template-based reasoning
  - used only when the LLM path is unavailable, times out, or fails the
    grounding checks
  - kept as a safety net so the pipeline still produces a valid output

The final ranking run should rely on the LLM-realized text produced here.
The deterministic reasoning_engine output remains the backup path only.

WHAT CHANGED FROM v1, AND WHY
----------------------------------------------------------------
v1 spawned N worker processes, each loading its own copy of the GGUF
model, then waited on the whole batch with one global timeout. That made
the batch hard to debug and easy to stall.

This version fixes that failure class directly:

1. ONE persistent model instance in the main process, so the model is
   loaded once and reused across all top-100 rows.

2. RAW completion prompting via llm.create_completion() on a hand-built
   ChatML string, which avoids dependence on llama-cpp-python's automatic
   chat-template handling.

3. A hard per-row watchdog via signal.alarm, so one bad row cannot
   consume the whole reasoning budget.

A short self-test runs first on a few rows. That quickly shows whether
the model/setup can finish inside budget before the full batch starts.
Every skip or fallback prints a reason so the run is debuggable from the
console output alone.

FACT SELECTION AND GROUNDING VERIFICATION ARE UNCHANGED FROM v1.
reasoning_engine._select_clauses() still chooses the facts. The
grounding verifier below still rejects fabricated numbers, ungrounded
entities, acronym mismatches, and tone mismatches when concerns are
present. Those failures fall back to the deterministic template output.

COMPLIANCE NOTE
----------------------------------------------------------------
This module loads a local GGUF file already present on disk and makes no
network calls. It is CPU-only and is only applied to the top-100 rows,
not the full candidate pool. The wall-clock cost is reported by rank.py.

MODEL CHOICE
----------------------------------------------------------------
The default model path is kept compatible with the existing setup. If a
larger local GGUF model is already available and still fits the runtime
budget, you can point --llm-model-path to that file without changing this
module's logic.

    pip install llama-cpp-python --break-system-packages
"""


import json
import math
import re
import signal
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import reasoning_engine as _re  # reuse fact selection, never re-derive it


# ============================================================================
# CSV round-trip NaN-truthiness helpers (unchanged from v1; kept in sync
# with rank.py's own duplication convention).
# ============================================================================
def coerce_honeypot_for_reasoning(value) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    s = str(value).strip().lower()
    return 1 if s in ("true", "1", "yes") else 0


def sanitize_row_for_reasoning(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = 0
        else:
            out[k] = v
    return out


# ============================================================================
# Model loading -- ONE instance, lazy singleton, CPU-only, all threads.
# ============================================================================
_LLM = None


def load_llm(model_path: str, n_ctx: int = 768, n_threads: int = 0):
    global _LLM
    if _LLM is not None:
        return _LLM
    from llama_cpp import Llama
    import os
    _LLM = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads or (os.cpu_count() or 4),
        n_gpu_layers=0,          # HARD CPU-only, per submission_spec.md Sec. 3
        use_mmap=True,
        verbose=False,
    )
    return _LLM


# ============================================================================
# Fact payload -- identical to v1, reuses reasoning_engine's own selection.
# ============================================================================
def _build_fact_payload(
    feature_columns: Sequence[str],
    contributions: Sequence[float],
    row: Mapping[str, Any],
    raw: Mapping[str, Any],
    candidate_id: str,
) -> Dict[str, Any]:
    pos_clauses, neg_clauses = _re._select_clauses(
        feature_columns, contributions, row, raw,
        _re.MAX_POSITIVE_CLAUSES, _re.MAX_NEGATIVE_CLAUSES + 1,
    )
    facts = {
        "candidate_id": candidate_id,
        "years_of_experience": raw.get("years_of_experience", row.get("years_of_experience")),
        "current_title": raw.get("current_title"),
        "current_company": raw.get("current_company"),
        "location": raw.get("location"),
        "top_skills": raw.get("top_skill_names") or [],
        "strengths": [c for _, c in pos_clauses],
        "concerns": [c for _, c in neg_clauses],
        "is_honeypot_flagged": bool(row.get("honeypot_flag")),
    }
    return facts


# ============================================================================
# Prompting -- built as a raw ChatML string, sent through create_completion,
# NOT create_chat_completion. Same content as v1's system+user messages;
# the only change is we render the template ourselves so generation does
# not depend on llama-cpp-python's auto chat-template detection/rendering.
# ============================================================================
_SYSTEM_PROMPT = (
    "You write ONE short recruiting note (1-2 sentences, under 320 characters) "
    "from a JSON fact list. Rules: use ONLY facts present in the JSON. Never "
    "invent a skill, employer, number, or location not in the JSON. If "
    "'concerns' is non-empty, your note must acknowledge at least one. Vary "
    "your sentence structure candidate to candidate; do not always start with "
    "the same phrase. Output ONLY the note text -- no preamble like 'Sure' or "
    "'Here is the note', no quotation marks, no labels.\n\n"
    "Example input:\n"
    '{"years_of_experience": 6, "current_title": "Senior ML Engineer", '
    '"location": "Bangalore", "strengths": ["6 years of experience, '
    'squarely in the JD\'s 5-9yr target band (fit 9/10)"], "concerns": '
    '["notice period of 90 days exceeds the JD\'s preferred <=30-day window"]}\n'
    "Example output:\n"
    "Six years in as a Senior ML Engineer puts this candidate squarely in "
    "the target experience band; the 90-day notice period runs well past "
    "what the role prefers, though."
)


def _build_user_prompt(facts: Dict[str, Any]) -> str:
    payload = {k: v for k, v in facts.items() if k != "candidate_id"}
    return "Facts:\n" + json.dumps(payload, ensure_ascii=False, indent=None)


def _build_chatml_prompt(facts: Dict[str, Any]) -> str:
    """Manually rendered ChatML -- matches Qwen2.5-Instruct's format without
    relying on llama-cpp-python's Jinja template auto-detection. If you swap
    to a model that doesn't use ChatML, change only this function."""
    user = _build_user_prompt(facts)
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _clean_generated_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^["\']|["\']$', "", text).strip()
    text = re.sub(
        r"^(sure[,!:]?\s*|here'?s?\s+(the\s+)?note[:\s]*|note[:\s]+)",
        "", text, flags=re.IGNORECASE,
    ).strip()
    return text


def _generate_raw(llm, facts: Dict[str, Any], max_tokens: int = 80) -> str:
    prompt = _build_chatml_prompt(facts)
    out = llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.55,
        top_p=0.9,
        repeat_penalty=1.15,
        stop=["<|im_end|>", "\n\n", "Example", "Facts:"],
    )
    return _clean_generated_text(out["choices"][0]["text"])


# ============================================================================
# Grounding validator -- unchanged from v1.
# ============================================================================
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_TITLE_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]+(?:\s+|-)){1,3}[A-Z][a-zA-Z0-9]+\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_ACRONYM_ALLOWLIST = {"JD", "AI", "ML", "NLP", "CPU", "GPU", "API", "CSV", "ID"}


def _strip_sentence_initial_caps(text: str) -> str:
    return re.sub(
        r"(^|(?<=[.!?]\s))([A-Za-z])",
        lambda m: m.group(1) + m.group(2).lower(),
        text,
    )


def _fact_text_blob(facts: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k, v in facts.items():
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v is not None:
            parts.append(str(v))
    return " | ".join(parts).lower()


def _verify_grounding(text: str, facts: Dict[str, Any]) -> Tuple[bool, str]:
    if not text or len(text) > _re.MAX_REASONING_CHARS + 40:
        return False, "empty or too long"
    blob = _fact_text_blob(facts)
    normalized = _strip_sentence_initial_caps(text)

    for num in _NUMBER_RE.findall(text):
        if num not in blob:
            return False, f"ungrounded number '{num}'"

    for phrase in _TITLE_PHRASE_RE.findall(normalized):
        if phrase.lower() not in blob:
            return False, f"ungrounded multi-word entity '{phrase}'"

    for acro in _ACRONYM_RE.findall(normalized):
        if acro in _ACRONYM_ALLOWLIST:
            continue
        if acro.lower() not in blob:
            return False, f"ungrounded acronym '{acro}'"

    text_lower = text.lower()
    if facts.get("concerns") and not any(
        w in text_lower for w in ("however", "but", "though", "concern", "gap",
                                    "lacks", "no ", "not ", "below", "outside",
                                    "caution", "shorter", "limited")
    ):
        return False, "concerns present in facts but not reflected in tone"

    return True, "ok"


# ============================================================================
# Per-row watchdog. A single stuck generation (model hang, pathological
# decode loop, whatever) can NEVER consume more than per_row_timeout_s --
# this is what actually prevents a repeat of "180s spent, 0 completed".
# Unix-only (SIGALRM); fine for the Linux sandbox this is validated in.
# ============================================================================
class _RowTimeout(Exception):
    pass


def _with_row_watchdog(seconds: float, fn, *args, **kwargs):
    if not hasattr(signal, "SIGALRM"):
        return fn(*args, **kwargs)  # non-Unix fallback: no watchdog, just run

    def _handler(signum, frame):
        raise _RowTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    old_alarm = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# ============================================================================
# Public entry point -- same signature/contract as v1's
# build_reasoning_for_top_n_llm, so rank.py needs no changes.
# ============================================================================
def build_reasoning_for_top_n_llm(
    top_raw_rows: List[dict],
    candidate_ids: List[str],
    feature_columns: Sequence[str],
    contrib_matrix,
    raw_facts_by_id: Dict[str, dict],
    model_path: str,
    time_budget_seconds: float = 180.0,
    n_threads: int = 0,
    max_workers: Optional[int] = None,  # kept for signature compatibility;
                                         # ignored -- see module docstring
                                         # for why single-instance replaced
                                         # the process pool.
) -> List[str]:
    n = len(candidate_ids)
    texts: List[str] = [""] * n
    is_llm: List[bool] = [False] * n  # tracks which rows got real LLM text

    # ------------------------------------------------------------------
    # Step 0: compute every fallback FIRST, unconditionally. This is the
    # safety net -- if the LLM stage does nothing at all below, the
    # function still returns a fully valid, fact-safe submission.
    # ------------------------------------------------------------------
    payload_cache: List[Dict[str, Any]] = [None] * n
    for i, cid in enumerate(candidate_ids):
        row = dict(top_raw_rows[i])
        row["honeypot_flag"] = coerce_honeypot_for_reasoning(row.get("honeypot_flag"))
        row = sanitize_row_for_reasoning(row)
        raw = raw_facts_by_id.get(cid) or {}
        contributions = contrib_matrix[i, :-1]
        texts[i] = _re.build_reasoning(feature_columns, contributions, row, raw, cid)
        payload_cache[i] = _build_fact_payload(
            feature_columns, contributions.tolist(), row, raw, cid
        )

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: load the model once. If this itself fails or is implausibly
    # slow, bail to templates immediately -- don't burn the budget finding
    # out row by row.
    # ------------------------------------------------------------------
    try:
        load_t0 = time.perf_counter()
        llm = load_llm(model_path, n_threads=n_threads)
        load_s = time.perf_counter() - load_t0
        print(f"  Reasoning realization: model loaded in {load_s:.1f}s.")
    except Exception as exc:
        print(f"  Reasoning realization: model load FAILED ({exc!r}) -- "
              f"all {n} rows keep their template fallback.")
        return texts

    # ------------------------------------------------------------------
    # Step 2: self-test on the first few rows, TIMED, before committing
    # the whole batch. This is what turns "180s of silence" into "10-20s
    # to find out whether this is even going to work."
    # ------------------------------------------------------------------
    self_test_n = min(3, n)
    per_row_timeout_s = 25.0
    row_latencies: List[float] = []
    for i in range(self_test_n):
        row_t0 = time.perf_counter()
        try:
            text = _with_row_watchdog(
                per_row_timeout_s, _generate_raw, llm, payload_cache[i]
            )
            ok, reason = _verify_grounding(text, payload_cache[i])
            row_s = time.perf_counter() - row_t0
            row_latencies.append(row_s)
            if ok:
                texts[i] = text
                is_llm[i] = True
                print(f"  Reasoning realization: self-test row {i+1}/{self_test_n} "
                      f"ok in {row_s:.1f}s.")
            else:
                print(f"  Reasoning realization: self-test row {i+1}/{self_test_n} "
                      f"generated but REJECTED ({reason}) in {row_s:.1f}s -- "
                      f"kept template fallback for this row.")
        except _RowTimeout:
            row_s = time.perf_counter() - row_t0
            row_latencies.append(row_s)
            print(f"  Reasoning realization: self-test row {i+1}/{self_test_n} "
                  f"TIMED OUT after {row_s:.1f}s -- kept template fallback.")
        except Exception as exc:
            row_s = time.perf_counter() - row_t0
            row_latencies.append(row_s)
            print(f"  Reasoning realization: self-test row {i+1}/{self_test_n} "
                  f"raised {exc!r} in {row_s:.1f}s -- kept template fallback.")

    avg_row_s = sum(row_latencies) / max(1, len(row_latencies))
    elapsed_so_far = time.perf_counter() - t_start
    remaining_budget = time_budget_seconds - elapsed_so_far
    projected_full_batch_s = avg_row_s * n

    print(f"  Reasoning realization: self-test avg {avg_row_s:.2f}s/row -> "
          f"projected {projected_full_batch_s:.0f}s for all {n} rows "
          f"(budget remaining: {remaining_budget:.0f}s).")

    if avg_row_s <= 0 or projected_full_batch_s > remaining_budget * 0.95:
        print(f"  Reasoning realization: projected time exceeds remaining "
              f"budget -- stopping here, remaining {n - self_test_n} rows "
              f"keep their template fallback. (Try a smaller model, fewer "
              f"max_tokens, or a longer --reasoning-time-budget.)")
        _print_summary(sum(is_llm), n, time.perf_counter() - t_start)
        return texts

    # ------------------------------------------------------------------
    # Step 3: run the remaining rows sequentially, each guarded by the
    # same watchdog, with a running deadline check so we degrade to
    # fallback gracefully near the end instead of overshooting the budget.
    # ------------------------------------------------------------------
    n_timeout = 0
    n_rejected = 0
    n_error = 0
    deadline = t_start + time_budget_seconds

    for i in range(self_test_n, n):
        if time.perf_counter() + avg_row_s > deadline:
            print(f"  Reasoning realization: stopping at row {i}/{n} -- "
                  f"not enough budget left for another row safely.")
            break
        row_t0 = time.perf_counter()
        try:
            text = _with_row_watchdog(
                per_row_timeout_s, _generate_raw, llm, payload_cache[i]
            )
            ok, reason = _verify_grounding(text, payload_cache[i])
            if ok:
                texts[i] = text
                is_llm[i] = True
            else:
                n_rejected += 1
        except _RowTimeout:
            n_timeout += 1
        except Exception:
            n_error += 1
        row_s = time.perf_counter() - row_t0
        avg_row_s = 0.7 * avg_row_s + 0.3 * row_s  # adapt estimate as we go
        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  Reasoning realization: {i+1}/{n} rows processed "
                  f"({sum(is_llm)} LLM-phrased so far, {time.perf_counter() - t_start:.0f}s elapsed).")

    n_llm_used = sum(is_llm)
    elapsed = time.perf_counter() - t_start
    n_fallback = n - n_llm_used
    print(f"  Reasoning realization: {n_llm_used} LLM-phrased, {n_fallback} "
          f"template-fallback ({n_timeout} timed out, {n_rejected} rejected "
          f"by grounding check, {n_error} raised an error), {elapsed:.1f}s "
          f"spent, single model instance, {n_threads or 'auto'} thread(s).")
    return texts


def _print_summary(n_llm_used: int, n: int, elapsed: float) -> None:
    print(f"  Reasoning realization: {n_llm_used} LLM-phrased, {n - n_llm_used} "
          f"template-fallback, {elapsed:.1f}s spent (self-test only -- full "
          f"batch was not attempted).")