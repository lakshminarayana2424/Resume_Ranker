#!/usr/bin/env python3
"""
rank.py
=======
Final ranking and submission assembly.

This is the last stage of the pipeline:
1. Load the trained LightGBM ranker from `11_train_lgbm_ranker.py`.
2. Load the full candidate feature matrix from `08_build_lgbm_features.py`.
3. Score every candidate on CPU.
4. Apply the hard honeypot veto before selecting the top 100.
5. Compute TreeSHAP contributions for the final top-N slice.
6. Load raw candidate facts only for those top-N rows.
7. Generate the final human-readable `reasoning` text.
8. Write the submission CSV in the exact format required by the hackathon validator.

What this script does not do:
- It does not retrain the model.
- It does not rebuild the full feature matrix from raw candidates.
- It does not use GPU inference.
- It does not download or call any online model or API at ranking time.

Inputs
------
- `--model-dir`: LightGBM model directory containing `lgbm_ranker.txt` and `model_metadata.json`.
- `--features-csv`: precomputed full-pool feature CSV from `08_build_lgbm_features.py`.
- `--raw-candidates`: the original candidates file, used only to fetch raw facts for the final top-N reasoning text.
- `--pseudo-labels-csv` (optional): incomplete labels used only for pseudo-metrics on the ranked output.
- `--final-ranked-csv`: honeypot flags used only for the final top-100 honeypot-rate check.

Output
------
- A CSV with exactly these columns:
  `candidate_id`, `rank`, `score`, `reasoning`

Docker / path behavior
----------------------
This script uses repo-relative paths only. The local helper modules are imported from the directory containing this file, so it works when run from the repo root or from inside the script directory in Docker. No hardcoded `Resume_Ranker/` path is required here.

Main command
------------
    python rank.py   --model-dir outputs/lgbm_model   --features-csv outputs/lgbm_features_100k.csv   --raw-candidates candidates.jsonl   --out outputs/submission.csv   --reasoning-mode llm   --llm-model-path models/Qwen2.5-0.5B-Instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf   --reasoning-time-budget 270
"""

import argparse
import csv
import gzip
import json
import math
import os
import re
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Belt-and-suspenders: keep the final ranking process CPU-only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm is not installed. `pip install lightgbm --break-system-packages`",
          file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Import the local reasoning engine after adding the script directory to sys.path.
import reasoning_engine  # noqa: E402 -- the only place reasoning-template logic lives


HACKATHON_RUNTIME_BUDGET_SECONDS = 300  # submission_spec.md Section 3
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")  # mirrors validate_submission.py

# Must mirror 11_train_lgbm_ranker.py / 12_predict_lgbm_cpu.py exactly --
# see those scripts' docstrings ("THE LIGHTGBM CATEGORICAL-FEATURE GOTCHA").
CATEGORICAL_SHIFT = 1
BOOL_LIKE_COLUMNS = ["honeypot_flag"]


def peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 ** 2)
    return raw / 1024


# ============================================================================
# Model loading / scoring -- intentional duplicate of 12_predict_lgbm_cpu.py.
# See that script's docstring for why these are duplicated rather than
# imported (digit-prefixed filenames aren't import-friendly, and this
# script is meant to be runnable standalone). Keep both in sync if you
# change one.
# ============================================================================
def load_model(model_dir: Path, model_metadata: dict) -> Tuple["lgb.Booster", Optional[int]]:
    """Load the single trained LightGBM model used for submission."""
    model_path = model_dir / "lgbm_ranker.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} doesn't exist -- nothing to load.")
    booster = lgb.Booster(model_file=str(model_path))
    return booster, model_metadata.get("best_iteration")


def predict_model(
    booster: "lgb.Booster", X: pd.DataFrame, best_iteration: Optional[int], num_threads: int,
) -> np.ndarray:
    return booster.predict(X, num_iteration=best_iteration, num_threads=num_threads)


def predict_contrib_model(
    booster: "lgb.Booster", X: pd.DataFrame, best_iteration: Optional[int], num_threads: int,
) -> np.ndarray:
    """Returns the TreeSHAP contribution matrix for the model,
    shape (n_rows, n_features + 1) -- last column is the bias/expected-value term.
    This is only called on the top-N slice (N=~100), never the full pool --
    see module docstring.
    """
    return booster.predict(
        X, num_iteration=best_iteration, num_threads=num_threads, pred_contrib=True
    )


def load_features_csv(path: str) -> Tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    df = pd.read_csv(path)
    return df, time.perf_counter() - t0


def coerce_features(
    df: pd.DataFrame, feature_columns: List[str], categorical_columns: List[str]
) -> pd.DataFrame:
    df = df.copy()
    for col in BOOL_LIKE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map(
                {"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0}
            )
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
            continue
        if col in BOOL_LIKE_COLUMNS:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int) + CATEGORICAL_SHIFT
    return df


def apply_honeypot_veto(df: pd.DataFrame) -> pd.DataFrame:
    """Same mechanism as 12_predict_lgbm_cpu.py's veto -- push every
    flagged candidate below every non-flagged one, preserve relative
    order within each group, recompute rank. No-op if honeypot_flag isn't
    a column at all (e.g. upstream honeypot-rules CSV wasn't supplied)."""
    if "honeypot_flag" not in df.columns:
        return df
    flagged = df["honeypot_flag"].fillna(0).astype(int) == 1
    clean = df.loc[~flagged].sort_values(
        ["predicted_score", "candidate_id"], ascending=[False, True]
    )
    pushed = df.loc[flagged].sort_values(
        ["predicted_score", "candidate_id"], ascending=[False, True]
    )
    out = pd.concat([clean, pushed], ignore_index=True)
    out["predicted_rank"] = np.arange(1, len(out) + 1)
    if len(pushed):
        print(f"  Honeypot veto: pushed {len(pushed)} flagged candidate(s) "
              f"below all {len(clean)} non-flagged candidates.")
    return out


# ============================================================================
# Raw candidate JSON streaming -- intentional duplicate of
# 08_build_lgbm_features.py's iter_candidates(). Used ONLY to pull raw
# profile/redrob_signals facts for the final top-N -- never to rebuild
# engineered features (that would be the slow path 12_'s docstring warns
# against, and is unnecessary here since --features-csv already has them).
# ============================================================================
def iter_raw_candidates(path: str) -> Iterable[dict]:
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


def load_raw_facts_for_ids(
    path: str, id_set: Set[str],
) -> Tuple[Dict[str, dict], int]:
    """Streams `path` once, calling reasoning_engine.extract_raw_facts()
    only for candidates in id_set. Returns (facts_by_id, n_scanned). See
    module docstring for why early-exit saves little here (order
    statistics) but is still free to include.
    """
    found: Dict[str, dict] = {}
    remaining = set(id_set)
    n_scanned = 0
    for c in iter_raw_candidates(path):
        n_scanned += 1
        cid = c.get("candidate_id")
        if cid in remaining:
            found[cid] = reasoning_engine.extract_raw_facts(c)
            remaining.discard(cid)
            if not remaining:
                break
    return found, n_scanned


# ============================================================================
# The NaN-truthiness / honeypot-blank-cell fix -- see module docstring "A
# CSV ROUND-TRIP QUIRK". Applied to every top-N row's RAW (unshifted)
# feature dict before it's handed to reasoning_engine.build_reasoning().
# ============================================================================
def coerce_honeypot_for_reasoning(value) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    s = str(value).strip().lower()
    return 1 if s in ("true", "1", "yes") else 0


def sanitize_row_for_reasoning(row: dict) -> dict:
    """Blank CSV cells round-trip through pandas as NaN, and NaN is
    truthy in Python -- left alone, every `if row.get("some_flag")` check
    in reasoning_engine.py's clause renderers would fire on missing data
    as if it were a real positive signal. Zero is the correct default for
    every flag/count-style column here (a missing keyword-family hit count
    means zero hits were found, not "unknown hits"), so this is a safe,
    blanket fix at this single integration boundary -- not a guess about
    any individual column's semantics."""
    out = {}
    for k, v in row.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = 0
        else:
            out[k] = v
    return out


def build_reasoning_for_top_n(
    top_raw_rows: List[dict],
    candidate_ids: List[str],
    feature_columns: List[str],
    contrib_matrix: np.ndarray,
    raw_facts_by_id: Dict[str, dict],
) -> List[str]:
    """contrib_matrix: shape (N, len(feature_columns) + 1), ensemble-
    averaged, last column = bias. top_raw_rows[i] is candidate_ids[i]'s
    RAW (unshifted) feature dict, straight from --features-csv."""
    texts = []
    for i, cid in enumerate(candidate_ids):
        row = dict(top_raw_rows[i])
        row["honeypot_flag"] = coerce_honeypot_for_reasoning(row.get("honeypot_flag"))
        row = sanitize_row_for_reasoning(row)
        contributions = contrib_matrix[i, :-1]  # drop the trailing bias term
        text = reasoning_engine.build_reasoning(
            feature_columns, contributions, row, raw_facts_by_id.get(cid), cid,
        )
        texts.append(text)
    return texts


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model-dir", required=True, type=Path,
                     help="Directory from 11_train_lgbm_ranker.py "
                          "(model_metadata.json + lgbm_ranker.txt).")
    ap.add_argument("--features-csv", required=True,
                     help="Precomputed full-pool feature CSV from "
                          "08_build_lgbm_features.py. This script only "
                          "supports the fast path (see 12_predict_lgbm_cpu.py's "
                          "docstring for why) -- there is no --candidates "
                          "live-feature-building option here.")
    ap.add_argument("--raw-candidates", default=None,
                     help="candidates.jsonl / .jsonl.gz / .json -- used ONLY "
                          "to pull raw profile/redrob_signals text (title, "
                          "location, company, education, top skills) for the "
                          "final top-N, for higher-quality reasoning. "
                          "Required unless --skip-raw-facts is passed.")
    ap.add_argument("--skip-raw-facts", action="store_true",
                     help="Skip raw-fact loading entirely (faster, but "
                          "reasoning falls back to engineered-feature-only "
                          "phrasing -- weaker on Stage 4's 'specific facts' "
                          "check). For quick smoke tests only.")
    ap.add_argument("--out", required=True,
                     help="Final submission CSV path. Rename to your "
                          "registered participant_id.csv before uploading "
                          "(submission_spec.md Section 2).")
    ap.add_argument("--top-n", type=int, default=100,
                     help="submission_spec.md requires exactly 100; only "
                          "change this for debugging.")
    ap.add_argument("--no-honeypot-veto", action="store_true",
                     help="Debug only -- see module docstring. Never use "
                          "this for an actual submission.")
    ap.add_argument("--num-threads", type=int, default=0,
                     help="LightGBM predict threads. 0 = all visible CPU cores.")
    ap.add_argument("--reasoning-mode", choices=["template", "llm"], default="template",
                     help="'template' (default): reasoning_engine.py's deterministic "
                          "SHAP-grounded templates. 'llm': same fact selection, but "
                          "sentence wording comes from a small local CPU-only GGUF "
                          "model (see llm_reasoning_realizer.py) to avoid the "
                          "recurring-skeleton problem template output has at 100 "
                          "rows -- with per-row grounding verification and automatic "
                          "fallback to 'template' if a generation can't be verified "
                          "or the time budget runs low. Requires --llm-model-path.")
    ap.add_argument("--llm-model-path", default=None,
                     help="Path to a local .gguf model file, required if "
                          "--reasoning-mode llm. Must already exist on disk -- "
                          "never downloaded at ranking time (see "
                          "llm_reasoning_realizer.py docstring).")
    ap.add_argument("--reasoning-time-budget", type=float, default=270.0,
                     help="Seconds reserved for LLM reasoning generation before "
                          "falling back to templates for remaining rows. Keeps "
                          "total wall-clock inside the 300s spec budget even on "
                          "slower hardware.")
    ap.add_argument("--max-workers", type=int, default=None,
                     help="Worker PROCESSES for parallel LLM reasoning generation "
                          "(--reasoning-mode llm only). Default: cpu_count()-1, "
                          "leaving one core for the main process. Each worker "
                          "loads its own single-threaded model instance -- on a "
                          "6-core machine, 5 workers running 1 candidate each in "
                          "parallel beats 1 candidate at a time using all 6 "
                          "threads, since generation is latency- not throughput-"
                          "bound per row.")
    ap.add_argument("--llm-threads-per-worker", type=int, default=1,
                     help="llama.cpp threads PER WORKER PROCESS. Deliberately "
                          "separate from --num-threads (that's LightGBM's, "
                          "already done by the time reasoning generation starts) "
                          "-- workers x this must not exceed your core count or "
                          "you'll oversubscribe and go slower, not faster.")
    args = ap.parse_args()

    if args.reasoning_mode == "llm" and not args.llm_model_path:
        print("ERROR: --llm-model-path is required when --reasoning-mode llm.",
              file=sys.stderr)
        sys.exit(1)

    if not args.skip_raw_facts and not args.raw_candidates:
        print("ERROR: --raw-candidates is required unless --skip-raw-facts "
              "is passed (see --help).", file=sys.stderr)
        sys.exit(1)

    t_wall_start = time.perf_counter()
    phase_times: Dict[str, float] = {}

    # ---- Phase 1: load model ----
    t0 = time.perf_counter()
    meta_path = args.model_dir / "model_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        model_metadata = json.load(f)
    feature_columns: List[str] = model_metadata["feature_columns"]
    categorical_columns: List[str] = model_metadata["categorical_columns"]
    booster, best_iteration = load_model(args.model_dir, model_metadata)
    phase_times["1_load_model"] = time.perf_counter() - t0
    print(f"Loaded model from {args.model_dir / 'lgbm_ranker.txt'} "
          f"({len(feature_columns)} feature columns expected).")

    # ---- Phase 2: load full-pool features ----
    raw_df, t_load = load_features_csv(args.features_csv)
    phase_times["2_load_features_csv"] = t_load
    print(f"Loaded {len(raw_df)} precomputed feature rows from "
          f"{args.features_csv} in {t_load:.2f}s.")

    missing_cols = [c for c in feature_columns if c not in raw_df.columns]
    if missing_cols:
        print(f"  WARNING: {len(missing_cols)} expected feature column(s) missing, "
              f"will be NaN: {missing_cols[:8]}{' ...' if len(missing_cols) > 8 else ''}")

    # ---- Phase 3: coerce dtypes (shifted, for the model) ----
    t0 = time.perf_counter()
    feats = coerce_features(raw_df, feature_columns, categorical_columns)
    X = feats[feature_columns]
    candidate_ids_full = feats["candidate_id"].to_numpy()
    phase_times["3_coerce_dtypes"] = time.perf_counter() - t0

    # ---- Phase 4: score the FULL pool ----
    t0 = time.perf_counter()
    scores = predict_model(booster, X, best_iteration, num_threads=args.num_threads)
    phase_times["4_predict_full_pool"] = time.perf_counter() - t0
    print(f"Scored {len(scores)} candidates in {phase_times['4_predict_full_pool']:.3f}s.")

    # ---- Phase 5: rank + honeypot veto ----
    t0 = time.perf_counter()
    ranked = pd.DataFrame({"candidate_id": candidate_ids_full, "predicted_score": scores})
    if "honeypot_flag" in feats.columns:
        ranked["honeypot_flag"] = feats["honeypot_flag"].fillna(0).astype(int)
    ranked = ranked.sort_values(
        ["predicted_score", "candidate_id"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked["predicted_rank"] = np.arange(1, len(ranked) + 1)
    if not args.no_honeypot_veto:
        ranked = apply_honeypot_veto(ranked)
    else:
        print("  WARNING: --no-honeypot-veto set -- this output is NOT safe "
              "to submit. See module docstring.")
    phase_times["5_rank_and_veto"] = time.perf_counter() - t0

    # ---- Phase 6: slice top-N ----
    top = ranked.head(args.top_n).reset_index(drop=True)
    if len(top) < args.top_n:
        print(f"ERROR: only {len(top)} candidates available, need {args.top_n}.",
              file=sys.stderr)
        sys.exit(1)
    top_ids = top["candidate_id"].tolist()
    top_id_set = set(top_ids)

    if "honeypot_flag" in top.columns:
        n_honeypot_in_top = int(top["honeypot_flag"].fillna(0).astype(int).sum())
        honeypot_rate = n_honeypot_in_top / max(len(top), 1)
        print(f"Honeypot rate in final top-{len(top)}: {n_honeypot_in_top} "
              f"({honeypot_rate:.1%}) -- submission_spec.md Sec. 7 disqualifies at >10%.")
        if honeypot_rate > 0.10:
            print("ERROR: honeypot rate in final top-100 exceeds the 10% Stage-3 "
                  "disqualification threshold even after the veto. Do not submit "
                  "this file -- investigate the veto logic / model before rerunning.",
                  file=sys.stderr)
            sys.exit(1)

    # Position lookup into the full feats/X frames, preserving the top-N order.
    pos_by_id = {cid: i for i, cid in enumerate(candidate_ids_full)}
    top_positions = [pos_by_id[cid] for cid in top_ids]
    X_top = X.iloc[top_positions].reset_index(drop=True)
    raw_rows_top = [raw_df.iloc[p].to_dict() for p in top_positions]

    # ---- Phase 7: TreeSHAP contributions for just the top-N ----
    t0 = time.perf_counter()
    contrib_matrix = predict_contrib_model(booster, X_top, best_iteration, num_threads=args.num_threads)
    phase_times["7_shap_top_n"] = time.perf_counter() - t0
    recon_score = contrib_matrix.sum(axis=1)
    max_abs_diff = float(np.max(np.abs(recon_score - top["predicted_score"].to_numpy())))
    print(f"Computed TreeSHAP contributions for top {len(top_ids)} "
          f"in {phase_times['7_shap_top_n']:.3f}s "
          f"(bias+contributions vs. model score, max abs diff "
          f"{max_abs_diff:.2e} -- should be ~0, confirms the contributions "
          f"and the displayed score come from the same prediction).")

    # ---- Phase 8: raw facts for just the top-N (full-pool scan, see docstring) ----
    raw_facts_by_id: Dict[str, dict] = {}
    if not args.skip_raw_facts:
        t0 = time.perf_counter()
        raw_facts_by_id, n_scanned = load_raw_facts_for_ids(args.raw_candidates, top_id_set)
        phase_times["8_load_raw_facts"] = time.perf_counter() - t0
        n_missing = len(top_id_set) - len(raw_facts_by_id)
        print(f"Scanned {n_scanned} candidate(s) in {args.raw_candidates} to find raw "
              f"facts for the top {len(top_id_set)} in {phase_times['8_load_raw_facts']:.2f}s"
              + (f" -- WARNING: {n_missing} top-{args.top_n} candidate_id(s) not found in "
                 f"that file (reasoning falls back to engineered-feature-only phrasing "
                 f"for them -- double check --raw-candidates points at the SAME file "
                 f"your features were built from)." if n_missing else "."))
    else:
        phase_times["8_load_raw_facts"] = 0.0
        print("Skipping raw-fact loading (--skip-raw-facts) -- reasoning will use "
              "engineered-feature-only phrasing.")

    # ---- Phase 9: build reasoning ----
    t0 = time.perf_counter()
    if args.reasoning_mode == "llm":
        import llm_reasoning_realizer  # local import: only needed for this mode
        reasoning_texts = llm_reasoning_realizer.build_reasoning_for_top_n_llm(
            raw_rows_top, top_ids, feature_columns, contrib_matrix, raw_facts_by_id,
            model_path=args.llm_model_path,
            time_budget_seconds=args.reasoning_time_budget,
            n_threads=args.llm_threads_per_worker,
            max_workers=args.max_workers,
        )
    else:
        reasoning_texts = build_reasoning_for_top_n(
            raw_rows_top, top_ids, feature_columns, contrib_matrix, raw_facts_by_id,
        )
    phase_times["9_build_reasoning"] = time.perf_counter() - t0
    n_empty = sum(1 for t in reasoning_texts if not t.strip())
    n_unique = len(set(reasoning_texts))
    print(f"Generated {len(reasoning_texts)} reasoning string(s) in "
          f"{phase_times['9_build_reasoning']:.2f}s "
          f"({n_unique} unique, {n_empty} empty).")
    if n_empty:
        print(f"  WARNING: {n_empty} empty reasoning string(s) -- "
              f"submission_spec.md Section 3 penalizes this at Stage 4.")
    if n_unique < len(reasoning_texts):
        print(f"  WARNING: {len(reasoning_texts) - n_unique} duplicate reasoning "
              f"string(s) among the top {args.top_n} -- check for candidates with "
              f"identical engineered features (e.g. both missing raw facts).")

    # ---- Phase 10: assemble + write the submission CSV ----
    t0 = time.perf_counter()
    out_df = pd.DataFrame({
        "candidate_id": top_ids,
        "rank": np.arange(1, len(top_ids) + 1),
        "score": np.round(top["predicted_score"].to_numpy(), 6),
        "reasoning": reasoning_texts,
    })

    # ---- Self-check: mirror validate_submission.py's core invariants ----
    errors = []
    if list(out_df.columns) != ["candidate_id", "rank", "score", "reasoning"]:
        errors.append("Column order/names don't match the spec.")
    if len(out_df) != 100:
        errors.append(f"Expected exactly 100 rows, got {len(out_df)}.")
    if out_df["rank"].tolist() != list(range(1, len(out_df) + 1)):
        errors.append("rank column is not exactly 1..N in order.")
    if out_df["candidate_id"].duplicated().any():
        errors.append("Duplicate candidate_id(s) in output.")
    bad_ids = [cid for cid in out_df["candidate_id"] if not CANDIDATE_ID_PATTERN.match(str(cid))]
    if bad_ids:
        errors.append(f"{len(bad_ids)} candidate_id(s) don't match CAND_XXXXXXX: {bad_ids[:5]}")
    scores_arr = out_df["score"].to_numpy()
    if np.any(np.diff(scores_arr) > 0):
        errors.append("score is not non-increasing by rank.")
    if errors:
        print("\nSELF-CHECK FAILED (this would be rejected by validate_submission.py):")
        for e in errors:
            print(f"  - {e}")
        print("Writing the file anyway so you can inspect it, but DO NOT submit it as-is.")
    else:
        print("\nSelf-check passed: row count, rank sequence, candidate_id format/"
              "uniqueness, and score monotonicity all match submission_spec.md. "
              "Still run the organizers' own validate_submission.py before uploading.")

    args.out = str(args.out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)
    phase_times["10_write_csv"] = time.perf_counter() - t0
    print(f"\nWrote {len(out_df)} ranked rows -> {args.out}")
    print("Remember to rename this to your registered participant_id.csv before "
          "uploading (submission_spec.md Section 2).")

    # ---- Report ----
    total_wall = time.perf_counter() - t_wall_start
    peak_mb = peak_rss_mb()

    print(f"\n{'phase':<28}{'seconds':>10}{'% of total':>12}")
    for name, secs in phase_times.items():
        pct = 100 * secs / total_wall if total_wall > 0 else 0.0
        print(f"{name:<28}{secs:>10.3f}{pct:>11.1f}%")
    print(f"{'TOTAL (wall clock)':<28}{total_wall:>10.3f}{100.0:>11.1f}%")

    budget_status = "PASS" if total_wall <= HACKATHON_RUNTIME_BUDGET_SECONDS else "FAIL"
    print(f"\nRuntime budget (submission_spec.md Sec. 3, 5 min / "
          f"{HACKATHON_RUNTIME_BUDGET_SECONDS}s): {total_wall:.1f}s -> {budget_status}")
    print(f"Peak RSS this run: {peak_mb:.0f} MB ({peak_mb / 1024:.2f} GB) -- "
          f"informational only, no 16GB gate enforced per your hardware.")
    print(f"CPU cores visible: {os.cpu_count()}. "
          f"CUDA_VISIBLE_DEVICES='' was set at process start.")


if __name__ == "__main__":
    main()