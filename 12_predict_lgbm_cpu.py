#!/usr/bin/env python3
"""
12_predict_lgbm_cpu.py
========================
CPU-only scoring script for the LightGBM ranker trained in 11_train_lgbm_ranker.py.

What this script does
---------------------
This file is the inference-side scoring core for the hackathon pipeline. It loads the
trained LightGBM model, scores the full candidate pool, sorts candidates by predicted
score, assigns predicted ranks, and writes a full ranked CSV plus a top-N preview.
It also reports timing, peak memory, and optional pseudo-ranking metrics when you
provide incomplete labels for evaluation.

Input modes
-----------
There are two supported ways to supply the candidate pool:

1) --features-csv
   Use the precomputed feature matrix produced by 08_build_lgbm_features.py.
   This is the recommended and fastest path for Docker and submission-time runs.

2) --candidates
   Build features live from candidates.jsonl / candidates.jsonl.gz / JSON array by
   calling feature_engineering.build_feature_row() for each candidate. This path is
   useful for testing, but it is slower because feature extraction happens at ranking
   time.

Why this script is CPU-only
---------------------------
LightGBM prediction is always CPU-based, even if the model was trained with GPU or
CUDA acceleration. To make that explicit at the process level, the script sets
CUDA_VISIBLE_DEVICES="" before importing LightGBM.

Docker / path behavior
----------------------
The script is designed to run from the repository root or from the directory that
contains this file. It looks for feature_engineering.py next to this script, so no
hardcoded Resume_Ranker/ prefix is required. Use relative output paths such as
outputs/... for the cleanest Docker setup.

Evaluation behavior
-------------------
If --pseudo-labels-csv is provided, the script treats missing candidate_ids as grade 0
and reports pseudo NDCG@10, NDCG@50, NDCG@N, MAP, P@10, and P@5 over the full ranked
pool. If --final-ranked-csv is provided, it also checks the honeypot rate in the top
100 ranked candidates for evaluation only.

Outputs
-------
- --out: full scored-and-ranked CSV with candidate_id, predicted_rank, predicted_score
- --top-n: optional preview CSV with the top-N rows
- console timing breakdown for load, feature preparation, prediction, ranking, and I/O

Usage
-----
    # Fast path: score the full pool from a precomputed feature CSV
    python 12_predict_lgbm_cpu.py         --model-dir outputs/lgbm_model         --features-csv outputs/lgbm_features_100k.csv         --pseudo-labels-csv outputs/lgbm_targets_10000.csv         --final-ranked-csv outputs/final_ranked.csv         --out outputs/scored_pool.csv

    # Slow(er) path: build features live from raw candidates
    python 12_predict_lgbm_cpu.py         --model-dir outputs/lgbm_model         --candidates candidates.jsonl.gz         --pseudo-labels-csv outputs/lgbm_targets_10000.csv         --final-ranked-csv outputs/final_ranked.csv         --out outputs/scored_pool.csv
"""


import argparse
import csv
import gzip
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Belt-and-suspenders: make it impossible for anything in this process to
# touch a GPU, even by accident. See module docstring.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm is not installed. `pip install lightgbm --break-system-packages`",
          file=sys.stderr)
    sys.exit(1)

# feature_engineering.py lives next to this script, so we add the script
# directory to sys.path and import it by module name.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from feature_engineering import FEATURE_COLUMNS, build_feature_row, derive_reference_date
except ImportError as e:
    FEATURE_COLUMNS = None  # only required for the --candidates (live) path
    build_feature_row = None
    derive_reference_date = None
    _FEATURE_ENGINEERING_IMPORT_ERROR = e

HACKATHON_RUNTIME_BUDGET_SECONDS = 300  # submission_spec.md Section 3

# Must mirror 11_train_lgbm_ranker.py exactly -- see module docstring.
CATEGORICAL_SHIFT = 1
BOOL_LIKE_COLUMNS: List[str] = []  # no honeypot input columns in this version

# ============================================================================
# Peak memory -- a single read at the end of the run, not periodic sampling.
# ru_maxrss is the high-water mark for the WHOLE process lifetime so far
# (Linux: KB: macOS: bytes), so reading it once at the end already captures
# the peak across every phase. Informational only, per your own 16GB note --
# this script does not abort if it's exceeded.
# ============================================================================
def peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 ** 2)
    return raw / 1024  # Linux: ru_maxrss is already in KB

# ============================================================================
# Candidate-pool loading (mirrors 08_build_lgbm_features.py's iter_candidates
# -- duplicated rather than imported because that module's filename starts
# with a digit and isn't import-friendly; keep these two functions in sync
# if you ever change the upstream version).
# ============================================================================
def iter_candidates(path: str) -> Iterable[dict]:
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

def build_features_live(
    candidates_path: str, reference_date_override: Optional[str],
    limit: Optional[int],
) -> Tuple[pd.DataFrame, float, float]:
    """The --candidates path. Returns (features_df, t_load_seconds, t_feature_seconds)."""
    if build_feature_row is None:
        raise RuntimeError(
            f"feature_engineering.py could not be imported (place it next "
            f"to this script): {_FEATURE_ENGINEERING_IMPORT_ERROR}"
        )

    t0 = time.perf_counter()
    candidates = []
    for i, c in enumerate(iter_candidates(candidates_path)):
        candidates.append(c)
        if limit and i + 1 >= limit:
            break
    t_load = time.perf_counter() - t0

    if reference_date_override:
        from datetime import date as _date
        ref_date = _date.fromisoformat(reference_date_override)
    else:
        ref_date = derive_reference_date(candidates)
        print(f"  WARNING: --reference-date not given; auto-derived "
              f"{ref_date} from this candidate file. This MUST match the "
              f"reference date 08_build_lgbm_features.py used when it built "
              f"the training features, or recency features (days_since_active "
              f"etc.) will drift between train and inference. Pass "
              f"--reference-date explicitly to be sure.")

    t1 = time.perf_counter()
    rows = []
    for c in candidates:
        row = build_feature_row(c, ref_date)
        rows.append(row)
    df = pd.DataFrame(rows)
    t_feature = time.perf_counter() - t1
    return df, t_load, t_feature

def load_features_csv(path: str) -> Tuple[pd.DataFrame, float]:
    # Deliberately NOT dtype=str: letting pandas' C parser infer native
    # int64/float64 dtypes directly is ~6x faster than forcing every cell
    # to string and re-parsing with pd.to_numeric afterward in
    # coerce_features() -- measured 0.95s vs 0.15s on a 20k-row, 131-column
    # feature CSV. coerce_features() still runs (it's needed for the
    # categorical shift and bool-like columns), it just has almost nothing
    # left to do for the already-numeric columns.
    t0 = time.perf_counter()
    df = pd.read_csv(path)
    return df, time.perf_counter() - t0

# ============================================================================
# Dtype coercion -- intentional duplicate of 11_train_lgbm_ranker.py. See
# module docstring.
# ============================================================================
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

# ============================================================================
# Pseudo-label evaluation helpers
# ============================================================================
GRADE_BOUNDARIES = [
    (5, 6),       # rank 1-5      -> grade 6
    (10, 5),      # rank 6-10     -> grade 5
    (25, 4),      # rank 11-25    -> grade 4
    (50, 3),      # rank 26-50    -> grade 3
    (150, 2),     # rank 51-150   -> grade 2
    (1000, 1),    # rank 151-1000 -> grade 1
]
DEFAULT_GRADE = 0

def rank_to_grade(rank: int) -> int:
    for max_rank, grade in GRADE_BOUNDARIES:
        if rank <= max_rank:
            return grade
    return DEFAULT_GRADE

def load_pseudo_label_lookup(path: Optional[str]) -> Dict[str, int]:
    """Load incomplete relevance labels for pseudo-evaluation.

    Missing candidate_ids are treated as grade 0 by the caller.
    Accepts a CSV with candidate_id plus either relevance_grade, grade,
    label, or llm_rank (the latter is converted to the same grade buckets
    used during training).
    """
    if not path:
        print(
            "No --pseudo-labels-csv given: pseudo NDCG metrics will be skipped."
        )
        return {}

    label_aliases = ["relevance_grade", "grade", "label"]
    rank_aliases = ["llm_rank", "rank"]

    lookup: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        label_col = next((a for a in label_aliases if a in fieldnames), None)
        rank_col = next((a for a in rank_aliases if a in fieldnames), None)

        if not label_col and not rank_col:
            print(
                f"WARNING: {path} has neither a relevance-grade column nor an "
                f"llm-rank column among {fieldnames}; pseudo metrics will be skipped."
            )
            return {}

        print(
            f"Pseudo-label CSV column mapping -> label={label_col}, rank={rank_col}"
        )

        for row in reader:
            cid = row.get("candidate_id")
            if not cid:
                continue
            try:
                if label_col is not None:
                    lookup[cid] = int(float(row.get(label_col, 0) or 0))
                else:
                    lookup[cid] = rank_to_grade(int(float(row.get(rank_col, 0) or 0)))
            except (TypeError, ValueError):
                continue

    print(f"Loaded pseudo-labels for {len(lookup)} candidates from {path}")
    return lookup

def _dcg_at_k(relevance_in_rank_order: np.ndarray, k: int, label_gain: np.ndarray) -> float:
    k = min(k, len(relevance_in_rank_order))
    if k <= 0:
        return 0.0
    rel = relevance_in_rank_order[:k]
    gains = label_gain[np.clip(rel, 0, len(label_gain) - 1)]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    return float(np.sum(gains * discounts))

def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int, label_gain: np.ndarray) -> float:
    pred_order = np.argsort(-y_pred, kind="stable")
    dcg = _dcg_at_k(y_true[pred_order], k, label_gain)
    ideal_order = np.argsort(-y_true, kind="stable")
    idcg = _dcg_at_k(y_true[ideal_order], k, label_gain)
    return dcg / idcg if idcg > 0 else 0.0

def map_score(y_relevant_binary: np.ndarray, y_pred: np.ndarray) -> float:
    order = np.argsort(-y_pred, kind="stable")
    rel = y_relevant_binary[order]
    n_rel = rel.sum()
    if n_rel == 0:
        return 0.0
    cum_rel = np.cumsum(rel)
    ranks = np.arange(1, len(rel) + 1)
    precision_at_i = cum_rel / ranks
    return float(np.sum(precision_at_i * rel) / n_rel)


def precision_at_k(y_relevant_binary: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    order = np.argsort(-y_pred, kind="stable")
    k = min(k, len(order))
    if k <= 0:
        return 0.0
    return float(y_relevant_binary[order[:k]].sum() / k)


def compute_pseudo_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_gain: np.ndarray,
    relevant_threshold: int,
) -> Dict[str, float]:
    n = len(y_true)
    y_bin = (y_true >= relevant_threshold).astype(int)
    return {
        "pseudo_ndcg@10": ndcg_at_k(y_true, y_pred, 10, label_gain),
        "pseudo_ndcg@50": ndcg_at_k(y_true, y_pred, 50, label_gain),
        "pseudo_ndcg@N": ndcg_at_k(y_true, y_pred, n, label_gain),
        "pseudo_map": map_score(y_bin, y_pred),
        "pseudo_p@10": precision_at_k(y_bin, y_pred, 10),
        "pseudo_p@5": precision_at_k(y_bin, y_pred, 5),
    }

def load_final_ranked_honeypot_lookup(path: Optional[str]) -> Dict[str, int]:
    """Load honeypot flags from final_ranked.csv for evaluation only."""
    if not path:
        return {}

    required_cols = {"candidate_id", "honeypot_flag"}
    lookup: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = required_cols - fieldnames
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {sorted(missing)}. "
                f"Expected at least candidate_id and honeypot_flag."
            )
        for row in reader:
            cid = row.get("candidate_id")
            if not cid:
                continue
            try:
                lookup[cid] = int(float(row.get("honeypot_flag", 0) or 0))
            except (TypeError, ValueError):
                lookup[cid] = 0
    print(f"Loaded honeypot flags for {len(lookup)} candidates from {path}")
    return lookup


def compute_top_honeypot_percentage(
    ranked_candidate_ids: List[str], honeypot_lookup: Dict[str, int], top_n: int = 100
) -> Tuple[int, int, float]:
    top_ids = ranked_candidate_ids[: min(top_n, len(ranked_candidate_ids))]
    if not top_ids:
        return 0, 0, 0.0
    honeypot_count = sum(1 for cid in top_ids if honeypot_lookup.get(cid, 0) == 1)
    pct = 100.0 * honeypot_count / len(top_ids)
    return honeypot_count, len(top_ids), pct


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model-dir", required=True, type=Path,
                     help="Directory from 11_train_lgbm_ranker.py "
                          "(lgbm_ranker.txt + model_metadata.json).")
    ap.add_argument("--features-csv", default=None,
                     help="Fast path (recommended): precomputed feature CSV "
                          "from 08_build_lgbm_features.py covering the full "
                          "pool. Mutually exclusive with --candidates.")
    ap.add_argument("--candidates", default=None,
                     help="Slow(er) path: raw candidates.jsonl / .jsonl.gz / "
                          ".json -- features built live. See module docstring "
                          "for why this is not what your final rank.py should "
                          "do if you can avoid it.")
    ap.add_argument("--reference-date", default=None,
                     help="Only used with --candidates. YYYY-MM-DD. Must "
                          "match what 08_build_lgbm_features.py used for the "
                          "training features -- see module docstring.")
    ap.add_argument("--limit", type=int, default=None,
                     help="Debug only: stop after N candidates (--candidates path).")
    ap.add_argument("--out", required=True, help="Full scored+ranked CSV output path.")
    ap.add_argument("--top-n", type=int, default=100,
                     help="Also write a <out>.top<N>.csv preview "
                          "(candidate_id, predicted_rank, predicted_score "
                          "only -- NOT yet submission_spec.md format, no "
                          "reasoning column). 0 disables this.")
    ap.add_argument("--pseudo-labels-csv", default=None,
                     help="Optional incomplete-label CSV for inference-side "
                          "pseudo evaluation. Missing candidate_ids are "
                          "treated as grade 0. Use your 10k/20k targets file "
                          "here if you want pseudo NDCG@10/@50/@N on the "
                          "full ranked pool.")
    ap.add_argument("--final-ranked-csv", required=True,
                     help="CSV with final_rank,candidate_id,honeypot_flag used "
                          "only for evaluation of the top-100 honeypot rate.")
    ap.add_argument("--num-threads", type=int, default=0,
                     help="LightGBM predict threads. 0 = all visible CPU cores.")
    args = ap.parse_args()

    if bool(args.features_csv) == bool(args.candidates):
        print("ERROR: pass exactly one of --features-csv or --candidates.", file=sys.stderr)
        sys.exit(1)

    t_wall_start = time.perf_counter()
    phase_times: Dict[str, float] = {}

    # ---- Phase 1: load model ----
    t0 = time.perf_counter()
    model_path = args.model_dir / "lgbm_ranker.txt"
    meta_path = args.model_dir / "model_metadata.json"
    booster = lgb.Booster(model_file=str(model_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        model_metadata = json.load(f)
    feature_columns: List[str] = model_metadata["feature_columns"]
    categorical_columns: List[str] = model_metadata["categorical_columns"]
    phase_times["1_load_model"] = time.perf_counter() - t0
    print(f"Loaded model from {model_path} "
          f"({len(feature_columns)} feature columns expected).")
    print(f"  Trained with device_used={model_metadata.get('device_used')}, "
          f"objective={model_metadata.get('objective')}, "
          f"best_iteration={model_metadata.get('best_iteration')}.")

    pseudo_label_lookup = load_pseudo_label_lookup(args.pseudo_labels_csv)
    pseudo_eval_enabled = bool(pseudo_label_lookup)
    relevant_threshold = int(model_metadata.get("relevant_grade_threshold", 3))

    # ---- Phase 2: load candidate data + build (or load) features ----
    if args.features_csv:
        raw_df, t_phase2 = load_features_csv(args.features_csv)
        phase_times["2a_load_features_csv"] = t_phase2
        phase_times["2b_build_features_live"] = 0.0
        print(f"Loaded {len(raw_df)} precomputed feature rows from "
              f"{args.features_csv} in {t_phase2:.2f}s.")
    else:
        raw_df, t_load, t_feature = build_features_live(
            args.candidates, args.reference_date, args.limit,
        )
        phase_times["2a_load_features_csv"] = t_load
        phase_times["2b_build_features_live"] = t_feature
        print(f"Loaded {len(raw_df)} raw candidates in {t_load:.2f}s, built "
              f"features for them in {t_feature:.2f}s "
              f"({len(raw_df) / max(t_feature, 1e-9):.0f} rows/sec).")

    missing_cols = [c for c in feature_columns if c not in raw_df.columns]
    if missing_cols:
        print(f"  WARNING: {len(missing_cols)} expected feature column(s) not "
              f"present in input, will be filled with NaN: {missing_cols[:8]}"
              f"{' ...' if len(missing_cols) > 8 else ''}")

    # ---- Phase 3: dtype coercion ----
    t0 = time.perf_counter()
    feats = coerce_features(raw_df, feature_columns, categorical_columns)
    X = feats[feature_columns]
    candidate_ids = feats["candidate_id"].to_numpy()
    phase_times["3_coerce_dtypes"] = time.perf_counter() - t0

    # ---- Phase 4: predict (CPU-only -- see module docstring) ----
    t0 = time.perf_counter()
    scores = booster.predict(
        X, num_iteration=model_metadata.get("best_iteration"),
        num_threads=args.num_threads,
    )
    phase_times["4_predict"] = time.perf_counter() - t0
    print(f"Scored {len(scores)} candidates in {phase_times['4_predict']:.3f}s "
          f"({len(scores) / max(phase_times['4_predict'], 1e-9):.0f} rows/sec).")

    # ---- Phase 5: rank + write output ----
    t0 = time.perf_counter()
    out_df = pd.DataFrame({"candidate_id": candidate_ids, "predicted_score": scores})

    out_df = out_df.sort_values(
        ["predicted_score", "candidate_id"], ascending=[False, True]
    ).reset_index(drop=True)
    out_df["predicted_rank"] = np.arange(1, len(out_df) + 1)

    # Inference-side pseudo evaluation:
    # - matched candidate_ids use the labels you provide (e.g. top 10k)
    # - every other candidate is treated as relevance grade 0
    pseudo_metrics = None
    pseudo_matched = 0
    if pseudo_eval_enabled:
        y_true = np.array(
            [pseudo_label_lookup.get(cid, 0) for cid in out_df["candidate_id"]],
            dtype=int,
        )
        pseudo_matched = int(sum(1 for cid in out_df["candidate_id"] if cid in pseudo_label_lookup))
        y_pred_for_eval = -out_df["predicted_rank"].to_numpy(dtype=float)
        label_gain = np.array(model_metadata["label_gain"], dtype=float)
        pseudo_metrics = compute_pseudo_metrics(
            y_true, y_pred_for_eval, label_gain, relevant_threshold
        )

    args.out = str(args.out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df[["candidate_id", "predicted_rank", "predicted_score"]].to_csv(args.out, index=False)

    if args.top_n > 0:
        top_path = str(Path(args.out).with_suffix("")) + f".top{args.top_n}.csv"
        out_df.head(args.top_n)[["candidate_id", "predicted_rank", "predicted_score"]].to_csv(
            top_path, index=False
        )
    phase_times["5_rank_and_write"] = time.perf_counter() - t0

    # ---- Report ----
    total_wall = time.perf_counter() - t_wall_start
    peak_mb = peak_rss_mb()

    print(f"\nWrote {len(out_df)} ranked rows -> {args.out}")
    if args.top_n > 0:
        print(f"Wrote top-{args.top_n} preview -> {top_path} "
              f"(NOT submission_spec.md format -- no reasoning column yet)")

    if pseudo_metrics is not None:
        print(
            f"Pseudo-label eval ({pseudo_matched}/{len(out_df)} candidates labeled; "
            f"unlabeled candidates treated as grade 0):"
        )
        print(
            f"  pseudo_ndcg@10 = {pseudo_metrics['pseudo_ndcg@10']:.6f}\n"
            f"  pseudo_ndcg@50 = {pseudo_metrics['pseudo_ndcg@50']:.6f}\n"
            f"  pseudo_ndcg@N  = {pseudo_metrics['pseudo_ndcg@N']:.6f}\n"
            f"  pseudo_map     = {pseudo_metrics['pseudo_map']:.6f}\n"
            f"  pseudo_p@10    = {pseudo_metrics['pseudo_p@10']:.6f}\n"
            f"  pseudo_p@5     = {pseudo_metrics['pseudo_p@5']:.6f}"
        )
        # Also keep a concise single-line summary for easy copy/paste.
        print(
            f"Pseudo metrics: @10={pseudo_metrics['pseudo_ndcg@10']:.6f}, "
            f"@50={pseudo_metrics['pseudo_ndcg@50']:.6f}, "
            f"@N={pseudo_metrics['pseudo_ndcg@N']:.6f}, "
            f"MAP={pseudo_metrics['pseudo_map']:.6f}, "
            f"P@10={pseudo_metrics['pseudo_p@10']:.6f}, "
            f"P@5={pseudo_metrics['pseudo_p@5']:.6f}"
        )

    final_ranked_lookup = load_final_ranked_honeypot_lookup(args.final_ranked_csv)
    honeypot_count, top_count, honeypot_pct = compute_top_honeypot_percentage(
        out_df["candidate_id"].tolist(), final_ranked_lookup, top_n=100
    )
    print(
        f"Final-ranked honeypot check on LightGBM top-100: "
        f"{honeypot_count}/{top_count} = {honeypot_pct:.2f}% honeypots"
    )

    print(f"\n{'phase':<28}{'seconds':>10}{'% of total':>12}")
    for name, secs in phase_times.items():
        pct = 100 * secs / total_wall if total_wall > 0 else 0.0
        print(f"{name:<28}{secs:>10.3f}{pct:>11.1f}%")
    print(f"{'TOTAL (wall clock)':<28}{total_wall:>10.3f}{100.0:>11.1f}%")

    budget_status = "PASS" if total_wall <= HACKATHON_RUNTIME_BUDGET_SECONDS else "FAIL"
    print(f"\nRuntime budget (submission_spec.md Sec. 3, 5 min / "
          f"{HACKATHON_RUNTIME_BUDGET_SECONDS}s): {total_wall:.1f}s -> {budget_status}")
    if budget_status == "FAIL":
        print("  Re-run with --features-csv pointing at a precomputed feature "
              "matrix instead of --candidates if you used the live path -- "
              "that's almost always where the time goes. See module docstring.")

    print(f"Peak RSS this run: {peak_mb:.0f} MB ({peak_mb / 1024:.2f} GB) "
          f"-- informational only, no 16GB gate enforced per your hardware.")
    print(f"CPU cores visible: {os.cpu_count()}. "
          f"CUDA_VISIBLE_DEVICES='' was set at process start (see module docstring).")

if __name__ == "__main__":
    main()