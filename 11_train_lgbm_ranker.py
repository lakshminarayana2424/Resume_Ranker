#!/usr/bin/env python3
"""
11_train_lgbm_ranker.py
=========================
Train the LightGBM LambdaMART ranker on the train/validation split
produced by 10_assemble_lgbm_training_set.py, then save a portable model
artifact that 12_predict_lgbm_cpu.py can load for the CPU-only,
no-network ranking step.

This script is designed to be Docker-friendly: it uses only the paths you
pass in through CLI arguments, so it can be run from the repository root,
the script directory, or a container working directory without any
hardcoded Resume_Ranker path.

WHAT THIS SCRIPT DOES
----------------------------------------------------------------
* Loads train/val features and labels from the assembled dataset folder.
* Coerces feature types so training and inference stay aligned.
* Trains a LightGBM ranking model using a single query group for this JD.
* Evaluates the model with a custom hackathon metric:
  0.50 * NDCG@10 + 0.30 * NDCG@50 + 0.15 * MAP + 0.05 * P@10.
* Saves the trained booster plus a metadata JSON file needed for
  reproducible inference in 12_predict_lgbm_cpu.py.
* Optionally runs a parallel hyperparameter search with --tune before the
  final full-budget training pass.

WHY LAMBDARANK IS THE DEFAULT
----------------------------------------------------------------
`relevance_grade` is an ordered relevance label, so the task is ranking,
not regression or classification. LambdaMART/LambdaRank is the best fit
because it optimizes pairwise ordering directly, which matches the
hackathon metric far better than a plain regression objective.

WHY THE CUSTOM EVAL MATTERS
----------------------------------------------------------------
LightGBM’s built-in ranking metrics do not exactly match the competition
metric definition, so this script computes NDCG@10, NDCG@50, MAP, and
P@10 directly and combines them into one composite score. Early stopping
tracks that composite score, so the checkpoint you save is the one that
best matches the score you are actually graded on.

WHY THE DEFAULTS ARE CONSERVATIVE
----------------------------------------------------------------
The labeled dataset is small, imbalanced, and noisy, so the default
hyperparameters intentionally bias the model toward simpler trees and
better generalization. `deterministic=True` and `force_row_wise=True`
help keep reruns reproducible, which is useful when you need to defend a
specific submitted model.

CATEGORICAL FEATURE HANDLING
----------------------------------------------------------------
feature_engineering.py uses -1 as a meaningful sentinel in some
categorical columns. LightGBM treats negative categorical values as
missing, so this script shifts categorical columns by +1 before training.
That same transformation must be applied during inference by
12_predict_lgbm_cpu.py.

DEVICE NOTES
----------------------------------------------------------------
* `--device cpu` is the default and is usually the best first run for
  this dataset size.
* `--device gpu` and `--device cuda` only work if your LightGBM build was
  compiled with GPU/CUDA support.
* If GPU training is requested but unavailable, the script falls back to
  CPU with an explicit warning instead of crashing the run.

USAGE
-----
Default CPU run:
    python ./11_train_lgbm_ranker.py \
        --dataset-dir outputs/lgbm_dataset_10000 \
        --out-dir outputs/lgbm_model

Single GPU run:
    python ./11_train_lgbm_ranker.py --dataset-dir outputs/lgbm_dataset \
        --out-dir outputs/lgbm_model --device cuda --gpu-device-id 1

Parallel search across GPUs 1 and 2, then retrain the best config:
    python ./11_train_lgbm_ranker.py --dataset-dir outputs/lgbm_dataset \
        --out-dir outputs/lgbm_model --device cuda --gpu-ids 1,2 \
        --tune --n-trials 24
"""


import argparse
import json
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm is not installed. `pip install lightgbm --break-system-packages`",
          file=sys.stderr)
    sys.exit(1)


# ============================================================================
# See module docstring "THE LIGHTGBM CATEGORICAL-FEATURE GOTCHA". Must stay
# in sync with the identical constant in 12_predict_lgbm_cpu.py.
# ============================================================================
CATEGORICAL_SHIFT = 1

# "True"/"False" string columns that need explicit boolean parsing instead
# of pd.to_numeric (which would just produce NaN for the word "True").
BOOL_LIKE_COLUMNS = []


# ============================================================================
# Independent, from-scratch implementations of NDCG@k / MAP / P@k, matching
# submission_spec.md Section 4's definitions exactly. Deliberately NOT using
# LightGBM's built-in metric computation -- see module docstring.
# ============================================================================
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


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, relevant_threshold: int, label_gain: np.ndarray
) -> Dict[str, float]:
    ndcg10 = ndcg_at_k(y_true, y_pred, 10, label_gain)
    ndcg50 = ndcg_at_k(y_true, y_pred, 50, label_gain)
    y_bin = (y_true >= relevant_threshold).astype(int)
    map_ = map_score(y_bin, y_pred)
    p10 = precision_at_k(y_bin, y_pred, 10)
    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_ + 0.05 * p10
    return {
        "hackathon_composite": composite,
        "ndcg@10": ndcg10,
        "ndcg@50": ndcg50,
        "map": map_,
        "p@10": p10,
    }


def make_hackathon_feval(relevant_threshold: int, label_gain: np.ndarray):
    """Returns a LightGBM-compatible feval. hackathon_composite MUST be
    first in the returned list -- with first_metric_only=True, that's the
    one early stopping actually watches; the rest are logged for visibility
    only."""
    def feval(preds: np.ndarray, dataset: "lgb.Dataset"):
        y_true = dataset.get_label().astype(int)
        m = compute_all_metrics(y_true, preds, relevant_threshold, label_gain)
        return [
            ("hackathon_composite", m["hackathon_composite"], True),
            ("ndcg@10", m["ndcg@10"], True),
            ("ndcg@50", m["ndcg@50"], True),
            ("map", m["map"], True),
            ("p@10", m["p@10"], True),
        ]
    return feval


# ============================================================================
# Data loading / dtype coercion
# ============================================================================
def load_metadata(dataset_dir: Path) -> dict:
    with open(dataset_dir / "dataset_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def coerce_features(
    df: pd.DataFrame, feature_columns: List[str], categorical_columns: List[str]
) -> pd.DataFrame:
    """Mirror EXACTLY in 12_predict_lgbm_cpu.py -- train/inference skew on
    this function is the single easiest way to silently break the model."""
    df = df.copy()

    for col in BOOL_LIKE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map(
                {"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0}
            )  # unmapped (incl. "", "nan") -> NaN, handled natively by LightGBM

    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
            continue
        if col in BOOL_LIKE_COLUMNS:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Categorical shift -- see module docstring "LIGHTGBM CATEGORICAL-
    # FEATURE GOTCHA". Missing categorical values become category 0 after
    # the shift (fillna(-1) -> +1 == 0), which is fine: LightGBM treats 0
    # as just another category, not as "missing" the way it treats NaN.
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int) + CATEGORICAL_SHIFT

    return df


def load_split(
    dataset_dir: Path, split: str, feature_columns: List[str], categorical_columns: List[str]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Native dtype inference, not dtype=str -- see the matching comment in
    # 12_predict_lgbm_cpu.py's load_features_csv for why (measured ~6x
    # faster on read_csv + coerce combined). Negligible at this dataset's
    # row count, but no reason to pay the cost either.
    feats = pd.read_csv(dataset_dir / f"{split}_features.csv")
    labels = pd.read_csv(dataset_dir / f"{split}_labels.csv")

    if list(feats["candidate_id"]) != list(labels["candidate_id"]):
        warnings.warn(
            f"{split}: feature/label row order mismatch -- reindexing "
            f"features to match labels' candidate_id order. This should "
            f"only happen if one of the two files was hand-edited."
        )
        feats = feats.set_index("candidate_id").loc[labels["candidate_id"]].reset_index()

    feats = coerce_features(feats, feature_columns, categorical_columns)
    X = feats[feature_columns]
    y = labels["relevance_grade"].to_numpy(dtype=int)
    cids = feats["candidate_id"].to_numpy()
    return X, y, cids


# ============================================================================
# Param construction with the GPU-index guard rail and graceful fallback
# ============================================================================
def build_params(
    objective: str, boosting: str, num_leaves: int, min_data_in_leaf: int,
    learning_rate: float, feature_fraction: float, bagging_fraction: float,
    bagging_freq: int, lambda_l1: float, lambda_l2: float, max_depth: int,
    min_gain_to_split: float, max_position: int, label_gain: List[int],
    seed: int, num_threads: int, device: str, gpu_device_id: Optional[int],
) -> dict:
    if gpu_device_id == 0:
        raise ValueError(
            "Refusing to use gpu_device_id=0 -- that card is reserved/"
            "thermal-throttling per your hardware notes. Use 1 or 2."
        )

    params = {
        "objective": objective,
        "boosting": boosting,
        "metric": "None",  # we evaluate exclusively via the custom feval
        "label_gain": label_gain,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "max_depth": max_depth,
        "min_gain_to_split": min_gain_to_split,
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": num_threads,
    }
    if objective == "lambdarank":
        params["lambdarank_truncation_level"] = max_position

    if device != "cpu":
        params["device_type"] = device
        params["gpu_device_id"] = gpu_device_id
        params["gpu_use_dp"] = False  # consumer GPUs: fp64 throughput is poor, stick to fp32
        if device == "gpu":
            params["gpu_platform_id"] = -1  # OpenCL: auto-pick platform on the chosen device
    return params


def train_with_device_fallback(
    params: dict, dtrain: "lgb.Dataset", dval: "lgb.Dataset",
    num_boost_round: int, early_stopping_rounds: int, feval,
    log_period: int = 50,
) -> Tuple["lgb.Booster", str]:
    """Attempts training with whatever device is in `params`. If the
    LightGBM build doesn't actually support that device (the standard pip
    wheel has no GPU/CUDA support compiled in), falls back to CPU loudly
    rather than crashing the whole run. Returns (booster, device_actually_used)."""
    requested_device = params.get("device_type", "cpu")
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=True),
        lgb.log_evaluation(period=log_period),
    ]
    try:
        booster = lgb.train(
            params, dtrain, num_boost_round=num_boost_round,
            valid_sets=[dval], valid_names=["val"], feval=feval, callbacks=callbacks,
        )
        return booster, requested_device
    except Exception as e:
        if requested_device == "cpu":
            raise  # CPU failing is a real problem, not a fallback case
        print(
            f"\n!!! Training on device_type='{requested_device}' "
            f"(gpu_device_id={params.get('gpu_device_id')}) failed:\n"
            f"    {type(e).__name__}: {e}\n"
            f"    This almost always means your installed lightgbm wheel "
            f"was not compiled with GPU/CUDA support (the default `pip "
            f"install lightgbm` is CPU-only). Falling back to device_type="
            f"'cpu' for this run so training still completes. To get real "
            f"GPU training, rebuild lightgbm with -DUSE_CUDA=1 (preferred "
            f"for an RTX 3060 Ti) or -DUSE_GPU=1 (OpenCL) -- see "
            f"https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html\n",
            file=sys.stderr,
        )
        cpu_params = dict(params)
        cpu_params["device_type"] = "cpu"
        cpu_params.pop("gpu_device_id", None)
        cpu_params.pop("gpu_platform_id", None)
        cpu_params.pop("gpu_use_dp", None)
        booster = lgb.train(
            cpu_params, dtrain, num_boost_round=num_boost_round,
            valid_sets=[dval], valid_names=["val"], feval=feval, callbacks=callbacks,
        )
        return booster, "cpu (fell back from " + requested_device + ")"


# ============================================================================
# Single training run (used directly, and as the unit of work for --tune)
# ============================================================================
def run_single_training(
    X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
    categorical_columns: List[str], label_gain: List[int], relevant_threshold: int,
    args_dict: dict, device: str, gpu_device_id: Optional[int],
) -> Tuple["lgb.Booster", Dict[str, float], Dict[str, float], str]:
    dtrain = lgb.Dataset(
        X_train, label=y_train, group=[len(y_train)],
        categorical_feature=categorical_columns, free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val, label=y_val, group=[len(y_val)],
        categorical_feature=categorical_columns, reference=dtrain, free_raw_data=False,
    )

    label_gain_arr = np.array(label_gain, dtype=float)
    feval = make_hackathon_feval(relevant_threshold, label_gain_arr)

    params = build_params(
        objective=args_dict["objective"], boosting=args_dict["boosting"],
        num_leaves=args_dict["num_leaves"], min_data_in_leaf=args_dict["min_data_in_leaf"],
        learning_rate=args_dict["learning_rate"], feature_fraction=args_dict["feature_fraction"],
        bagging_fraction=args_dict["bagging_fraction"], bagging_freq=args_dict["bagging_freq"],
        lambda_l1=args_dict["lambda_l1"], lambda_l2=args_dict["lambda_l2"],
        max_depth=args_dict["max_depth"], min_gain_to_split=args_dict["min_gain_to_split"],
        max_position=args_dict["max_position"], label_gain=label_gain,
        seed=args_dict["seed"], num_threads=args_dict["num_threads"],
        device=device, gpu_device_id=gpu_device_id,
    )

    booster, device_used = train_with_device_fallback(
        params, dtrain, dval, args_dict["num_boost_round"],
        args_dict["early_stopping_rounds"], feval, log_period=args_dict.get("log_period", 50),
    )

    best_iter = booster.best_iteration or booster.current_iteration()
    train_pred = booster.predict(X_train, num_iteration=best_iter)
    val_pred = booster.predict(X_val, num_iteration=best_iter)
    train_metrics = compute_all_metrics(y_train, train_pred, relevant_threshold, label_gain_arr)
    val_metrics = compute_all_metrics(y_val, val_pred, relevant_threshold, label_gain_arr)

    return booster, train_metrics, val_metrics, device_used


# ============================================================================
# --tune: parallel random hyperparameter search across up to 2 GPUs (or CPU)
# ============================================================================
SEARCH_SPACE = {
    "num_leaves": [15, 31, 63],
    "min_data_in_leaf": [10, 20, 30, 50, 80],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "feature_fraction": [0.6, 0.7, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.7, 0.8, 1.0],
    "lambda_l1": [0.0, 0.1, 0.5, 1.0],
    "lambda_l2": [0.0, 0.1, 0.5, 1.0],
    "max_position": [10, 20, 30, 50],
}


def _sample_trial_params(rng: np.random.RandomState, base_args: dict) -> dict:
    sampled = dict(base_args)
    for k, choices in SEARCH_SPACE.items():
        sampled[k] = choices[rng.randint(len(choices))]
    return sampled


def _run_trial_worker(payload: dict) -> dict:
    """Top-level (picklable) function executed in a worker process for
    --tune. Re-imports everything it needs since it runs in a fresh
    interpreter."""
    import lightgbm as lgb_local  # noqa: F401 -- ensures import error surfaces per-worker, not silently
    trial_id = payload["trial_id"]
    try:
        X_train = pd.read_pickle(payload["train_X_path"])
        y_train = np.load(payload["train_y_path"])
        X_val = pd.read_pickle(payload["val_X_path"])
        y_val = np.load(payload["val_y_path"])

        booster, train_m, val_m, device_used = run_single_training(
            X_train, y_train, X_val, y_val,
            payload["categorical_columns"], payload["label_gain"],
            payload["relevant_threshold"], payload["trial_args"],
            payload["device"], payload["gpu_device_id"],
        )
        return {
            "trial_id": trial_id, "ok": True, "device_used": device_used,
            "val_metrics": val_m, "train_metrics": train_m,
            "params": payload["trial_args"], "best_iteration": booster.best_iteration,
        }
    except Exception as e:  # noqa: BLE001 -- a single failed trial must not kill the search
        return {"trial_id": trial_id, "ok": False, "error": f"{type(e).__name__}: {e}"}


def run_hyperparameter_search(
    X_train, y_train, X_val, y_val, categorical_columns, label_gain, relevant_threshold,
    base_args: dict, device: str, gpu_ids: List[int], n_trials: int, tmp_dir: Path, seed: int,
) -> List[dict]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    train_X_path = tmp_dir / "tune_train_X.pkl"
    train_y_path = tmp_dir / "tune_train_y.npy"
    val_X_path = tmp_dir / "tune_val_X.pkl"
    val_y_path = tmp_dir / "tune_val_y.npy"
    X_train.to_pickle(train_X_path)
    np.save(train_y_path, y_train)
    X_val.to_pickle(val_X_path)
    np.save(val_y_path, y_val)

    rng = np.random.RandomState(seed)
    # Each trial gets a smaller boosting budget than the final retrain --
    # the search is for finding a good REGION of hyperparameter space, not
    # for squeezing out the last 0.1% with a full-budget run per candidate.
    trial_base = dict(base_args)
    trial_base["num_boost_round"] = max(300, base_args["num_boost_round"] // 4)
    trial_base["early_stopping_rounds"] = max(40, base_args["early_stopping_rounds"] // 3)

    if device == "cpu":
        n_workers = max(1, min(4, (os.cpu_count() or 4)))
        per_trial_threads = max(1, (os.cpu_count() or n_workers) // n_workers)
        device_for_trial = lambda i: ("cpu", None)
    else:
        if any(g == 0 for g in gpu_ids):
            raise ValueError("--gpu-ids must not include 0 (thermal-throttling card).")
        n_workers = min(len(gpu_ids), n_trials) if gpu_ids else 1
        per_trial_threads = max(1, (os.cpu_count() or 4) // max(n_workers, 1))
        device_for_trial = lambda i: (device, gpu_ids[i % len(gpu_ids)])

    print(f"\n[tune] {n_trials} trials, {n_workers} parallel worker(s), "
          f"device={device}, gpu_ids={gpu_ids if device != 'cpu' else 'n/a'}, "
          f"{per_trial_threads} LightGBM thread(s)/trial.")

    payloads = []
    for i in range(n_trials):
        trial_args = _sample_trial_params(rng, trial_base)
        trial_args["num_threads"] = per_trial_threads
        dev, gid = device_for_trial(i)
        payloads.append({
            "trial_id": i, "train_X_path": str(train_X_path), "train_y_path": str(train_y_path),
            "val_X_path": str(val_X_path), "val_y_path": str(val_y_path),
            "categorical_columns": categorical_columns, "label_gain": label_gain,
            "relevant_threshold": relevant_threshold, "trial_args": trial_args,
            "device": dev, "gpu_device_id": gid,
        })

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_run_trial_worker, p): p["trial_id"] for p in payloads}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if r["ok"]:
                print(f"  [trial {r['trial_id']:>3}] device={r['device_used']:<22} "
                      f"val_composite={r['val_metrics']['hackathon_composite']:.4f}  "
                      f"(ndcg@10={r['val_metrics']['ndcg@10']:.3f}, "
                      f"ndcg@50={r['val_metrics']['ndcg@50']:.3f}, "
                      f"map={r['val_metrics']['map']:.3f}, p@10={r['val_metrics']['p@10']:.3f})")
            else:
                print(f"  [trial {r['trial_id']:>3}] FAILED: {r['error']}")
    print(f"[tune] {len(results)} trials finished in {time.time() - t0:.1f}s.")

    for p in (train_X_path, train_y_path, val_X_path, val_y_path):
        try:
            p.unlink()
        except OSError:
            pass

    return sorted(
        [r for r in results if r["ok"]],
        key=lambda r: r["val_metrics"]["hackathon_composite"], reverse=True,
    )


# ============================================================================
# Reporting
# ============================================================================
def print_metrics_table(train_metrics: dict, val_metrics: dict) -> None:
    print(f"\n{'metric':<22}{'train':>10}{'val':>10}{'gap':>10}")
    for k in ["hackathon_composite", "ndcg@10", "ndcg@50", "map", "p@10"]:
        gap = train_metrics[k] - val_metrics[k]
        print(f"{k:<22}{train_metrics[k]:>10.4f}{val_metrics[k]:>10.4f}{gap:>10.4f}")
    if train_metrics["hackathon_composite"] - val_metrics["hackathon_composite"] > 0.15:
        print(
            "\n  NOTE: train/val gap on hackathon_composite exceeds 0.15 -- this is a "
            "real overfitting signal worth acting on (raise min_data_in_leaf / "
            "lambda_l1 / lambda_l2, lower num_leaves, or lower learning_rate and "
            "let early stopping pick a later-but-flatter iteration), not just a "
            "single-query-split quirk. See the module docstring in "
            "10_assemble_lgbm_training_set.py for why this split can't be a "
            "textbook cross-query holdout."
        )


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset-dir", required=True, type=Path,
                     help="Directory from 10_assemble_lgbm_training_set.py "
                          "(train/val_features.csv, train/val_labels.csv, "
                          "dataset_metadata.json).")
    ap.add_argument("--out-dir", required=True, type=Path,
                     help="Where to write lgbm_ranker.txt + model_metadata.json.")

    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"],
                     help="'cpu' (default, recommended at this dataset size -- "
                          "see module docstring), 'gpu' (OpenCL), or 'cuda' "
                          "(NVIDIA CUDA, the better target for an RTX 3060 Ti "
                          "if you build LightGBM with GPU support).")
    ap.add_argument("--gpu-device-id", type=int, default=1,
                     help="Single-run GPU index. Must be 1 or 2 -- 0 is refused "
                          "(thermal-throttling card).")
    ap.add_argument("--gpu-ids", default="1,2",
                     help="Comma-separated GPU indices available for --tune's "
                          "parallel trials. Must not include 0.")

    ap.add_argument("--objective", default="lambdarank", choices=["lambdarank", "rank_xendcg"])
    ap.add_argument("--boosting", default="gbdt", choices=["gbdt", "dart", "goss"],
                     help="gbdt is the recommended default at this dataset size "
                          "(see module docstring re: goss/dart trade-offs).")
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--min-data-in-leaf", type=int, default=25)
    ap.add_argument("--max-depth", type=int, default=-1)
    ap.add_argument("--learning-rate", type=float, default=0.04)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-freq", type=int, default=1)
    ap.add_argument("--lambda-l1", type=float, default=0.1)
    ap.add_argument("--lambda-l2", type=float, default=0.1)
    ap.add_argument("--min-gain-to-split", type=float, default=0.0)
    ap.add_argument("--max-position", type=int, default=30,
                     help="lambdarank_truncation_level: how deep into the "
                          "ranked list NDCG-based lambda gradients are "
                          "computed. 30 sits between the metric's NDCG@10 "
                          "(50%% weight) and NDCG@50 (30%% weight) cutoffs.")
    ap.add_argument("--num-boost-round", type=int, default=3000)
    ap.add_argument("--early-stopping-rounds", type=int, default=150)
    ap.add_argument("--relevant-grade-threshold", type=int, default=3,
                     help="relevance_grade >= this is treated as 'relevant' "
                          "for this script's own MAP/P@10 diagnostics -- a "
                          "local proxy for the hidden ground truth's 'tier "
                          "3+' definition (Section 4 of submission_spec.md), "
                          "chosen to line up with this dataset's own "
                          "rank<=50 grade boundary.")
    ap.add_argument("--num-threads", type=int, default=0,
                     help="0 = let LightGBM use all visible cores. Lowered "
                          "automatically per-trial during --tune.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-period", type=int, default=50)

    ap.add_argument("--tune", action="store_true",
                     help="Run a parallel random hyperparameter search first "
                          "(see SEARCH_SPACE), then retrain the best config "
                          "found at full --num-boost-round budget.")
    ap.add_argument("--n-trials", type=int, default=24)
    ap.add_argument("--tune-tmp-dir", type=Path, default=None,
                     help="Defaults to <out-dir>/tune_tmp.")

    return ap.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    metadata = load_metadata(args.dataset_dir)
    feature_columns: List[str] = metadata["feature_columns"]
    categorical_columns: List[str] = metadata["categorical_columns"]
    label_gain: List[int] = metadata["label_gain"]

    print(f"Loaded dataset metadata: {len(feature_columns)} feature columns, "
          f"{len(categorical_columns)} categorical, "
          f"train_group_size={metadata['train_group_size']}, "
          f"val_group_size={metadata['val_group_size']}.")

    X_train, y_train, _ = load_split(args.dataset_dir, "train", feature_columns, categorical_columns)
    X_val, y_val, _ = load_split(args.dataset_dir, "val", feature_columns, categorical_columns)
    print(f"Loaded {len(y_train)} train rows / {len(y_val)} val rows.")

    if args.gpu_device_id == 0:
        print("ERROR: --gpu-device-id 0 is refused (thermal-throttling card). "
              "Use 1 or 2.", file=sys.stderr)
        sys.exit(1)

    base_args = vars(args).copy()

    chosen_params = None  # populated if --tune ran, used for the final report

    if args.tune:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
        tmp_dir = args.tune_tmp_dir or (args.out_dir / "tune_tmp")
        ranked = run_hyperparameter_search(
            X_train, y_train, X_val, y_val, categorical_columns, label_gain,
            args.relevant_grade_threshold, base_args, args.device, gpu_ids,
            args.n_trials, tmp_dir, args.seed,
        )
        if not ranked:
            print("ERROR: every --tune trial failed; see error messages above. "
                  "Falling back to the single default-hyperparameter run below.",
                  file=sys.stderr)
        else:
            best = ranked[0]
            print(f"\n[tune] Best trial #{best['trial_id']}: "
                  f"val_composite={best['val_metrics']['hackathon_composite']:.4f}")
            print(f"[tune] Best params: { {k: best['params'][k] for k in SEARCH_SPACE} }")
            chosen_params = {k: best["params"][k] for k in SEARCH_SPACE}
            base_args.update(chosen_params)

    print(f"\nFinal training run -- device={args.device}"
          + (f", gpu_device_id={args.gpu_device_id}" if args.device != "cpu" else "")
          + f", objective={base_args['objective']}, boosting={base_args['boosting']}"
          + (" (hyperparameters selected by --tune)" if chosen_params else " (CLI/defaults)"))

    booster, train_metrics, val_metrics, device_used = run_single_training(
        X_train, y_train, X_val, y_val, categorical_columns, label_gain,
        args.relevant_grade_threshold, base_args, args.device, args.gpu_device_id,
    )

    print_metrics_table(train_metrics, val_metrics)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "lgbm_ranker.txt"
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)

    elapsed = time.time() - t_start
    model_metadata = {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "categorical_shift": CATEGORICAL_SHIFT,
        "bool_like_columns": BOOL_LIKE_COLUMNS,
        "label_gain": label_gain,
        "relevant_grade_threshold": args.relevant_grade_threshold,
        "objective": base_args["objective"],
        "boosting": base_args["boosting"],
        "hyperparameters": {
            k: base_args[k] for k in
            ["num_leaves", "min_data_in_leaf", "max_depth", "learning_rate",
             "feature_fraction", "bagging_fraction", "bagging_freq",
             "lambda_l1", "lambda_l2", "min_gain_to_split", "max_position"]
        },
        "tuned_via_search": bool(chosen_params),
        "best_iteration": booster.best_iteration,
        "num_boost_round_requested": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "device_requested": args.device,
        "device_used": device_used,
        "gpu_device_id_requested": args.gpu_device_id if args.device != "cpu" else None,
        "seed": args.seed,
        "training_wall_clock_seconds": round(elapsed, 1),
        "lightgbm_version": lgb.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_dataset_dir": str(args.dataset_dir),
    }
    meta_path = args.out_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    print(f"\nWrote model -> {model_path}")
    print(f"Wrote model metadata -> {meta_path}")
    print(f"Best iteration: {booster.best_iteration} / {args.num_boost_round} requested.")
    print(f"Total wall-clock time: {elapsed:.1f}s "
          f"(training is NOT subject to the 5-minute submission budget -- "
          f"that applies only to 12_predict_lgbm_cpu.py).")


if __name__ == "__main__":
    main()