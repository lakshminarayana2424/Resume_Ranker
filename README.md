# Redrob Hackathon v4: Candidate Discovery & Ranking Pipeline

This repository contains the complete machine learning pipeline used to evaluate, score, and rank 100,000 synthetic candidate profiles against a Senior AI Engineer job description.

The architecture is split into three phases to satisfy both ranking quality and the strict hackathon submission constraints (5-minute max runtime, CPU-only, no network access at inference time):

1. **Phase 1 — Teacher Model Pipeline (Heavy Compute / GPU).** Cross-encoder retrieval, deterministic honeypot rules, and LLM-based scoring/auditing are used to build a high-quality "ground truth" ranking. This phase is unrestricted — it can use GPUs, network access, and as much time as it needs.
2. **Phase 2 — Student Distillation (CPU Prep).** The teacher's ranking signal is distilled into a feature matrix and used to train a fast LightGBM LambdaMART "student" model.
3. **Phase 3 — Hackathon Submission (CPU Inference).** The trained LightGBM model, plus an LLM-generated reasoning layer (with a deterministic fallback), produces the final top-100 submission CSV in under 5 minutes on CPU with no network access.

---

## 📂 Repository Structure

```text
.
├── 01_prepare_documents.py
├── 02_cross_encoder_rerank.py
├── 03_merge_results.py
├── run_all.py                      <- multi-GPU launcher for Step 2
├── 04_honeypot_rules.py
├── 05_apply_honeypot_to_ranking.py
├── 06b_honeypot_population_stats.py
├── 06_llm_pointwise_score.py
├── 07_llm_honeypot_check.py
├── 08_build_lgbm_features.py
├── 09_build_lgbm_targets.py
├── 10_assemble_lgbm_training_set.py
├── 11_train_lgbm_ranker.py
├── 12_predict_lgbm_cpu.py          <- optional, offline CPU test of the trained model
├── feature_engineering.py          <- shared feature module (imported by 08, 12, and rank.py)
├── rank.py                         <- MAIN SUBMISSION ENTRY POINT
├── llm_reasoning_realizer.py       <- primary reasoning generator (LLM)
├── reasoning_engine.py             <- deterministic fallback reasoning generator
├── validate_submission.py          <- format validator, run before uploading
├── candidates.jsonl.gz             <- source dataset (100,000 candidates), gzip-compressed
│                                      decompress with: gunzip candidates.jsonl.gz
├── sample_candidates.json          <- first 50 candidates, for quick schema checks
├── sample_submission.csv           <- format reference only, not a real ranking
├── submission_metadata_template.yaml
├── requirements.txt
├── vllm-env/                       <- Python 3.10 virtual environment [NOT in repo]
│                                      create per Setup instructions below
├── models/                         <- model weights [NOT in repo — ~20 GB total]
│   ├── bge-reranker-v2-m3/         <- BAAI/bge-reranker-v2-m3 (cross-encoder, Step 2)
│   ├── Qwen2.5-0.5B-Instruct/      <- GGUF build used by llm_reasoning_realizer.py
│   │   └── qwen2.5-0.5b-instruct-q4_k_m.gguf
│   └── Qwen3-8B-AWQ/               <- Qwen/Qwen3-8B-AWQ (used by Steps 6 and 7)
├── shards/                         <- sharded candidate documents for Step 2
└── outputs/                        <- pre-computed intermediate results (see below)
```

### Pre-computed outputs are included on purpose

Several stages of Phase 1 (cross-encoder scoring over 100K candidates, and especially the two LLM stages) take hours to run end-to-end on this machine's 3 GPUs. To let you verify the pipeline quickly **without re-running days of compute**, the `outputs/` directory ships with everything already generated, including:

- `scores_0.csv`, `scores_1.csv`, `scores_2.csv`, `cross_encoder_ranked.csv`
- `honeypot_flags.csv`, `cross_encoder_ranked_honeypot.csv`
- `honeypot_population_stats.json`
- `llm_pointwise_top30000.csv`, `llm_pointwise_scores.jsonl`
- `final_ranked.csv`, `llm_honeypot_checks.jsonl`
- `lgbm_features_100k.csv`, `lgbm_targets_10000.csv`, `lgbm_dataset_10000/`
- `lgbm_model/`, `scored_pool.csv`, `submission.csv`

This means you can skip straight to **Phase 3** and run `rank.py` against the pre-computed `lgbm_features_100k.csv` and `models/lgbm_model` to get a submission CSV in minutes. If you want to verify the full pipeline yourself, every stage below is independently runnable and will overwrite/regenerate the corresponding file in `outputs/`.

---

## ⚙️ Setup & Installation

> **Prerequisites:** Python 3.10.x, CUDA-capable GPU(s) with drivers installed, and `git` on PATH. All commands below are run from the repo root.

### Step 1 — Clone and decompress the dataset

```bash
git clone <repo-url>
cd <repo-dir>

# The candidate dataset ships as gzip to stay under GitHub's 100 MB file limit.
# Decompress it once — all pipeline scripts expect candidates.jsonl.
gunzip candidates.jsonl.gz
```

### Step 2 — Create the virtual environment

The repo was developed and tested on **Python 3.10.12**. Using a different minor version may cause dependency conflicts with `vllm` and `torch`.

```bash
# confirm you have the right Python
python3.10 --version          # should print Python 3.10.x

# create the venv (name it vllm-env to match the excluded directory)
python3.10 -m venv vllm-env

# activate it — you must do this in every new shell before running any script
source vllm-env/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install --upgrade pip

# install all pinned dependencies
pip install -r requirements.txt
```

> **Note on vLLM:** `vllm==0.23.0` is listed in `requirements.txt` and will be installed by the command above. vLLM has a hard dependency on a matching `torch` build; if pip cannot resolve the CUDA wheel automatically you may need to pre-install `torch==2.11.0` for your CUDA version first, then re-run `pip install -r requirements.txt`.

The LLM reasoning stage in Phase 3 additionally requires `llama-cpp-python` (not in `requirements.txt` because it needs a GPU/CPU build flag to be chosen at install time). Install the CPU build — it is only used during the 5-minute submission run, which is CPU-only:

```bash
pip install llama-cpp-python
```

If you want the CUDA-accelerated build for faster local testing outside the submission window:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Step 4 — Download the models

All three models must be in the `models/` directory before running any pipeline script. The total download is approximately **20 GB**; ensure you have sufficient disk space. The commands below use `huggingface-cli`, which is installed as part of `huggingface_hub` in `requirements.txt`.

```bash
mkdir -p models
```

**4a — BAAI/bge-reranker-v2-m3** (~1.1 GB) — used by Step 2 (`02_cross_encoder_rerank.py`):

```bash
huggingface-cli download BAAI/bge-reranker-v2-m3 \
    --local-dir models/bge-reranker-v2-m3
```

**4b — Qwen/Qwen3-8B-AWQ** (~8.2 GB) — used by Steps 6 and 7 (`06_llm_pointwise_score.py`, `07_llm_honeypot_check.py`):

```bash
huggingface-cli download Qwen/Qwen3-8B-AWQ \
    --local-dir models/Qwen3-8B-AWQ
```

**4c — Qwen/Qwen2.5-0.5B-Instruct GGUF** (~0.4 GB) — used by `llm_reasoning_realizer.py` in Phase 3. Only the single `q4_k_m` GGUF file is needed, not the full model repo:

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --local-dir models/Qwen2.5-0.5B-Instruct
```

After all downloads complete, verify the layout:

```bash
ls models/
# bge-reranker-v2-m3   Qwen2.5-0.5B-Instruct   Qwen3-8B-AWQ

ls models/Qwen2.5-0.5B-Instruct/
# qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### Required environment variable (read before running any vLLM stage)

Steps **06** (`06_llm_pointwise_score.py`) and **07** (`07_llm_honeypot_check.py`) both load `Qwen3-8B-AWQ` through vLLM. On this setup, vLLM's FlashInfer sampler has caused intermittent instability. Disabling it costs a small amount of speed but avoids that instability entirely — set this in every shell before running either script:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

It's easiest to just add this to your shell profile (`~/.bashrc` / `~/.zshrc`) or to the top of the conda environment's activation hook so you don't have to remember it per-session.

---

## 🚀 Step-by-Step Execution Guide

### Phase 1: Ground Truth Generation (GPU-Heavy)

**Step 1 — Prepare documents**
Builds a compact per-candidate text document and shards the pool for cross-encoder scoring.

```bash
# quick correctness check on the 50-candidate sample (single shard)
python 01_prepare_documents.py --candidates sample_candidates.json --shards 1 --out-dir shards_test

# full run, one shard per GPU
python 01_prepare_documents.py --candidates candidates.jsonl --shards 3 --out-dir shards
```

**Step 2 — Cross-encoder reranking**

- **Recommended (default path):** use the multi-GPU launcher, which spawns one worker process per GPU and waits for all of them to finish. This is what was used to produce the pre-computed `scores_*.csv` files.

  ```bash
  python run_all.py --shard-dir shards --out-dir outputs --num-gpus 3
  ```

- **Fallback (if you don't have 3 GPUs):** `run_all.py` only orchestrates Step 2 — if you have fewer GPUs, or just one, run the worker script manually once per shard, changing `--gpu-id` and `--out` for each:

  ```bash
  python 02_cross_encoder_rerank.py \
      --shard shards/shard_0.jsonl \
      --query-file shards/query.txt \
      --out outputs/scores_0.csv \
      --gpu-id 0 \
      --model-name ./models/bge-reranker-v2-m3
  ```

  Repeat for `shard_1.jsonl` / `scores_1.csv`, `shard_2.jsonl` / `scores_2.csv`, etc. If you only have one GPU, you can either run all shards sequentially on `--gpu-id 0`, or re-run Step 1 with `--shards 1` to produce a single shard.

  Recommended models for an 8GB GPU: `BAAI/bge-reranker-v2-m3` or `cross-encoder/ms-marco-MiniLM-L-12-v2`.

**Step 3 — Merge results**
Always run this manually — it's a fast, single-process, CPU-only merge of whatever `scores_*.csv` files exist. It de-duplicates by `candidate_id`, sorts by score descending, and assigns a dense rank.

```bash
python 03_merge_results.py --scores-glob "outputs/scores_*.csv" --out outputs/cross_encoder_ranked.csv
```

**Step 4 — Deterministic honeypot detection**
Applies only hard, deterministic rules (impossible dates, career-span/experience contradictions, expert-skill/zero-duration mismatches) — no LLM involved.

```bash
python 04_honeypot_rules.py --candidates candidates.jsonl --out outputs/honeypot_flags.csv
```

**Step 5 — Apply honeypot flags to the ranking**
Pushes every flagged candidate below every clean candidate while preserving score order within each group.

```bash
python 05_apply_honeypot_to_ranking.py \
    --ranked outputs/cross_encoder_ranked.csv \
    --honeypots outputs/honeypot_flags.csv \
    --out outputs/cross_encoder_ranked_honeypot.csv
```

**Step 6b — Calibrate honeypot thresholds (run before Step 7)**
Scans the *full* 100K pool once to derive tail-percentile thresholds for each honeypot signal (experience-vs-career-span gap, single-role overage, zero-duration expert skills, invalid education dates, role overlaps). Pure Python — no GPU, no model calls. This is a precomputation step, and its output JSON is passed as thresholds into Step 7.

```bash
python 06b_honeypot_population_stats.py \
    --candidates candidates.jsonl \
    --reference-date 2025-05-27 \
    --out outputs/honeypot_population_stats.json
```

**Step 6 — LLM pointwise scoring**
Scores the top 30,000 candidates (post cross-encoder + rule-based honeypot filtering) using Qwen3-8B-AWQ via vLLM, with a five-dimension rubric (retrieval depth, evaluation rigor, applied ML engineering, credibility, bonus/availability signals). Remember to `export VLLM_USE_FLASHINFER_SAMPLER=0` first.

```bash
python 06_llm_pointwise_score.py \
    --ranked outputs/cross_encoder_ranked_honeypot.csv \
    --candidates candidates.jsonl \
    --top-n 30000 \
    --out outputs/llm_pointwise_top30000.csv \
    --model-name ./models/Qwen3-8B-AWQ \
    --num-gpus 1 \
    --max-model-len 6144 \
    --gpu-memory-utilization 0.75 \
    --enforce-eager \
    --batch-size 1
```

This stage is resumable — it caches every completed scoring pass to `outputs/llm_pointwise_scores.jsonl`, so an interrupted run can pick up where it left off.

**Step 7 — LLM honeypot audit**
Final integrity-check layer. Walks down the merged pool (pointwise-ranked top 30K, then the remaining cross-encoder ranking) until it has collected exactly `--top-k` clean candidates, auditing each against the Step 6b thresholds using Qwen3-8B-AWQ. Any newly-discovered honeypot is pushed to the very bottom and backfilled from the next-best candidate. Remember to `export VLLM_USE_FLASHINFER_SAMPLER=0` first.

```bash
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
```

This stage is also resumable via `outputs/llm_honeypot_checks.jsonl`.

---

### Phase 2: Student Model Distillation (LightGBM)

**Step 8 — Build LGBM features**
Builds one feature row per candidate for the *entire* 100K pool, using only deterministic logic in `feature_engineering.py` (redrob_signals fields + structured career/education/skills fields). No LLM scores are used as inputs here — they aren't available for the full pool, and using them would create train/inference skew since they also won't be available at submission time.

```bash
python 08_build_lgbm_features.py \
    --candidates candidates.jsonl \
    --out outputs/lgbm_features_100k.csv
```

**Step 9 — Build LGBM targets**
Converts the Step 7 rank output into small integer relevance grades (rank 1–5 → grade 6, down to rank 1001+ → grade 0), with grade boundaries deliberately pinned to the NDCG@10 / NDCG@50 evaluation cutoffs. Any candidate flagged as a honeypot is force-set to grade 0 regardless of LLM rank.

```bash
python 09_build_lgbm_targets.py \
    --final-ranked-csv outputs/final_ranked.csv \
    --max-rank 10000 \
    --out outputs/lgbm_targets_10000.csv
```

**Step 10 — Assemble the LGBM training set**
Joins the Step 8 features with the Step 9 targets into a grade-stratified train/validation split that LightGBM's `lambdarank` objective can consume directly.

```bash
python 10_assemble_lgbm_training_set.py \
    --features outputs/lgbm_features_100k.csv \
    --targets outputs/lgbm_targets_10000.csv \
    --out-dir outputs/lgbm_dataset_10000
```

**Step 11 — Train the LGBM ranker**
Trains `lightgbm.LGBMRanker` with the `lambdarank` objective and early-stops on a custom eval metric that matches the hackathon composite exactly: `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`.

```bash
# default CPU run
python 11_train_lgbm_ranker.py \
    --dataset-dir outputs/lgbm_dataset_10000 \
    --out-dir outputs/lgbm_model

# single-GPU run (only if your LightGBM build has CUDA support)
python 11_train_lgbm_ranker.py --dataset-dir outputs/lgbm_dataset_10000 \
    --out-dir outputs/lgbm_model --device cuda --gpu-device-id 1

# parallel hyperparameter search across GPUs 1 and 2, then retrain the best config
python 11_train_lgbm_ranker.py --dataset-dir outputs/lgbm_dataset_10000 \
    --out-dir outputs/lgbm_model --device cuda --gpu-ids 1,2 \
    --tune --n-trials 24
```

**Step 12 (optional) — Offline CPU test of the trained model**
Not part of the submission path. Useful for sanity-checking the trained model and pseudo-metrics before running the real submission script.

```bash
# fast path — from the precomputed feature CSV
python 12_predict_lgbm_cpu.py \
    --model-dir outputs/lgbm_model \
    --features-csv outputs/lgbm_features_100k.csv \
    --pseudo-labels-csv outputs/lgbm_targets_10000.csv \
    --final-ranked-csv outputs/final_ranked.csv \
    --out outputs/scored_pool.csv

# slower path — build features live from raw candidates
python 12_predict_lgbm_cpu.py \
    --model-dir outputs/lgbm_model \
    --candidates candidates.jsonl.gz \
    --pseudo-labels-csv outputs/lgbm_targets_10000.csv \
    --final-ranked-csv outputs/final_ranked.csv \
    --out outputs/scored_pool.csv
```

---

### Phase 3: The Hackathon Submission Pipeline

To satisfy the 5-minute, CPU-only, no-network constraint, the submission step is fully isolated from every GPU/LLM stage above. It only depends on the trained LightGBM model and the precomputed feature matrix.

**The one command that produces the submission CSV:**

```bash
python rank.py \
    --model-dir outputs/lgbm_model \
    --features-csv outputs/lgbm_features_100k.csv \
    --raw-candidates candidates.jsonl \
    --out outputs/submission.csv \
    --reasoning-mode llm \
    --llm-model-path models/Qwen2.5-0.5B-Instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --reasoning-time-budget 270
```

`rank.py` is the **official submission entry point**. It:

1. Loads the trained LightGBM ranker from `outputs/lgbm_model`.
2. Loads the full 100K-candidate feature matrix (no live feature computation, no GPU).
3. Scores every candidate on CPU.
4. Applies the hard honeypot veto before selecting the top 100.
5. Computes TreeSHAP contributions for just the final top-100 rows.
6. Loads raw candidate facts only for those top-100 rows.
7. Generates the human-readable `reasoning` text for each.
8. Writes `submission.csv` in the exact `candidate_id,rank,score,reasoning` format required by `validate_submission.py`.

It does **not** retrain the model, rebuild the feature matrix from raw JSON, use a GPU, or make any network/API call.

#### Reasoning architecture: LLM primary, template fallback

Because the hackathon expects 1–2 sentences of grounded reasoning for each of the top 100 rows, this is handled by two separate modules with a clear priority order:

| Module | Role |
| :- | :- |
| `llm_reasoning_realizer.py` | **Primary reasoning generator.** Runs a single persistent local GGUF model (Qwen2.5-0.5B-Instruct) on CPU via `llama-cpp-python`, with a per-row watchdog timeout and a grounding verifier that rejects fabricated numbers, ungrounded entities, or tone that contradicts the candidate's rank. This is the path that is evaluated under the default `--reasoning-mode llm`. |
| `reasoning_engine.py` | **Deterministic fallback only.** Renders reasoning text directly from the trained model's own TreeSHAP feature contributions for that row, combined with the candidate's raw profile facts. It is never invoked unless the LLM path is unavailable, exceeds its time budget, or fails grounding verification. |

**Why keep a fallback at all?** So a single bad LLM response, a missing model file, or a per-row timeout can't turn a working ranking into a failed submission. The fallback:

- guarantees `rank.py` always produces a complete, valid `submission.csv` even if the local LLM crashes or is unavailable,
- demonstrates general robustness in the pipeline, and
- keeps the repository reproducible even for a reviewer who can't run the local GGUF model in their environment.

**What matters for evaluation:** under the documented default command above (`--reasoning-mode llm`), the reasoning column is generated by `llm_reasoning_realizer.py` for essentially all 100 rows. `reasoning_engine.py` only fires as a per-row safety net, and both modules are strictly downstream of the ranking — neither one can change a candidate's score or rank, since TreeSHAP contributions are computed *from* the already-final LightGBM score, not the other way around.

> **Note at the top of `reasoning_engine.py`:**
> This module is **not part of the normal ranking pipeline**. The official submission uses **LLM-based reasoning generation** implemented in `llm_reasoning_realizer.py`. This file exists only as a deterministic fallback to guarantee that the ranking pipeline can still complete and produce a valid submission if the local LLM model is unavailable, reasoning generation exceeds the allotted time budget, or a generated response fails grounding verification. Under a correctly configured repository and the recommended reproduce command, this module is invoked only per-row on failure, and has no influence on ranking scores.

#### Validate before submitting

```bash
python validate_submission.py outputs/submission.csv
```

This checks the header, row count, rank/candidate_id uniqueness, score monotonicity, and tie-break ordering exactly as the hackathon's server-side validator does — run it before every upload.

---

## Compute Constraints Recap (Phase 3 only)

| Constraint | Limit |
| :- | :- |
| Runtime | ≤ 5 minutes wall-clock |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only — no GPU |
| Network | Off — no calls to any hosted API |
| Disk | ≤ 5 GB intermediate state |

These constraints apply **only** to `rank.py` (Phase 3). Phases 1 and 2 are precomputation and may freely use GPUs, network access, and unbounded time — that's the entire point of the teacher/student split.
