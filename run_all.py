#!/usr/bin/env python3
"""
Launches one cross-encoder worker process per GPU and waits for all workers
to finish.

The launcher resolves sibling paths from the script location so it works the
same way from the repo root, the parent directory, or a Docker container.
Each worker scores one shard with 02_cross_encoder_rerank.py, and the outputs
are written as separate scores_*.csv files for the merge step.

Usage:
    python run_all.py --shard-dir shards --out-dir outputs --num-gpus 3
"""
import argparse
import os
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default="shards")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--num-gpus", type=int, default=3)
    ap.add_argument("--model-name", default="models/bge-reranker-v2-m3")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    shard_dir = _resolve_path(args.shard_dir)
    out_dir = _resolve_path(args.out_dir)
    query_file = os.path.join(shard_dir, "query.txt")
    script = os.path.join(BASE_DIR, "02_cross_encoder_rerank.py")

    os.makedirs(out_dir, exist_ok=True)

    procs = []
    for gpu_id in range(args.num_gpus):
        shard_path = os.path.join(shard_dir, f"shard_{gpu_id}.jsonl")
        out_path = os.path.join(out_dir, f"scores_{gpu_id}.csv")
        cmd = [
            sys.executable, script,
            "--shard", shard_path,
            "--query-file", query_file,
            "--out", out_path,
            "--gpu-id", str(gpu_id),
            "--model-name", args.model_name,
            "--batch-size", str(args.batch_size),
            "--max-length", str(args.max_length),
        ]
        print("Launching:", " ".join(cmd))
        procs.append(subprocess.Popen(cmd))

    exit_codes = [p.wait() for p in procs]
    if any(code != 0 for code in exit_codes):
        print(f"One or more workers failed. Exit codes: {exit_codes}")
        sys.exit(1)
    print("All shards complete.")


if __name__ == "__main__":
    main()