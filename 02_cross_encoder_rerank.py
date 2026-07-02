#!/usr/bin/env python3
"""
Step 2 — cross-encoder reranking for one candidate shard.

This worker scores one shard of candidate documents against the saved JD
query and appends a CSV of cross-encoder scores. It is designed for the
hackathon pipeline to be:
- GPU-friendly: run one shard per GPU
- resumable: already-scored candidate_id values are skipped on restart
- Docker-friendly: all input/output paths are taken from the arguments you pass

Run once per GPU (or let run_all.py launch all of them for you):
python 02_cross_encoder_rerank.py \
    --shard shards/shard_0.jsonl \
    --query-file shards/query.txt \
    --out outputs/scores_0.csv \
    --gpu-id 0 \
    --model-name BAAI/bge-reranker-v2-m3

Recommended models for an 8GB GPU:
    BAAI/bge-reranker-v2-m3
    cross-encoder/ms-marco-MiniLM-L-12-v2
"""
import argparse
import csv
import json
import os
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def iter_shard(shard_path):
    with open(shard_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield rec["candidate_id"], rec["doc_text"]


def already_done(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    done.add(row[0])
    return done


def score_pairs(model, tokenizer, device, query, texts, max_length):
    inputs = tokenizer(
        [query] * len(texts),
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits.view(-1).float()
    return logits.sigmoid().cpu().tolist()


def score_with_backoff(model, tokenizer, device, query, texts, max_length):
    """Split the batch on CUDA OOM and retry instead of failing the whole run."""
    try:
        return score_pairs(model, tokenizer, device, query, texts, max_length)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() or len(texts) == 1:
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        mid = len(texts) // 2
        left = score_with_backoff(model, tokenizer, device, query, texts[:mid], max_length)
        right = score_with_backoff(model, tokenizer, device, query, texts[mid:], max_length)
        return left + right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--query-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--model-name", default="models/bge-reranker-v2-m3")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--cpu", action="store_true", help="Force CPU (for testing without a GPU)")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    out_parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_parent or ".", exist_ok=True)

    torch.set_num_threads(2)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Running on CPU (no GPU detected or --cpu passed).")
    else:
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Running on {torch.cuda.get_device_name(args.gpu_id)} (cuda:{args.gpu_id})")

    with open(args.query_file, "r", encoding="utf-8") as f:
        query = f.read().strip()

    print(f"Loading model {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        model.half()

    done_ids = already_done(args.out)
    if done_ids:
        print(f"Resuming: {len(done_ids)} candidates already scored, will be skipped.")

    write_header = not os.path.exists(args.out)
    out_f = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.writer(out_f)
    if write_header:
        writer.writerow(["candidate_id", "cross_encoder_score"])
        out_f.flush()

    batch_ids, batch_texts = [], []
    processed, skipped = 0, 0
    t0 = time.time()

    def flush_batch():
        nonlocal batch_ids, batch_texts, processed
        if not batch_ids:
            return
        scores = score_with_backoff(model, tokenizer, device, query, batch_texts, args.max_length)
        for cid, sc in zip(batch_ids, scores):
            writer.writerow([cid, f"{sc:.6f}"])
        processed += len(batch_ids)
        out_f.flush()
        batch_ids = []
        batch_texts = []

    for cid, doc_text in iter_shard(args.shard):
        if cid in done_ids:
            skipped += 1
            continue
        batch_ids.append(cid)
        batch_texts.append(doc_text)
        if len(batch_ids) >= args.batch_size:
            flush_batch()
            if processed % args.log_every == 0 and processed > 0:
                rate = processed / (time.time() - t0)
                print(f"  scored {processed} (skipped {skipped} already-done) "
                      f"-- {rate:.1f} candidates/sec")

    flush_batch()
    out_f.close()

    elapsed = time.time() - t0
    rate = processed / elapsed if elapsed > 0 else 0
    print(f"Done. Scored {processed} new candidates in {elapsed:.1f}s ({rate:.1f}/sec). "
          f"Output: {args.out}")


if __name__ == "__main__":
    main()