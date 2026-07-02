#!/usr/bin/env python3
"""
Build per-candidate text documents and shard them for scoring.

Reads candidates from:
  - a gzipped JSONL file
  - a plain JSONL file
  - a plain JSON array file

For each candidate, the script builds a compact text document for the
cross-encoder and writes the records round-robin into shard files so each
worker can process a smaller slice from disk.

Usage:
    # quick correctness check on the 50-candidate sample (single shard)
    python 01_prepare_documents.py --candidates sample_candidates.json --shards 1 --out-dir shards_test

    # full run, one shard per GPU
    python 01_prepare_documents.py --candidates candidates.jsonl --shards 3 --out-dir shards
"""
import argparse
import gzip
import json
import os

# Dense, factual restatement of the JD requirements used as the scoring query.
# Keep this text aligned with the hackathon target role when needed.
DEFAULT_QUERY = (
    "Senior AI Engineer role, 5 to 9 years of experience. Must have "
    "production experience with embeddings-based retrieval systems such as "
    "sentence-transformers, OpenAI embeddings, BGE, or E5, deployed to real "
    "users. Must have production experience with vector databases or "
    "hybrid search infrastructure such as Pinecone, Weaviate, Qdrant, "
    "Milvus, OpenSearch, Elasticsearch, or FAISS. Strong Python programming. "
    "Hands-on experience designing evaluation frameworks for ranking "
    "systems, including NDCG, MRR, MAP, and A/B test interpretation. Has "
    "shipped a ranking, search, or recommendation system to production at a "
    "product company, not only at a pure research lab or a services and "
    "consulting company. Comfortable with LLM fine-tuning and hybrid "
    "retrieval architecture decisions. Located in or willing to relocate to "
    "Pune, Noida, Hyderabad, Mumbai, or Delhi NCR, India."
)


def fmt_money(d):
    if not d:
        return "unspecified"
    return f"{d.get('min', '?')}-{d.get('max', '?')} LPA"


def build_doc_text(c):
    """Flatten one candidate record into a single text block for scoring."""
    p = c.get("profile", {}) or {}
    sig = c.get("redrob_signals", {}) or {}
    parts = []

    parts.append(f"Headline: {p.get('headline', '')}")
    parts.append(
        f"Current role: {p.get('current_title', '')} at "
        f"{p.get('current_company', '')} "
        f"({p.get('current_company_size', '')} employees, "
        f"{p.get('current_industry', '')} industry)"
    )
    parts.append(
        f"Experience: {p.get('years_of_experience', '')} years. "
        f"Location: {p.get('location', '')}, {p.get('country', '')}."
    )
    parts.append(f"Summary: {p.get('summary', '')}")

    history = c.get("career_history", []) or []
    if history:
        parts.append("Career history:")
        for h in history[:6]:
            parts.append(
                f"- {h.get('title', '')} at {h.get('company', '')} "
                f"({h.get('industry', '')}, {h.get('company_size', '')} "
                f"employees), {h.get('duration_months', 0)} months. "
                f"{h.get('description', '')}"
            )

    edu = c.get("education", []) or []
    if edu:
        parts.append("Education:")
        for e in edu[:4]:
            parts.append(
                f"- {e.get('degree', '')} in {e.get('field_of_study', '')} "
                f"from {e.get('institution', '')} "
                f"({e.get('start_year', '')}-{e.get('end_year', '')})"
            )

    skills = c.get("skills", []) or []
    if skills:
        skill_strs = [
            f"{s.get('name', '')} ({s.get('proficiency', '')}, "
            f"{s.get('duration_months', 0)}mo, "
            f"{s.get('endorsements', 0)} endorsements)"
            for s in skills[:25]
        ]
        parts.append("Skills: " + "; ".join(skill_strs))

    certs = c.get("certifications", []) or []
    if certs:
        parts.append(
            "Certifications: "
            + "; ".join(
                f"{x.get('name', '')} ({x.get('issuer', '')}, {x.get('year', '')})"
                for x in certs[:10]
            )
        )

    parts.append(
        f"Open to work: {sig.get('open_to_work_flag', '')}. "
        f"Preferred work mode: {sig.get('preferred_work_mode', '')}. "
        f"Willing to relocate: {sig.get('willing_to_relocate', '')}. "
        f"Notice period: {sig.get('notice_period_days', '')} days. "
        f"Expected salary: {fmt_money(sig.get('expected_salary_range_inr_lpa'))}."
    )

    return "\n".join(parts)


def iter_candidates(path):
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
        # Plain JSON array file, for example sample_candidates.json.
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data:
            yield c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl.gz, a .jsonl file, or a JSON array file")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "shards"))
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--query-text", default=None, help="Override the default JD query text")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    query = args.query_text or DEFAULT_QUERY
    query_path = os.path.join(args.out_dir, "query.txt")
    with open(query_path, "w", encoding="utf-8") as f:
        f.write(query)

    shard_paths = [os.path.join(args.out_dir, f"shard_{i}.jsonl") for i in range(args.shards)]
    shard_files = [open(p, "w", encoding="utf-8") for p in shard_paths]

    n = 0
    skipped = 0
    try:
        for c in iter_candidates(args.candidates):
            cid = c.get("candidate_id")
            if not cid:
                skipped += 1
                continue
            doc_text = build_doc_text(c)
            rec = {"candidate_id": cid, "doc_text": doc_text}
            shard_idx = n % args.shards
            shard_files[shard_idx].write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 5000 == 0:
                print(f"  processed {n} candidates...")
    finally:
        for f in shard_files:
            f.close()

    print(f"Done. Wrote {n} candidates across {args.shards} shard(s) to {args.out_dir}/")
    if skipped:
        print(f"Skipped {skipped} records with no candidate_id.")
    print(f"Query saved to {query_path} -- edit this file directly to tune what the reranker scores against.")
    for p in shard_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()