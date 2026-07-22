"""
Measure PaperTrail's hybrid retrieval quality: Recall@1/5/10 and MRR.

Runs the *actual* retrieval stack (ChromaDB dense + BM25, fused with RRF) — no
LLM required, since retrieval quality is independent of answer generation.
Indexes a controlled corpus with known gold passages, or a real labeled set.

    python -m eval_recall.recall_eval                    # controlled corpus
    python -m eval_recall.recall_eval --labeled q.jsonl  # your own queries

Isolation: uses a temp STATE_DIR so it never touches a real library.
"""
import argparse
import os
import statistics
import tempfile


def _setup_isolated_state():
    # Point PaperTrail's storage at a throwaway dir before importing it.
    os.environ["STATE_DIR"] = tempfile.mkdtemp(prefix="pt_eval_")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def index_corpus(papers):
    from papertrail import retrieval, state
    state.reset()
    for p in papers:
        chunks = [{"text": t, "page": i + 1} for i, t in enumerate(p["chunks"])]
        retrieval.add_to_vector_store(p["paper_id"], chunks, {"title": p["title"]})
    return state.collection.count()


def evaluate(queries, ks=(1, 5, 10), top_k=10):
    from papertrail import retrieval
    hits_at = {k: 0 for k in ks}
    reciprocal_ranks = []
    per_type = {}

    for q in queries:
        results = retrieval._search_chunks(q["query"], top_k=top_k)
        retrieved_ids = [r.get("chunk_id") for r in results]
        gold = q["gold_chunk_id"]

        rank = retrieved_ids.index(gold) + 1 if gold in retrieved_ids else None
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in ks:
            if rank is not None and rank <= k:
                hits_at[k] += 1

        t = q.get("type", "all")
        per_type.setdefault(t, []).append(1 if (rank and rank <= 5) else 0)

    n = len(queries)
    return {
        "n_queries": n,
        "recall_at": {k: hits_at[k] / n for k in ks},
        "mrr": statistics.mean(reciprocal_ranks),
        "recall5_by_type": {t: sum(v) / len(v) for t, v in per_type.items()},
    }


def render(metrics, n_chunks, n_papers):
    L = ["# PaperTrail Retrieval — Recall Evaluation\n"]
    L.append(f"Corpus: {n_papers} papers, {n_chunks} chunks. "
             f"{metrics['n_queries']} queries. Retrieval: ChromaDB dense + BM25 + RRF.\n")
    L.append("| Metric | Value |")
    L.append("|---|---:|")
    for k, v in metrics["recall_at"].items():
        L.append(f"| Recall@{k} | {v*100:.1f}% |")
    L.append(f"| MRR | {metrics['mrr']:.3f} |")
    L.append("")
    L.append("## Recall@5 by query type")
    L.append("| Type | Recall@5 |")
    L.append("|---|---:|")
    for t, v in metrics["recall5_by_type"].items():
        L.append(f"| {t} | {v*100:.1f}% |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeled", help="JSONL of {query, gold_chunk_id}; requires a matching indexed library")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n_papers", type=int, default=15)
    ap.add_argument("--out", default="reports/recall.md")
    args = ap.parse_args()

    _setup_isolated_state()
    from eval_recall.corpus import build, load_labeled

    papers, gen_queries = build(seed=args.seed, n_papers=args.n_papers)
    n_chunks = index_corpus(papers)
    queries = load_labeled(args.labeled) if args.labeled else gen_queries

    metrics = evaluate(queries)
    report = render(metrics, n_chunks, len(papers))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
