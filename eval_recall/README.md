# Retrieval Evaluation — Recall@k / MRR

Measures PaperTrail's hybrid retrieval quality (ChromaDB dense + BM25 + RRF)
directly — no LLM needed, since Recall@k is a property of retrieval, not answer
generation.

```bash
python -m eval_recall.recall_eval                    # controlled corpus
python -m eval_recall.recall_eval --labeled q.jsonl  # your own labeled queries
pytest eval_recall/                                  # metric-logic unit tests
```

## Controlled benchmark

`corpus.py` builds an invented corpus (15 papers, 120 chunks) where relevance is
defined by construction: each paper has one planted result passage, datasets are
shared across papers (so a dataset name alone can't disambiguate), and queries
range from lexically explicit to semantic-only. This makes Recall well-defined
without hand labeling, while staying non-trivial.

Result on this benchmark (all-MiniLM-L6-v2 embeddings + BM25 + RRF):

| Metric | Value |
|---|---:|
| Recall@1 | 77.8% |
| Recall@5 | 100.0% |
| Recall@10 | 100.0% |
| MRR | 0.870 |

These are honest numbers on a *constructed* corpus — they validate the pipeline
and the harness, not a claim about arbitrary real papers.

## Real evaluation

To measure on real papers: index them through the normal ingest path, write a
JSONL of `{"query": ..., "gold_chunk_id": "<paper>_chunk_<i>"}` judgments, and
run with `--labeled`. The harness reports the same metrics on your data.
