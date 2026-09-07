# PaperTrail Retrieval — Recall Evaluation

Corpus: 15 papers, 120 chunks. 45 queries. Retrieval: ChromaDB dense + BM25 + RRF.

| Metric | Value |
|---|---:|
| Recall@1 | 77.8% |
| Recall@5 | 100.0% |
| Recall@10 | 100.0% |
| MRR | 0.870 |

## Recall@5 by query type
| Type | Recall@5 |
|---|---:|
| explicit | 100.0% |
| paraphrase | 100.0% |
| semantic | 100.0% |