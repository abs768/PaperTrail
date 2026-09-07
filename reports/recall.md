# PaperTrail Retrieval — Configuration Ablation

Corpus: 15 papers, 120 chunks, 45 queries. Each lane is the application's own retrieval code, isolated: `dense` is the Chroma half, `bm25` the lexical half, `hybrid` the RRF fusion the app actually uses, and `hybrid+rerank` that fusion followed by the cross-encoder.

**No quality metric separates these configurations on this corpus.** Recall@5 and Recall@10 saturate at 100% for every lane, and Recall@1 is identical across all of them, so the only column that actually distinguishes the configurations is latency. A 120-chunk corpus with one planted answer per paper is too small a haystack to rank retrieval strategies: this table demonstrates that the ablation runs against the real code paths, not that the lanes are equivalent in general. Read it as evidence about the benchmark as much as about the retriever.

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| dense (Chroma only) | 77.8% | 100.0% | 100.0% | 0.869 |
| BM25 only | 77.8% | 100.0% | 100.0% | 0.870 |
| hybrid (RRF fusion) | 77.8% | 100.0% | 100.0% | 0.870 |
| hybrid + cross-encoder rerank | 77.8% | 100.0% | 100.0% | 0.870 |

Median latency is omitted from this file so it regenerates byte-identically and a rerun produces an empty diff. It is not reproducible enough to commit: measured spread over four back-to-back runs was 0.01 ms for BM25 and 0.09–0.17 ms for the dense and hybrid lanes, but 6.8 ms for the cross-encoder — so no rounding is both stable and honest. Every run prints it to stdout, and `--with-latency` puts it back in this table.

## Recall@5 by query type — hybrid (the shipped configuration)

| Type | Recall@5 |
|---|---:|
| explicit | 100.0% |
| paraphrase | 100.0% |
| semantic | 100.0% |