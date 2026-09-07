---
title: PaperTrail
emoji: 📚
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# PaperTrail: The Research Memory Agent

[![CI](https://github.com/abs768/PaperTrail/actions/workflows/ci.yml/badge.svg)](https://github.com/abs768/PaperTrail/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![React 19](https://img.shields.io/badge/react-19-61dafb.svg)](https://react.dev/)

Drop research papers in; ask questions across all of them. PaperTrail extracts
entities from every PDF into a knowledge graph, indexes the text for hybrid
search, and answers questions with citations it has **verified against the
source text** — every quote is checked character-for-character against the
passage it claims to come from, and anything that fails is dropped rather than
shown.

- **Grounded citations, not claimed ones.** The model emits only a passage
  number and a verbatim quote; the paper title and page are filled in
  server-side. Unverifiable citations never reach the UI.
  → [How grounding works](#grounded-citations)
- **Measured retrieval, not asserted.** Recall@1 77.8%, MRR 0.870 on a controlled
  benchmark, ablated across four retrieval configurations — including the finding
  that fusion and reranking buy nothing measurable on it.
  → [Evaluation](#evaluation)
- **Hybrid retrieval.** ChromaDB dense vectors + BM25, fused with Reciprocal
  Rank Fusion, with an optional cross-encoder rerank stage.
- **Works without an API key**, in a reduced mode — retrieval is fully local.
  → [Without an API key](#without-an-api-key)

## Quick Setup (5 minutes)

### 1. Backend

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key (or GEMINI_API_KEY as fallback)
export GROQ_API_KEY="gsk_your-key-here"
# Optional: pick a different Groq model
# export GROQ_MODEL="llama-3.1-8b-instant"

# Run the server
python main.py
```

Instead of exporting variables you can also `cp .env.example .env` and edit it —
the server loads `.env` automatically. See [.env.example](./.env.example) for
every available knob (models, extraction budgets, admin token, upload limits…).

Optional: for the cross-encoder rerank stage (better retrieval precision, ~2 GB
of extra dependencies), install `pip install -r requirements-reranker.txt` and
run with `ENABLE_RERANKER=1`.

Backend runs at `http://localhost:8000`

### 2. Frontend

The `frontend/` folder is a complete Vite + React app:

```bash
cd frontend
npm install
npm run dev
```

Vite will serve at `http://localhost:5173` and talk to the backend at `:8000`.

### 3. Demo Flow

1. Open `http://localhost:5173` in your browser
2. Go to **Upload** tab → upload 2-3 research PDFs
3. Switch to **Knowledge Graph** → show the auto-generated entity graph
4. Go to **Ask** tab → ask a cross-paper question like:
   - "What methods are used across these papers?"
   - "Which papers evaluate on the same datasets?"
   - "Compare the approaches used in my papers"
5. Show the cited answer with source references

## Architecture

![PaperTrail architecture](./architecture.png)

```
PDF Upload → Text Extraction (PyMuPDF)
           → Entity Extraction (Groq llama-3.3-70b)
           → Knowledge Graph (NetworkX)
           → Vector Embeddings (ChromaDB)

Query → Vector Search (ChromaDB) + BM25 → Reciprocal Rank Fusion
      → Graph Traversal (NetworkX)
      → Answer Generation (Groq llama-3.3-70b)
      → Citation Verification + Faithfulness Check
      → Cited Response
```

The backend lives in the `papertrail/` package:

| Module | Responsibility |
|--------|----------------|
| `config.py` | env config, storage paths, LLM client, structured-output helpers |
| `models.py` | Pydantic models (LLM outputs + API request bodies) |
| `state.py` | knowledge graph + paper metadata + ChromaDB collection, JSON persistence |
| `textproc.py` | PDF text extraction, title recovery, chunking |
| `kgraph.py` | entity canonicalization, graph building/traversal |
| `extraction.py` | LLM entity extraction + source-grounded validation |
| `retrieval.py` | hybrid vector+BM25 retrieval (RRF), optional reranker |
| `query.py` | GraphRAG query pipeline with grounded citations |
| `api.py` | FastAPI app and endpoints |

The frontend mirrors this: `frontend/src/components/` holds one component per
panel (Upload, Knowledge Graph, Ask, Library, Sidebar) plus shared pieces.

## Grounded citations

The usual failure mode of a RAG system is a citation that looks authoritative
and points at nothing. PaperTrail is built so the model structurally cannot
fabricate provenance.

Retrieved passages are handed to the model as a numbered list. For each claim it
makes, it may emit only two things:

- `passage_idx` — which numbered passage it used
- `quote` — a contiguous verbatim span copied from that passage

It does **not** emit the paper title, the page number, or the chunk id. Those
are looked up server-side from `passage_idx` ([`query.py`](papertrail/query.py)),
so a citation's provenance comes from the retrieval index, never from the model.

Every quote is then checked against the full text of the passage it cites
(`_verify_quote`). Matching normalizes the things PDF extraction and language
models genuinely disagree about — smart quotes, en/em dashes, footnote daggers
and asterisks, non-breaking spaces, casing, whitespace — then requires either an
exact substring match or a sliding-window similarity ≥ 0.85, which absorbs OCR
noise without letting a paraphrase through. Quotes under 3 words are rejected as
too weak to ground anything. **Citations that fail verification are dropped from
the response**, not flagged and shown; they are returned separately in
`dropped_sources` for debugging.

A second pass (`_check_faithfulness`) re-reads the answer against the passages
and flags substantive factual claims that nothing supports. The reported
confidence is then bounded from above by that support score, so an answer cannot
present itself as confident and unsupported at the same time.

Entity extraction gets the same treatment at ingest time: extracted entities
that do not actually appear in the source are dropped
([`extraction.py`](papertrail/extraction.py)), with alias-aware matching so a
model that writes "recurrent neural network" is still accepted when the paper
says "RNN".

## Evaluation

Retrieval quality is measured, not asserted. [`eval_recall/`](eval_recall/) runs
the real retrieval stack and ablates it across four configurations, reporting
Recall@k, MRR, and median latency. No LLM is involved, because recall is a
property of retrieval rather than of answer generation. Each lane is the
application's own code isolated — `dense` is `_vector_search_raw`, `bm25` is
`_bm25_search`, `hybrid` is the `_search_chunks` the app actually calls, and
`hybrid+rerank` is that under `ENABLE_RERANKER=1`.

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR | Median latency † |
|---|---:|---:|---:|---:|---:|
| dense (Chroma only) | 77.8% | 100.0% | 100.0% | 0.869 | 66 ms |
| BM25 only | 77.8% | 100.0% | 100.0% | 0.870 | 0.3 ms |
| hybrid (RRF fusion) | 77.8% | 100.0% | 100.0% | 0.870 | 67 ms |
| hybrid + cross-encoder rerank | 77.8% | 100.0% | 100.0% | 0.870 | 165 ms |

**What this shows: on this corpus, RRF fusion does not beat either lane alone —
every configuration returns identical Recall@1 and MRR within 0.001, so the
fusion is buying no measurable accuracy over plain BM25, which is ~200× cheaper.
The cross-encoder likewise earns nothing here: it triples median latency for
zero movement on any quality metric, so on this evidence its ~2 GB of
dependencies are not justified.** The honest reading is that the benchmark is
too easy to separate them rather than that the lanes are truly equivalent — a
120-chunk corpus with one planted answer per paper is a small haystack, and a
result this flat is a finding about the corpus as much as about the retriever.
Anyone choosing a configuration on this basis should build a harder labeled set
first; what the ablation establishes is that the harness exercises the real code
paths and that the current default is not demonstrably better than the cheapest
option.

† Latency is machine-dependent and is **not** in the committed artifact, which
is byte-stable by design; these figures come from a local run and are printed by
every run. Measured spread over four back-to-back runs: 0.01 ms (BM25),
0.09–0.17 ms (dense/hybrid), 6.8 ms (cross-encoder).

Committed artifact: [reports/recall.md](reports/recall.md). Regenerate it with:

```bash
python -m eval_recall.recall_eval                 # all four configs
python -m eval_recall.recall_eval --config bm25   # one lane
python -m eval_recall.recall_eval --with-latency  # include latency in the file
```

`hybrid+rerank` needs `pip install -r requirements-reranker.txt`. Without it the
row is left blank with a stated reason rather than filled in — the harness
verifies the cross-encoder actually loaded, because `_rerank_chunks` falls back
to fused order on failure and would otherwise republish hybrid's numbers under a
rerank label.

The harness isolates itself in a temporary `STATE_DIR`, so running it never
touches your real library. It takes a few seconds.

**What the benchmark is.** A constructed corpus of 15 papers / 120 chunks / 45
queries, where relevance is defined by construction rather than hand-labeled:
each paper contains exactly one planted result passage, which is the gold chunk
for its queries. All names are invented (`ZephyrNet`, `CartoQA`, …) so no real
paper can leak in. Two choices keep it from being trivial — each dataset is
shared by three papers, so a dataset name alone cannot identify the answer; and
each paper gets three query phrasings: `explicit` (all key tokens present),
`paraphrase` (metric dropped, reworded), and `semantic` (method name omitted
entirely, so the retriever must disambiguate among the three papers sharing that
dataset).

**What it does and doesn't show.** These are honest numbers on an invented
corpus: they validate the retrieval pipeline and the harness, not a claim about
arbitrary real papers. Recall@5 saturates at 100% across all three query types
*and* all four configurations, and Recall@1 is flat at 77.8% everywhere — so on
this corpus no quality metric separates the lanes at all, and latency is the only
column that discriminates. A 120-chunk corpus with one planted answer per paper
is simply too small to rank retrieval strategies. Treat the flat result as a
limit of the benchmark, and build a harder labeled set before using it to pick a
configuration.

To evaluate on your own papers, index them, write a JSONL of
`{"query": ..., "gold_chunk_id": "<paper_id>_chunk_<i>"}` judgments, and see
[`eval_recall/README.md`](eval_recall/README.md) for the current state of the
`--labeled` path.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/health | Health check |
| POST | /upload | Upload and process a PDF |
| POST | /upload-url | Ingest a paper from a URL (arXiv URLs also get exact title/authors from the arXiv API) |
| POST | /note | Add a text note |
| POST | /query | Ask a question (GraphRAG) |
| POST | /query/stream | Same, as Server-Sent Events: live pipeline progress, then the result |
| GET | /papers | List all papers |
| GET | /papers/{paper_id} | Fetch a single paper |
| GET | /graph | Get knowledge graph (nodes + edges). `?limit=N` trims large graphs for rendering: papers/notes are always kept, then the highest-degree entities fill the budget |
| GET | /export | Portable JSON backup: paper metadata + knowledge graph. `?include_chunks=true` adds every indexed chunk for a complete snapshot |
| GET | /stats | System statistics |
| DELETE | /papers/{paper_id} | Delete a single paper † |
| DELETE | /reset | Reset everything † |

† If the `ADMIN_TOKEN` env var is set, these require a matching `X-Admin-Token`
header — recommended on public deploys.

## Without an API key

Embeddings and BM25 run locally, so **hybrid retrieval and knowledge-graph
traversal work with no API key at all**. What a key buys is the generative half
of the pipeline:

| Stage | Needs a key |
|---|---|
| Chunking, embedding, hybrid retrieval (vector + BM25 + RRF) | no |
| Knowledge-graph traversal over an existing graph | no |
| Entity extraction at ingest (i.e. *building* the graph) | yes |
| Answer synthesis, citation verification, faithfulness check | yes |

Queries against a key-less server return `retrieval_only: true` with the ranked
passages, their source papers, and the relevant subgraph — the evidence, just
not the essay. Because extraction is what needs the key, a key-less server can
*serve* a graph but cannot *build* one.

## Public demo mode

`DEMO_MODE=1` serves a fixed, read-only library: the snapshot at
`$DEMO_SNAPSHOT` (default `demo/library.json`) is loaded on boot if the library
is empty, and `/upload`, `/upload-url`, `/note`, `DELETE /papers/{id}` and
`DELETE /reset` all return 403. That is what makes a public deployment safe to
leave running — no credentials on it, and no way for a visitor to fill its disk
or wipe it for everyone else.

Build the snapshot once, locally, with a key:

```bash
export GROQ_API_KEY=gsk_...
python main.py                          # one terminal
python scripts/build_demo_library.py    # another
```

That ingests a fixed set of papers through the ordinary `/upload-url` path — so
the graph in the snapshot is real extraction output, not authored data — then
writes `demo/library.json` via `GET /export?include_chunks=true`. Commit it and
deploy with `DEMO_MODE=1` and **no** API key set. The snapshot stores chunk
text rather than vectors, so embeddings are recomputed locally on load.

## Deployment (single container)

The included `Dockerfile` builds the frontend and serves it through the FastAPI
backend on a single port.

### Hugging Face Spaces (free, recommended)

1. Create a new Space → SDK: **Docker** (the README frontmatter already declares
   this so HF will pick it up automatically).
2. Push this repo to the Space.
3. In the Space's **Settings → Variables and secrets**, either add
   `GROQ_API_KEY` for the full pipeline, or set `DEMO_MODE=1` with no key for a
   read-only public demo (see [Public demo mode](#public-demo-mode)).
4. Wait for the build. The app appears at `https://huggingface.co/spaces/<you>/<name>`.

### Render / Fly.io / Railway

Any "deploy from Dockerfile" provider works. Just set `GROQ_API_KEY` in the
service's environment. The container listens on `$PORT` (default 7860).

### Local Docker test

```bash
docker build -t papertrail .
docker run -p 7860:7860 -e GROQ_API_KEY=$GROQ_API_KEY papertrail
# open http://localhost:7860
```

> **Note**: State persists across restarts — ChromaDB uses a persistent client
> and the knowledge graph + paper metadata are saved as JSON. Everything lives
> under `STATE_DIR` (`/data` on HF Spaces if it exists, else `./state`), so
> mount a volume there to keep your library across container recreations.

## Tests

```bash
pip install pytest
pytest tests/          # application test suite
pytest eval_recall/    # metric-math tests for the recall harness
```

CI (GitHub Actions) runs the Python test suite plus frontend lint and build on
every push and pull request. Note that CI runs `pytest tests/`, and
`pyproject.toml` sets `testpaths = ["tests"]` — so the `eval_recall/` tests are
**not** covered by CI and have to be run explicitly, as above.
