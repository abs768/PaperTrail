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
| GET | /graph | Get knowledge graph (nodes + edges) |
| GET | /stats | System statistics |
| DELETE | /papers/{paper_id} | Delete a single paper † |
| DELETE | /reset | Reset everything † |

† If the `ADMIN_TOKEN` env var is set, these require a matching `X-Admin-Token`
header — recommended on public deploys.

## Without an API Key

The system still works without an API key — it just skips entity extraction (no graph building) and uses only vector search for queries. For the demo, you really want the API key to show the full pipeline.

## Deployment (single container)

The included `Dockerfile` builds the frontend and serves it through the FastAPI
backend on a single port.

### Hugging Face Spaces (free, recommended)

1. Create a new Space → SDK: **Docker** (the README frontmatter already declares
   this so HF will pick it up automatically).
2. Push this repo to the Space.
3. In the Space's **Settings → Variables and secrets**, add `GROQ_API_KEY`.
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
pytest tests/
```

CI (GitHub Actions) runs the Python test suite plus frontend lint and build on
every push and pull request.
