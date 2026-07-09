"""PaperTrail entrypoint. The application lives in the `papertrail` package:

    papertrail/config.py      — env config, storage paths, LLM client, structured-output helpers
    papertrail/models.py      — Pydantic models (LLM outputs + API request bodies)
    papertrail/state.py       — knowledge graph + papers_db + ChromaDB collection, JSON persistence
    papertrail/textproc.py    — PDF text extraction, title recovery, chunking
    papertrail/kgraph.py      — entity canonicalization, graph building/traversal
    papertrail/extraction.py  — LLM entity extraction + validation
    papertrail/retrieval.py   — hybrid vector+BM25 retrieval (RRF), optional reranker
    papertrail/query.py       — GraphRAG query pipeline with grounded citations
    papertrail/api.py         — FastAPI app and endpoints
"""
import os

from papertrail.api import app  # noqa: F401  (uvicorn target: main:app)

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
