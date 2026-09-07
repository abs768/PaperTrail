"""FastAPI application: ingestion, query, graph, and admin endpoints."""
import asyncio
import hashlib
import json
import logging
import os
import pathlib
import re
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import state
from .config import DEMO_MODE, DEMO_SNAPSHOT, UPLOAD_DIR, client as _llm_client
from .extraction import extract_entities
from .kgraph import add_to_knowledge_graph
from .models import NoteRequest, QueryRequest, UrlUploadRequest
from .query import graphrag_query
from .retrieval import add_to_vector_store, mark_bm25_dirty, reset_bm25_index
from .textproc import (
    chunk_pages,
    chunk_text,
    extract_first_page_title_heuristic,
    extract_pdf_metadata_title,
    extract_text_from_pdf,
    _llm_title_is_grounded,
    _looks_like_real_title,
)

logger = logging.getLogger("papertrail")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """In demo mode, load the bundled snapshot into an empty library on boot.

    Chunk embeddings are recomputed locally on load, so this needs no API key
    — which is the point: the public demo carries no credentials.
    """
    if DEMO_MODE:
        from .seed import seed_if_empty
        summary = seed_if_empty(DEMO_SNAPSHOT)
        if summary:
            logger.info("Demo library ready: %s", summary)
    yield


app = FastAPI(title="PaperTrail API", version="2.0.0", lifespan=_lifespan)

# Wildcard origins with credentials is an invalid CORS combination (and the app
# uses no cookies), so credentials stay off. Origins can be pinned via env.
_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional shared secret for destructive endpoints (DELETE /reset, DELETE /papers/…).
# Unset (default) = open, which is fine for local use; set it on public deploys.
_ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# Cap on uploaded / fetched PDF size. Research PDFs are single-digit MB;
# the cap mainly protects the server from accidental or hostile huge files.
_MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024


def _check_admin(supplied: str):
    if _ADMIN_TOKEN and supplied != _ADMIN_TOKEN:
        raise HTTPException(403, "Admin token required (X-Admin-Token header)")


def _check_writable():
    """Reject mutations in demo mode.

    The public demo serves a fixed library: leaving ingestion open would let
    anyone grow its storage without bound, and deletion open would let one
    visitor empty it for everyone.
    """
    if DEMO_MODE:
        raise HTTPException(
            403,
            "This is a read-only public demo. Run PaperTrail locally to add "
            "your own papers — see the repository README.",
        )


# Serializes mutations of the shared state (kg / papers_db / vector collection).
# Endpoints run in the threadpool, so two concurrent uploads or a delete during
# an upload would otherwise interleave. The slow part of ingestion (LLM entity
# extraction) happens OUTSIDE the lock — only the state writes are serialized.
_mutate_lock = threading.Lock()


@app.get("/api/health")
def health():
    """Health plus the two capability flags the UI needs to describe itself:
    whether the library is writable, and whether LLM synthesis is available."""
    return {
        "status": "ok",
        "service": "PaperTrail API",
        "version": "2.0.0",
        "demo_mode": DEMO_MODE,
        "read_only": DEMO_MODE,
        "llm_enabled": _llm_client is not None,
        "papers": len(state.papers_db),
    }


# ══════════════════════════════════════════════════════════════════════════════
# INGESTION
# ══════════════════════════════════════════════════════════════════════════════


def _ingest_pdf_bytes(file_bytes: bytes, source_label: str, metadata_override: dict = None) -> dict:
    """Index a PDF from raw bytes. `source_label` is the filename or URL.
    `metadata_override` supplies authoritative title/authors (e.g. from the
    arXiv API) that beat both LLM extraction and PDF heuristics."""
    file_hash = hashlib.md5(file_bytes).hexdigest()[:12]
    paper_id = f"paper:{file_hash}"

    if paper_id in state.papers_db:
        return {"message": "Paper already indexed", "paper_id": paper_id, "title": state.papers_db[paper_id]["title"], "duplicate": True}

    file_path = os.path.join(UPLOAD_DIR, f"{file_hash}.pdf")
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    pages = extract_text_from_pdf(file_path)
    if not pages:
        raise HTTPException(400, "Could not extract text from PDF")

    full_text = "\n\n".join(p["text"] for p in pages)
    try:
        entities = extract_entities(full_text)
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ("429", "rate limit", "quota", "resource_exhausted")):
            raise HTTPException(429, "AI API rate limited — try again in a moment.")
        raise HTTPException(500, f"Entity extraction failed: {e}")

    # Title resolution priority:
    #   LLM-extracted (if grounded) > PDF metadata > first-page font-size heuristic > URL filename.
    # The grounding check guards against the 8B model paraphrasing the title
    # (e.g. emitting 'Transfer Learning for NLP via BERT' instead of the actual
    # 'BERT: Pre-training of Deep Bidirectional Transformers...'). If the
    # LLM-emitted title doesn't appear verbatim near the start of the source,
    # we treat it as a hallucination and fall through to the deterministic
    # extractors below.
    pdf_meta_title = extract_pdf_metadata_title(file_path)
    heuristic_title = "" if pdf_meta_title else extract_first_page_title_heuristic(file_path)
    fallback_title = (
        pdf_meta_title
        or heuristic_title
        or source_label.rsplit("/", 1)[-1].replace(".pdf", "")
    )
    llm_title = (entities.get("title") or "").strip()
    title = llm_title if _llm_title_is_grounded(llm_title, full_text) else fallback_title
    # Authoritative metadata (arXiv API) wins over everything.
    if metadata_override:
        ov_title = (metadata_override.get("title") or "").strip()
        if _looks_like_real_title(ov_title):
            title = ov_title
        if metadata_override.get("authors"):
            entities["authors"] = list(metadata_override["authors"])
    entities["title"] = title

    page_chunks = chunk_pages(pages)

    with _mutate_lock:
        # Re-check: another thread may have indexed the same file while we were extracting.
        if paper_id in state.papers_db:
            return {"message": "Paper already indexed", "paper_id": paper_id, "title": state.papers_db[paper_id]["title"], "duplicate": True}
        add_to_knowledge_graph(paper_id, entities)
        add_to_vector_store(paper_id, page_chunks, {"title": title})
        state.papers_db[paper_id] = {
            "title": title,
            "filename": source_label,
            "pages": len(pages),
            "chunks": len(page_chunks),
            "entities": entities,
            "uploaded_at": datetime.now().isoformat(),
        }
        state.save_state()

    return {
        "message": "Paper indexed successfully",
        "paper_id": paper_id,
        "title": title,
        "pages": len(pages),
        "chunks": len(page_chunks),
        "entities_found": {
            "authors": len(entities.get("authors", [])),
            "methods": len(entities.get("methods", [])),
            "datasets": len(entities.get("datasets", [])),
            "metrics": len(entities.get("metrics", [])),
            "concepts": len(entities.get("key_concepts", [])),
            "relationships": len(entities.get("relationships", [])),
        },
        "entities_sample": {
            "authors": entities.get("authors", [])[:5],
            "methods": entities.get("methods", [])[:5],
            "datasets": entities.get("datasets", [])[:4],
            "key_concepts": entities.get("key_concepts", [])[:6],
        },
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    _check_writable()
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    file_bytes = await file.read()
    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"PDF too large (max {_MAX_UPLOAD_BYTES // (1024*1024)} MB)")
    # Ingestion does blocking I/O + LLM calls (15-30s) — keep it off the event loop.
    return await run_in_threadpool(_ingest_pdf_bytes, file_bytes, file.filename)


def _normalize_paper_url(url: str) -> str:
    """Turn arXiv abs URLs into PDF URLs; pass through everything else."""
    url = url.strip()
    m = re.match(r"https?://(?:www\.)?arxiv\.org/abs/([^?#\s]+)", url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
    if "arxiv.org/pdf/" in url and not url.lower().endswith(".pdf"):
        return url + ".pdf"
    return url


def _assert_public_http_url(url: str):
    """SSRF guard: only http(s), and the host must not resolve to a private,
    loopback, or link-local address — otherwise a hosted deploy could be used
    to probe its internal network."""
    import socket
    import ipaddress
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, "Only http(s) URLs are supported")
    host = parsed.hostname
    if not host:
        raise HTTPException(400, "URL has no host")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise HTTPException(400, f"Could not resolve host: {host}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise HTTPException(400, "URL resolves to a private or local address")


_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#\s]+?)(?:\.pdf)?$", re.IGNORECASE)
_ARXIV_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


def _parse_arxiv_atom(xml_text: str) -> dict:
    """Parse an arXiv Atom API response into {'title', 'authors'}. Empty dict
    if there's no usable entry (unknown ids come back as stub entries)."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}
    entry = root.find("a:entry", _ARXIV_ATOM_NS)
    if entry is None:
        return {}
    title = re.sub(r"\s+", " ", entry.findtext("a:title", "", _ARXIV_ATOM_NS) or "").strip()
    if not title or title.lower() == "error":
        return {}
    authors = [
        re.sub(r"\s+", " ", (a.findtext("a:name", "", _ARXIV_ATOM_NS) or "")).strip()
        for a in entry.findall("a:author", _ARXIV_ATOM_NS)
    ]
    return {"title": title, "authors": [a for a in authors if a]}


async def _fetch_arxiv_metadata(url: str) -> dict:
    """For arXiv URLs, pull the exact title/authors from the arXiv export API —
    far more reliable than extracting them from the PDF. Soft-fails to {}."""
    m = _ARXIV_ID_RE.search(url)
    if not m:
        return {}
    import httpx
    try:
        async with httpx.AsyncClient(timeout=15.0) as h:
            resp = await h.get("https://export.arxiv.org/api/query", params={"id_list": m.group(1)})
        if resp.status_code != 200:
            return {}
        meta = _parse_arxiv_atom(resp.text)
        if meta:
            logger.info(f"arXiv metadata: {meta['title']!r}, {len(meta['authors'])} authors")
        return meta
    except Exception as e:
        logger.warning(f"arXiv metadata fetch failed: {e}")
        return {}


@app.post("/upload-url")
async def upload_pdf_from_url(req: UrlUploadRequest):
    _check_writable()
    import httpx
    url = _normalize_paper_url(req.url)
    _assert_public_http_url(url)
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as h:
            resp = await h.get(url, headers={"User-Agent": "PaperTrail/1.0 (research memory agent)"})
            if resp.status_code != 200:
                raise HTTPException(400, f"Could not fetch URL ({resp.status_code})")
            # Redirects may land somewhere else — re-check the final host too.
            _assert_public_http_url(str(resp.url))
            ct = resp.headers.get("content-type", "")
            if "pdf" not in ct.lower() and not url.lower().endswith(".pdf"):
                raise HTTPException(400, f"URL did not return a PDF (content-type: {ct})")
            file_bytes = resp.content
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Fetch failed: {e}")
    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"PDF too large (max {_MAX_UPLOAD_BYTES // (1024*1024)} MB)")
    arxiv_meta = await _fetch_arxiv_metadata(url)
    return await run_in_threadpool(_ingest_pdf_bytes, file_bytes, url, arxiv_meta)


def _ingest_note(title: str, content: str) -> dict:
    note_id = f"note:{uuid.uuid4().hex[:12]}"
    entities = extract_entities(content)
    chunks = chunk_text(content)

    with _mutate_lock:
        add_to_knowledge_graph(note_id, {**entities, "title": title})
        add_to_vector_store(note_id, chunks, {"title": title})
        state.papers_db[note_id] = {
            "title": title,
            "type": "note",
            "chunks": len(chunks),
            "entities": entities,
            "uploaded_at": datetime.now().isoformat(),
        }
        state.save_state()
    return {"message": "Note added", "note_id": note_id, "title": title}


@app.post("/note")
async def add_note(note: NoteRequest):
    _check_writable()
    return await run_in_threadpool(_ingest_note, note.title, note.content)


# ══════════════════════════════════════════════════════════════════════════════
# QUERY
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/query")
async def query(req: QueryRequest):
    if state.collection.count() == 0 and len(state.kg.nodes) == 0:
        return {"answer": "Your library is empty. Upload some papers first!", "sources": []}
    return await run_in_threadpool(graphrag_query, req.question, req.top_k)


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Same pipeline as /query, but delivered as Server-Sent Events:
    `progress` events as the pipeline advances (classifying → retrieving →
    generating → verifying → fact_checking), then one `result` event with the
    full answer payload."""
    if state.collection.count() == 0 and len(state.kg.nodes) == 0:
        async def empty_gen():
            payload = {"event": "result", "data": {"answer": "Your library is empty. Upload some papers first!", "sources": []}}
            yield f"data: {json.dumps(payload)}\n\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream")

    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()

    def progress(stage: str, detail: dict):
        # Called from the worker thread — hop back onto the event loop.
        loop.call_soon_threadsafe(events.put_nowait, {"event": "progress", "stage": stage, "detail": detail})

    async def run_query():
        try:
            result = await run_in_threadpool(graphrag_query, req.question, req.top_k, progress)
            events.put_nowait({"event": "result", "data": result})
        except Exception as e:
            logger.error(f"Streamed query failed: {e}")
            events.put_nowait({"event": "error", "message": str(e)})
        finally:
            events.put_nowait(None)  # sentinel: stream done

    async def gen():
        task = asyncio.create_task(run_query())
        try:
            while True:
                item = await events.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            await task

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# LIBRARY / GRAPH / ADMIN
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/papers")
def list_papers():
    return {
        "papers": [
            {
                "id": pid,
                "title": pdata["title"],
                "type": pdata.get("type", "paper"),
                "pages": pdata.get("pages"),
                "chunks": pdata.get("chunks", 0),
                "uploaded_at": pdata.get("uploaded_at"),
            }
            for pid, pdata in state.papers_db.items()
        ],
        "total": len(state.papers_db),
    }


@app.get("/papers/{paper_id:path}")
def get_paper(paper_id: str):
    paper_id = paper_id.replace("%3A", ":")
    if paper_id not in state.papers_db:
        raise HTTPException(404, "Paper not found")
    return state.papers_db[paper_id]


@app.get("/graph")
def get_graph(limit: int = 0):
    """Return the knowledge graph. With `?limit=N` (N > 0), large graphs are
    trimmed for rendering: paper/note nodes are always kept, then the
    highest-degree entities fill the remaining budget. Edges between dropped
    nodes are omitted."""
    kg = state.kg
    keep = set(kg.nodes)
    truncated = False
    if 0 < limit < len(keep):
        anchors = [n for n, d in kg.nodes(data=True) if d.get("type") in ("paper", "note")]
        rest = sorted(
            (n for n in kg.nodes if n not in set(anchors)),
            key=kg.degree, reverse=True,
        )
        keep = set(anchors) | set(rest[: max(limit - len(anchors), 0)])
        truncated = len(keep) < len(kg.nodes)
    nodes = [
        {"id": nid, "label": data.get("label", data.get("name", nid)), "type": data.get("type", "unknown")}
        for nid, data in kg.nodes(data=True) if nid in keep
    ]
    edges = [
        {"source": src, "target": tgt, "relation": data.get("relation", "related")}
        for src, tgt, data in kg.edges(data=True) if src in keep and tgt in keep
    ]
    return {
        "nodes": nodes, "edges": edges,
        "node_count": len(nodes), "edge_count": len(edges),
        "total_nodes": len(kg.nodes), "truncated": truncated,
    }


@app.get("/export")
def export_library(include_chunks: bool = False):
    """Export the whole library as portable JSON: paper metadata and the
    knowledge graph (node-link format). `?include_chunks=true` adds every
    indexed text chunk with its metadata — a complete backup that pairs with
    the JSON state format."""
    import networkx as nx
    payload = {
        "exported_at": datetime.now().isoformat(),
        "paper_count": len(state.papers_db),
        "papers": state.papers_db,
        "graph": nx.node_link_data(state.kg),
    }
    if include_chunks:
        got = state.collection.get(include=["documents", "metadatas"])
        payload["chunks"] = [
            {
                "id": cid,
                "text": (got.get("documents") or [])[i],
                "metadata": (got.get("metadatas") or [])[i],
            }
            for i, cid in enumerate(got.get("ids") or [])
        ]
    return payload


@app.get("/stats")
def get_stats():
    node_types = {}
    for _, data in state.kg.nodes(data=True):
        t = data.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    return {
        "papers": len(state.papers_db),
        "graph_nodes": len(state.kg.nodes),
        "graph_edges": len(state.kg.edges),
        "vector_chunks": state.collection.count(),
        "node_types": node_types,
    }


@app.delete("/papers/{paper_id:path}")
def delete_paper(paper_id: str, x_admin_token: str = Header(default="")):
    _check_writable()
    _check_admin(x_admin_token)
    paper_id = paper_id.replace("%3A", ":")
    with _mutate_lock:
        if paper_id not in state.papers_db:
            raise HTTPException(404, "Paper not found")
        # Drop chunks
        try:
            state.collection.delete(where={"paper_id": paper_id})
            mark_bm25_dirty()
        except Exception as e:
            logger.warning(f"Chroma delete failed for {paper_id}: {e}")
        # Drop graph nodes that are exclusive to this paper: the paper node, plus
        # any entity that this paper was the last one referencing. Only the
        # deleted paper's former neighbours can be newly orphaned, and a node
        # still listed in papers_db is a library item, not an orphan — a paper
        # whose extraction yielded no entities has degree 0 from the start and
        # must survive an unrelated delete.
        if state.kg.has_node(paper_id):
            neighbours = set(state.kg.predecessors(paper_id)) | set(state.kg.successors(paper_id))
            state.kg.remove_node(paper_id)
            for n in neighbours:
                if n not in state.papers_db and state.kg.degree(n) == 0:
                    state.kg.remove_node(n)
        state.papers_db.pop(paper_id, None)
        # Remove the stored PDF (paper ids are "paper:{file_hash}"; notes have no file).
        if paper_id.startswith("paper:"):
            pdf_path = os.path.join(UPLOAD_DIR, f"{paper_id.split(':', 1)[1]}.pdf")
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
            except OSError as e:
                logger.warning(f"Could not remove stored PDF {pdf_path}: {e}")
        state.save_state()
    return {"message": "Paper deleted", "paper_id": paper_id}


@app.delete("/reset")
def reset_system(x_admin_token: str = Header(default="")):
    _check_writable()
    _check_admin(x_admin_token)
    with _mutate_lock:
        state.reset()
        reset_bm25_index()
        for name in os.listdir(UPLOAD_DIR):
            try:
                os.remove(os.path.join(UPLOAD_DIR, name))
            except OSError as e:
                logger.warning(f"Could not remove {name} during reset: {e}")
    return {"message": "System reset complete"}


# ── Serve built frontend (single-container deploy) ────────────────────────────
_FRONTEND_DIST = (pathlib.Path(__file__).parent.parent / "frontend" / "dist").resolve()


def _safe_dist_file(full_path: str):
    """Resolve `full_path` under the dist root, or None if it would escape.

    `full_path` is attacker-controlled and may contain traversal sequences
    (including percent-encoded ones like %2e%2e%2f that survive URL
    normalization). Resolving the joined path and confirming it stays within
    _FRONTEND_DIST is what stops it from serving .env, source, or /etc/passwd.

    Defined unconditionally, outside the `is_dir()` guard below, so the escape
    check is importable and testable on a checkout with no built frontend —
    otherwise the regression tests for this silently vanish wherever CI does not
    run a frontend build. With no dist present every path simply resolves to a
    non-file and returns None, which is the safe answer anyway.
    """
    candidate = (_FRONTEND_DIST / full_path).resolve()
    if candidate != _FRONTEND_DIST and _FRONTEND_DIST not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


# The routes, unlike the helper, only make sense with a build present: mounting
# a catch-all when there is no dist would turn every unknown API path into a
# broken file response instead of an honest 404.
if _FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa_fallback(full_path: str):
        candidate = _safe_dist_file(full_path)
        if candidate is not None:
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")
