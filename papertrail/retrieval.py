"""Hybrid retrieval: ChromaDB vector search + BM25, fused with Reciprocal Rank
Fusion, with an optional cross-encoder rerank stage."""
import json
import logging
import os
import re
from typing import Optional

from . import state

logger = logging.getLogger("papertrail")

# ── Hybrid retrieval state (BM25 alongside vector store) ──────────────────────
_bm25_index = None  # rank_bm25.BM25Okapi
_bm25_chunk_ids: list[str] = []  # parallel arrays — same index across all three
_bm25_paper_ids: list[str] = []
_bm25_dirty = True


def _bm25_tokenize(s: str) -> list[str]:
    """Lightweight tokenizer: lowercase, alnum runs only."""
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _rebuild_bm25_index() -> None:
    """Rebuild the in-memory BM25 index from the current ChromaDB collection.
    Cheap for small libraries; we just rebuild on every add/delete."""
    global _bm25_index, _bm25_chunk_ids, _bm25_paper_ids, _bm25_dirty
    try:
        if state.collection.count() == 0:
            _bm25_index = None
            _bm25_chunk_ids = []
            _bm25_paper_ids = []
            _bm25_dirty = False
            return
        from rank_bm25 import BM25Okapi
        data = state.collection.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or [{} for _ in docs]
        tokenized = [_bm25_tokenize(d) for d in docs]
        if not tokenized or not any(tokenized):
            _bm25_index = None
            _bm25_chunk_ids = []
            _bm25_paper_ids = []
        else:
            _bm25_index = BM25Okapi(tokenized)
            _bm25_chunk_ids = list(ids)
            _bm25_paper_ids = [(m or {}).get("paper_id", "") for m in metas]
        _bm25_dirty = False
        logger.info(f"BM25 index rebuilt over {len(_bm25_chunk_ids)} chunks")
    except Exception as e:
        logger.warning(f"BM25 rebuild failed (will fall back to vector-only): {e}")
        _bm25_index = None
        _bm25_chunk_ids = []
        _bm25_paper_ids = []
        _bm25_dirty = False


def mark_bm25_dirty() -> None:
    """Flag the BM25 index for lazy rebuild on next search."""
    global _bm25_dirty
    _bm25_dirty = True


def reset_bm25_index() -> None:
    """Drop the BM25 index entirely (used by /reset)."""
    global _bm25_index, _bm25_chunk_ids, _bm25_paper_ids, _bm25_dirty
    _bm25_index = None
    _bm25_chunk_ids = []
    _bm25_paper_ids = []
    _bm25_dirty = False


def _ensure_bm25() -> None:
    if _bm25_dirty:
        _rebuild_bm25_index()


def _bm25_search(query: str, top_k: int, paper_ids: list = None) -> list[tuple[str, float]]:
    """Return [(chunk_id, score), …] ranked by BM25. Empty if no index."""
    _ensure_bm25()
    if _bm25_index is None or not _bm25_chunk_ids:
        return []
    tokens = _bm25_tokenize(query)
    if not tokens:
        return []
    scores = _bm25_index.get_scores(tokens)
    pid_set = set(paper_ids) if paper_ids else None
    indexed = [
        (i, s) for i, s in enumerate(scores)
        if s > 0 and (pid_set is None or _bm25_paper_ids[i] in pid_set)
    ]
    indexed.sort(key=lambda t: t[1], reverse=True)
    return [(_bm25_chunk_ids[i], s) for i, s in indexed[:top_k]]


# ── Cross-encoder reranker (opt-in) ───────────────────────────────────────────
# Re-scores (query, passage) pairs after the hybrid fusion. Heavy to load
# (~80MB + torch warm-up), so gated behind ENABLE_RERANKER=1.
_reranker = None
_reranker_load_attempted = False


def _reranker_enabled() -> bool:
    return os.getenv("ENABLE_RERANKER", "").lower() in ("1", "true", "yes", "on")


def _get_reranker():
    global _reranker, _reranker_load_attempted
    if _reranker_load_attempted:
        return _reranker
    _reranker_load_attempted = True
    if not _reranker_enabled():
        return None
    try:
        from sentence_transformers import CrossEncoder
        model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info(f"Loading cross-encoder reranker: {model_name}")
        _reranker = CrossEncoder(model_name)
        logger.info("Reranker loaded.")
    except Exception as e:
        logger.warning(f"Reranker disabled — failed to load: {e}")
        _reranker = None
    return _reranker


def _rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Re-score (query, chunk.text) pairs with the cross-encoder, return top_k.
    Falls through to fused order if the reranker is disabled or errors."""
    rer = _get_reranker()
    if rer is None or not chunks:
        return chunks[:top_k]
    try:
        pairs = [(query, c.get("text", "") or "") for c in chunks]
        scores = rer.predict(pairs)
    except Exception as e:
        logger.warning(f"Rerank failed, falling back to fused order: {e}")
        return chunks[:top_k]
    scored = sorted(
        ((float(s), c) for s, c in zip(scores, chunks)),
        key=lambda t: t[0],
        reverse=True,
    )
    return [{**c, "rerank_score": round(s, 4)} for s, c in scored[:top_k]]


# ── Vector + hybrid search ─────────────────────────────────────────────────────


def _vector_search_raw(query: str, top_k: int, paper_ids: list = None) -> list[dict]:
    """Pure vector search via ChromaDB. Returns chunk dicts with full text."""
    collection = state.collection
    if collection.count() == 0:
        return []

    where = None
    if paper_ids:
        where = {"paper_id": {"$in": list(paper_ids)}} if len(paper_ids) > 1 else {"paper_id": paper_ids[0]}

    kwargs = {"query_texts": [query], "n_results": min(top_k, collection.count())}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    if not results.get("documents") or not results["documents"][0]:
        return []

    matches = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i] or {}
        chunk_id = (results.get("ids") or [[]])[0][i] if results.get("ids") else None
        matches.append(
            {
                "chunk_id": chunk_id,
                "paper_id": meta.get("paper_id"),
                "paper_title": meta.get("title", "Unknown"),
                "page": meta.get("page"),
                "text": results["documents"][0][i],
                "distance": round(results["distances"][0][i], 4)
                if results.get("distances")
                else None,
            }
        )
    return matches


def _hydrate_chunks(chunk_ids: list[str]) -> dict[str, dict]:
    """Fetch chunk dicts (text + metadata) for a list of ids. Useful for BM25-only hits."""
    if not chunk_ids:
        return {}
    try:
        got = state.collection.get(ids=list(chunk_ids), include=["documents", "metadatas"])
    except Exception as e:
        logger.warning(f"Hydration fetch failed: {e}")
        return {}
    out: dict[str, dict] = {}
    for i, cid in enumerate(got.get("ids") or []):
        meta = (got.get("metadatas") or [{}])[i] or {}
        text = (got.get("documents") or [""])[i] or ""
        out[cid] = {
            "chunk_id": cid,
            "paper_id": meta.get("paper_id"),
            "paper_title": meta.get("title", "Unknown"),
            "page": meta.get("page"),
            "text": text,
            "distance": None,
        }
    return out


def _search_chunks(
    query: str,
    top_k: int = 5,
    paper_ids: list = None,
    truncate_to: Optional[int] = None,
) -> list[dict]:
    """Hybrid retrieval: vector + BM25 fused via Reciprocal Rank Fusion.

    Each result dict has: chunk_id, paper_id, paper_title, page, text, distance.
    `truncate_to=None` returns full text (used for grounding); a positive int
    truncates (used for the LLM-facing tool wrapper).
    """
    if state.collection.count() == 0:
        return []

    # Overshoot top_k on each lane so RRF has more material to fuse over.
    # Wider pool when reranker is on — it benefits from more candidates.
    pool_mult = 6 if _reranker_enabled() else 3
    pool_n = max(top_k * pool_mult, 20)

    vector_hits = _vector_search_raw(query, pool_n, paper_ids)
    bm25_hits = _bm25_search(query, pool_n, paper_ids)

    # Reciprocal Rank Fusion. K=60 is the standard constant from the original
    # RRF paper (Cormack et al.); damps top-rank dominance.
    K_RRF = 60
    rrf: dict[str, float] = {}
    for rank, c in enumerate(vector_hits):
        cid = c.get("chunk_id")
        if cid:
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (K_RRF + rank + 1)
    for rank, (cid, _score) in enumerate(bm25_hits):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (K_RRF + rank + 1)

    # Hydrate any chunk ids that came only from BM25 (no vector hit).
    by_id = {c["chunk_id"]: c for c in vector_hits if c.get("chunk_id")}
    missing = [cid for cid in rrf.keys() if cid not in by_id]
    if missing:
        by_id.update(_hydrate_chunks(missing))

    # Sort by fused score, drop anything we couldn't hydrate.
    ordered = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)
    fused: list[dict] = []
    for cid in ordered:
        chunk = by_id.get(cid)
        if chunk is None:
            continue
        # Stamp the fusion score so callers can inspect it.
        fused.append({**chunk, "rrf_score": round(rrf[cid], 5)})

    # Cross-encoder rerank (opt-in via ENABLE_RERANKER) — pulls top_k from the pool.
    final = _rerank_chunks(query, fused, top_k) if _reranker_enabled() else fused[:top_k]

    if truncate_to:
        final = [
            {**c, "text": (c["text"][:truncate_to] if c.get("text") and len(c["text"]) > truncate_to else c.get("text"))}
            for c in final
        ]
    return final


def search_vector_store(query: str, top_k: int = 5, paper_ids: list = None) -> str:
    """JSON wrapper used by the function-calling tool surface. Truncates text
    to 600 chars to keep the tool-result payload small. For grounding, call
    `_search_chunks` directly to get full text + chunk_id."""
    matches = _search_chunks(query, top_k=top_k, paper_ids=paper_ids, truncate_to=600)
    if not matches:
        return json.dumps({"results": [], "message": "No papers indexed yet."})
    return json.dumps({"results": matches})


def add_to_vector_store(paper_id: str, chunks, metadata: dict):
    """Accepts either list[str] (legacy) or list[{text, page}] (with provenance)."""
    if not chunks:
        return
    docs, metas, ids = [], [], []
    for i, c in enumerate(chunks):
        if isinstance(c, dict):
            text = c["text"]
            page = c.get("page")
        else:
            text = c
            page = None
        ids.append(f"{paper_id}_chunk_{i}")
        docs.append(text)
        meta = {"paper_id": paper_id, "title": metadata.get("title", "Unknown"), "chunk_index": i}
        if page is not None:
            meta["page"] = page
        metas.append(meta)
    state.collection.add(documents=docs, ids=ids, metadatas=metas)
    mark_bm25_dirty()
    logger.info(f"Added {len(docs)} chunks to vector store")


_rebuild_bm25_index()
