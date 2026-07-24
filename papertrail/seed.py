"""Load a portable library snapshot into an empty library.

This is the counterpart to `GET /export?include_chunks=true`: that endpoint
writes the JSON, this module reads it back. Together they let a library built
on one machine be served from another — which is how the public demo works.
The snapshot carries chunk *text*, not vectors, so embeddings are recomputed
locally on load by ChromaDB's bundled model; no API key is involved.
"""
import json
import logging
import pathlib

import networkx as nx

from . import state

logger = logging.getLogger("papertrail")

# ChromaDB rejects oversized single adds; stay well under the limit.
_ADD_BATCH = 500


def _iter_batches(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def load_snapshot(source, replace: bool = False) -> dict:
    """Load a snapshot from a path, file object, or already-parsed dict.

    Returns a summary of what was loaded. With `replace=False` (the default)
    loading into a non-empty library is refused, so a seeded deployment never
    silently clobbers real data.
    """
    if isinstance(source, dict):
        payload = source
    else:
        path = pathlib.Path(source)
        if not path.exists():
            raise FileNotFoundError(f"No snapshot at {path}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

    if state.papers_db and not replace:
        raise RuntimeError(
            f"Library already holds {len(state.papers_db)} paper(s); "
            "pass replace=True to overwrite"
        )
    if replace:
        state.reset()

    papers = payload.get("papers") or {}
    graph = payload.get("graph") or {"directed": True, "nodes": [], "links": []}
    chunks = payload.get("chunks") or []

    state.papers_db.update(papers)
    # node_link_graph returns a new object, so rebind through the module rather
    # than mutating — other modules read state.kg by attribute for this reason.
    state.kg = nx.node_link_graph(graph)

    added = 0
    if chunks:
        ids = [c["id"] for c in chunks]
        docs = [c["text"] for c in chunks]
        metas = [c.get("metadata") or {} for c in chunks]
        for id_b, doc_b, meta_b in zip(
            _iter_batches(ids, _ADD_BATCH),
            _iter_batches(docs, _ADD_BATCH),
            _iter_batches(metas, _ADD_BATCH),
        ):
            state.collection.add(ids=id_b, documents=doc_b, metadatas=meta_b)
            added += len(id_b)

    # The BM25 index is derived from the collection, so it has to be rebuilt.
    from .retrieval import mark_bm25_dirty
    mark_bm25_dirty()

    state.save_state()

    summary = {
        "papers": len(papers),
        "graph_nodes": state.kg.number_of_nodes(),
        "graph_edges": state.kg.number_of_edges(),
        "chunks": added,
        "exported_at": payload.get("exported_at"),
    }
    logger.info(
        "Seeded library from snapshot: %(papers)d papers, %(graph_nodes)d nodes, "
        "%(graph_edges)d edges, %(chunks)d chunks", summary
    )
    return summary


def seed_if_empty(path) -> dict | None:
    """Load `path` only when the library is empty. Returns None if skipped.

    Safe to call on every boot: a redeploy with persistent storage keeps
    whatever is already there, and a fresh container gets the demo library.
    """
    path = pathlib.Path(path)
    if not path.exists():
        logger.info("No demo snapshot at %s — starting with an empty library", path)
        return None
    if state.papers_db:
        logger.info("Library already has %d paper(s) — not seeding", len(state.papers_db))
        return None
    try:
        return load_snapshot(path)
    except Exception as e:
        logger.error("Failed to seed from %s: %s", path, e)
        return None
