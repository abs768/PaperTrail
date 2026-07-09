"""Shared mutable state: the knowledge graph, paper metadata, and the ChromaDB
collection — plus JSON persistence.

Other modules must access these via attribute (`state.kg`, `state.papers_db`,
`state.collection`) rather than `from ... import kg`, because reset() rebinds
the module-level names.
"""
import json
import logging

import chromadb
import networkx as nx
from chromadb.utils import embedding_functions

from .config import CHROMA_DIR, LEGACY_STATE_FILE, STATE_FILE

logger = logging.getLogger("papertrail")

# Knowledge Graph (in-memory, persisted as JSON)
kg = nx.DiGraph()
papers_db: dict = {}

# ChromaDB (persistent)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="papertrail_chunks",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},
)


def save_state():
    """Persist knowledge graph + paper metadata to disk as JSON."""
    try:
        payload = {"kg": nx.node_link_data(kg), "papers_db": papers_db}
        tmp = STATE_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        tmp.replace(STATE_FILE)  # atomic swap so a crash mid-write can't corrupt state
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def _load_legacy_pickle() -> bool:
    """One-time migration from the old pickle format. Returns True if loaded."""
    global kg, papers_db
    if not LEGACY_STATE_FILE.exists():
        return False
    try:
        import pickle
        with open(LEGACY_STATE_FILE, "rb") as f:
            data = pickle.load(f)
        kg = data.get("kg", nx.DiGraph())
        papers_db = data.get("papers_db", {})
        save_state()  # rewrite as JSON
        LEGACY_STATE_FILE.rename(LEGACY_STATE_FILE.with_suffix(".pkl.migrated"))
        logger.info("Migrated legacy pickle state to JSON")
        return True
    except Exception as e:
        logger.error(f"Failed to migrate legacy pickle state: {e}")
        return False


def load_state():
    """Load knowledge graph + paper metadata from disk if present."""
    global kg, papers_db
    if not STATE_FILE.exists():
        if _load_legacy_pickle():
            return
        if not STATE_FILE.exists():
            return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        kg = nx.node_link_graph(data.get("kg") or {"directed": True, "nodes": [], "links": []})
        papers_db = data.get("papers_db", {})
        logger.info(f"Restored state: {len(papers_db)} papers, {len(kg.nodes)} graph nodes, {collection.count()} chunks")
    except Exception as e:
        logger.error(f"Failed to load state: {e}")


def reset():
    """Wipe the graph, paper metadata, and the vector collection."""
    global kg, papers_db, collection
    kg = nx.DiGraph()
    papers_db = {}
    try:
        chroma_client.delete_collection("papertrail_chunks")
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(
        name="papertrail_chunks",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    save_state()


load_state()
