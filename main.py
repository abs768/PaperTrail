"""
PaperTrail: The Research Memory Agent
Backend — FastAPI + NetworkX + ChromaDB
Uses OpenAI Structured Outputs + Function Calling patterns
"""
import time
import json
import hashlib
import os
import uuid
import logging
from datetime import datetime
from typing import Optional

import fitz  # PyMuPDF
import networkx as nx
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
import pickle
import pathlib
import re

# Persistent storage path. HF Spaces persistent storage is /data; fall back to ./state locally.
_STATE_CANDIDATES = [pathlib.Path(p) for p in [os.getenv("STATE_DIR", ""), "/data", "./state"] if p]
STATE_DIR = next((p for p in _STATE_CANDIDATES if p.parent.exists() and os.access(p.parent, os.W_OK)), pathlib.Path("./state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.pkl"
CHROMA_DIR = STATE_DIR / "chroma"
UPLOAD_DIR = str(STATE_DIR / "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"State directory: {STATE_DIR}")

# ── Initialize ────────────────────────────────────────────────────────────────
app = FastAPI(title="PaperTrail API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Knowledge Graph (in-memory, persisted via pickle)
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
    """Persist knowledge graph + paper metadata to disk."""
    try:
        with open(STATE_FILE, "wb") as f:
            pickle.dump({"kg": kg, "papers_db": papers_db}, f)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def load_state():
    """Load knowledge graph + paper metadata from disk if present."""
    global kg, papers_db
    if not STATE_FILE.exists():
        return
    try:
        with open(STATE_FILE, "rb") as f:
            data = pickle.load(f)
            kg = data.get("kg", nx.DiGraph())
            papers_db = data.get("papers_db", {})
        logger.info(f"Restored state: {len(papers_db)} papers, {len(kg.nodes)} graph nodes, {collection.count()} chunks")
    except Exception as e:
        logger.error(f"Failed to load state: {e}")


load_state()


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
        if collection.count() == 0:
            _bm25_index = None
            _bm25_chunk_ids = []
            _bm25_paper_ids = []
            _bm25_dirty = False
            return
        from rank_bm25 import BM25Okapi
        data = collection.get(include=["documents", "metadatas"])
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


_rebuild_bm25_index()


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


# LLM client — Groq (OpenAI-compatible API), with optional Gemini fallback
_GROQ_KEY = os.getenv("GROQ_API_KEY")
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if _GROQ_KEY:
    client = OpenAI(api_key=_GROQ_KEY, base_url="https://api.groq.com/openai/v1")
    _DEFAULT_MODEL = "llama-3.3-70b-versatile"
elif _GEMINI_KEY:
    client = OpenAI(api_key=_GEMINI_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    _DEFAULT_MODEL = "gemini-2.5-flash"
else:
    client = None
    _DEFAULT_MODEL = ""

# Two model lanes: a fast cheap one for classification/extraction, a strong one for synthesis.
MODEL_FAST    = os.getenv("LLM_MODEL_FAST",    os.getenv("GROQ_MODEL", _DEFAULT_MODEL))
MODEL_QUALITY = os.getenv("LLM_MODEL_QUALITY", os.getenv("GROQ_MODEL", _DEFAULT_MODEL))
MODEL = MODEL_QUALITY  # back-compat alias


# ── Rate-limit retry helper ────────────────────────────────────────────────────
def _api_call_with_retry(fn, *args, max_retries: int = 4, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(k in err for k in ("429", "rate limit", "quota", "resource_exhausted", "too many requests"))
            if is_rate_limit and attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)  # 5 → 10 → 20 → 40 s
                logger.warning(f"Rate limit hit (attempt {attempt+1}/{max_retries}), retrying in {wait}s…")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Max retries exceeded")


# ── Structured output helper (json_object mode, model-agnostic) ──────────────
def _extract_json_object(text: str) -> str:
    """Best-effort extraction of a JSON object from a raw LLM response."""
    if not text:
        return "{}"
    # Strip markdown fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    # Take the largest balanced {...} substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_structured(messages: list, response_model, model: str = None):
    """Call chat completion in json_object mode and parse into a Pydantic model.

    Tries `response_format=json_object` first; if that fails (e.g. the model
    emitted preamble or the provider rejects the output), retries without the
    format constraint and salvages the JSON object from the response.
    """
    schema = response_model.model_json_schema()
    primer = {
        "role": "system",
        "content": (
            "You MUST return exactly one JSON object that conforms to this JSON Schema. "
            "Output the JSON object and NOTHING else — no prose, no markdown fences, "
            "no commentary before or after. Start the response with `{` and end with `}`.\n\n"
            f"Schema:\n{json.dumps(schema)}"
        ),
    }
    msgs = [primer, *messages]
    target_model = model or MODEL_QUALITY

    # Attempt 1: strict json_object mode
    try:
        completion = _api_call_with_retry(
            client.chat.completions.create,
            model=target_model,
            messages=msgs,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        return response_model.model_validate_json(content)
    except Exception as e:
        err = str(e).lower()
        # Bubble rate limits straight up; only fall through on JSON-validation errors.
        if any(k in err for k in ("429", "rate limit", "quota", "resource_exhausted")):
            raise
        logger.warning(f"Strict JSON mode failed ({e.__class__.__name__}); retrying with salvage parse")

    # Attempt 2: free-form, then salvage JSON
    completion = _api_call_with_retry(
        client.chat.completions.create,
        model=target_model,
        messages=msgs,
    )
    raw = completion.choices[0].message.content or "{}"
    return response_model.model_validate_json(_extract_json_object(raw))


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT MODELS (Pydantic — parsed by OpenAI directly)
# ══════════════════════════════════════════════════════════════════════════════


class Metric(BaseModel):
    name: str = Field(description="Name of the metric (e.g., accuracy, F1, BLEU)")
    value: str = Field(description="Reported value or qualitative result")


class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    relation: str = Field(
        description="Relation type: proposes, uses, evaluates_on, outperforms, extends, cites, applies, compares_with"
    )
    target: str = Field(description="Target entity name")


class PaperEntities(BaseModel):
    """Structured extraction of entities from a research paper."""

    title: Optional[str] = Field(description="Paper title if found, else null")
    authors: list[str] = Field(description="Author names found in the paper")
    methods: list[str] = Field(
        description="Methods, algorithms, or techniques mentioned"
    )
    datasets: list[str] = Field(description="Dataset names mentioned")
    metrics: list[Metric] = Field(description="Metrics and their reported values")
    key_concepts: list[str] = Field(
        description="Important domain concepts and technical terms"
    )
    relationships: list[Relationship] = Field(
        description="Explicit relationships between entities found in the text"
    )


class QueryClassification(BaseModel):
    """Router: classify what kind of query this is."""

    query_type: str = Field(
        description="Type: 'factual' (specific fact), 'comparative' (compare papers/methods), 'exploratory' (broad overview), 'relational' (connections between entities)"
    )
    key_entities: list[str] = Field(
        description="Key entities/concepts from the question to search for"
    )
    search_strategy: str = Field(
        description="Recommended search: 'vector_heavy' (rely on passages), 'graph_heavy' (rely on entity connections), 'balanced' (both)"
    )


class CitedSource(BaseModel):
    """A grounded citation. paper_title and page are derived server-side from
    passage_idx — do not have the LLM emit them. The LLM emits only:
      - passage_idx: the 1-indexed passage number it pulled the quote from
      - quote: a verbatim contiguous quote from that passage's text
      - relevant_detail: one short sentence describing what the quote supports
    """

    passage_idx: int = Field(
        description="1-indexed passage number from the numbered context list. MUST refer to an actual supplied passage."
    )
    quote: str = Field(
        description="Verbatim contiguous quote (5-30 words) copied character-for-character from the passage. No paraphrasing."
    )
    relevant_detail: str = Field(
        description="One short sentence describing what claim this quote supports."
    )
    # Filled in server-side from passage_idx — LLM does not emit these.
    paper_title: Optional[str] = Field(default=None)
    page: Optional[int] = Field(default=None)
    chunk_id: Optional[str] = Field(default=None)
    verified: Optional[bool] = Field(default=None)


class FaithfulnessReport(BaseModel):
    """Output of the post-generation fact-check pass."""

    unsupported_claims: list[str] = Field(
        description="Substantive factual claims from the ANSWER that are NOT supported by any PASSAGE. "
                    "Quote each claim verbatim from the answer. Skip meta-statements, hedges, and definitions."
    )
    support_score: float = Field(
        description="Fraction of substantive factual claims in the ANSWER that ARE supported by the PASSAGES (0..1)."
    )
    notes: str = Field(
        default="",
        description="One-sentence reasoning."
    )


class QueryAnswer(BaseModel):
    """Structured answer to a user's question."""

    answer: str = Field(description="Clear, comprehensive answer to the question")
    sources: list[CitedSource] = Field(
        description="Grounded citations. Each source must reference a numbered passage and quote it verbatim."
    )
    confidence: float = Field(
        description="Confidence in the answer (0-1), lower if context was sparse"
    )
    follow_up_questions: list[str] = Field(
        description="2-3 suggested follow-up questions the user might ask"
    )


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION CALLING TOOLS (for GraphRAG query pipeline)
# ══════════════════════════════════════════════════════════════════════════════

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_vector_store",
            "description": "Search the vector store for text passages relevant to the user's question. Returns top matching chunks from indexed papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query to find relevant passages"},
                    "top_k": {"type": "integer", "description": "Number of results to return (1-20)"},
                },
                "required": ["query", "top_k"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "traverse_knowledge_graph",
            "description": "Traverse the knowledge graph to find entities and their relationships relevant to the query. Useful for finding connections between papers, methods, datasets, and authors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names to look up in the graph"},
                    "hops": {"type": "integer", "description": "Number of hops to traverse from each entity (1-3)"},
                },
                "required": ["entities", "hops"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_details",
            "description": "Get full metadata and extracted entities for a specific paper by title or ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_title": {"type": "string", "description": "Title or partial title of the paper to look up"},
                },
                "required": ["paper_title"],
            },
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════


def _vector_search_raw(query: str, top_k: int, paper_ids: list = None) -> list[dict]:
    """Pure vector search via ChromaDB. Returns chunk dicts with full text."""
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
        got = collection.get(ids=list(chunk_ids), include=["documents", "metadatas"])
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
    if collection.count() == 0:
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


def traverse_knowledge_graph(entities: list[str], hops: int = 2) -> str:
    """Traverse the knowledge graph from given entities."""
    relevant_info = []

    for entity in entities:
        entity_lower = entity.lower().strip()
        entity_canon = _canonicalize_entity(entity)
        for node_id, data in kg.nodes(data=True):
            node_name = data.get("name", data.get("title", "")).lower()
            node_canon = _canonicalize_entity(data.get("name", data.get("title", "")))
            if (
                entity_lower in node_name
                or node_name in entity_lower
                or (entity_canon and entity_canon == node_canon)
                or (entity_canon and (entity_canon in node_canon or node_canon in entity_canon))
            ):
                visited = {node_id}
                frontier = [node_id]
                for _ in range(hops):
                    next_frontier = []
                    for n in frontier:
                        neighbors = list(kg.successors(n)) + list(kg.predecessors(n))
                        for nb in neighbors:
                            if nb not in visited:
                                visited.add(nb)
                                next_frontier.append(nb)
                    frontier = next_frontier

                for nid in visited:
                    ndata = kg.nodes[nid]
                    node_info = {
                        "id": nid,
                        "type": ndata.get("type", "unknown"),
                        "name": ndata.get("name", ndata.get("title", nid)),
                        "connections": [],
                    }
                    for _, target, edata in kg.out_edges(nid, data=True):
                        tdata = kg.nodes.get(target, {})
                        node_info["connections"].append(
                            {
                                "relation": edata.get("relation", "related"),
                                "target": tdata.get("name", tdata.get("title", target)),
                                "target_type": tdata.get("type", "unknown"),
                            }
                        )
                    for source, _, edata in kg.in_edges(nid, data=True):
                        sdata = kg.nodes.get(source, {})
                        node_info["connections"].append(
                            {
                                "relation": f"<-{edata.get('relation', 'related')}-",
                                "target": sdata.get("name", sdata.get("title", source)),
                                "target_type": sdata.get("type", "unknown"),
                            }
                        )
                    relevant_info.append(node_info)

    if not relevant_info:
        return json.dumps({"results": [], "message": f"No matches found for: {entities}"})

    seen = set()
    unique = []
    for info in relevant_info:
        if info["id"] not in seen:
            seen.add(info["id"])
            unique.append(info)

    return json.dumps({"results": unique[:30]})


def get_paper_details(paper_title: str) -> str:
    """Get paper metadata by title."""
    paper_title_lower = paper_title.lower()
    for pid, pdata in papers_db.items():
        if paper_title_lower in pdata.get("title", "").lower():
            return json.dumps(
                {
                    "paper_id": pid,
                    "title": pdata["title"],
                    "authors": pdata.get("entities", {}).get("authors", []),
                    "methods": pdata.get("entities", {}).get("methods", []),
                    "datasets": pdata.get("entities", {}).get("datasets", []),
                    "key_concepts": pdata.get("entities", {}).get("key_concepts", []),
                    "metrics": pdata.get("entities", {}).get("metrics", []),
                    "pages": pdata.get("pages"),
                    "chunks": pdata.get("chunks", 0),
                }
            )
    return json.dumps({"error": f"Paper not found: {paper_title}"})


def call_tool(name: str, args: dict) -> str:
    """Dispatch tool calls."""
    if name == "search_vector_store":
        return search_vector_store(**args)
    elif name == "traverse_knowledge_graph":
        return traverse_knowledge_graph(**args)
    elif name == "get_paper_details":
        return get_paper_details(**args)
    return json.dumps({"error": f"Unknown tool: {name}"})


# ══════════════════════════════════════════════════════════════════════════════
# PDF PROCESSING
# ══════════════════════════════════════════════════════════════════════════════


def extract_text_from_pdf(file_path: str) -> list[dict]:
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


_TITLE_JUNK_MARKERS = ("untitled", "microsoft word", "untitled document", "doc.tex", ".tex")


def _looks_like_real_title(title: str) -> bool:
    if not title:
        return False
    if any(b in title.lower() for b in _TITLE_JUNK_MARKERS):
        return False
    if len(title) < 4 or len(title) > 250:
        return False
    return True


def extract_pdf_metadata_title(file_path: str) -> str:
    """Pull the PDF's embedded metadata title. Some arXiv PDFs populate this
    (newer submissions) but older ones don't, so this is a soft fallback."""
    try:
        doc = fitz.open(file_path)
        title = ((doc.metadata or {}).get("title") or "").strip()
        doc.close()
        return title if _looks_like_real_title(title) else ""
    except Exception:
        return ""


def _llm_title_is_grounded(title: str, full_text: str) -> bool:
    """Return True if the LLM-extracted title actually appears in the source.

    The 8B model sometimes paraphrases the title (e.g. emits 'Transfer Learning
    for NLP via BERT' for the BERT paper, whose real title is 'BERT:
    Pre-training of Deep Bidirectional Transformers for Language
    Understanding'). Verbatim substring match in the head of the document is a
    cheap, reliable check — if the LLM extracted what's actually written, the
    string will be there. Trailing punctuation differences (':', '.', ',') are
    ignored because PDF text extraction occasionally drops them."""
    if not title or not full_text:
        return False
    head = full_text[:6000].lower()
    needle = title.lower().strip().rstrip(".,:;")
    if len(needle) < 4:
        return False
    return needle in head


def extract_first_page_title_heuristic(file_path: str) -> str:
    """Pick out the title by finding the largest-font text in the upper half of
    page 1, ignoring rotated text and left-margin watermarks (the arXiv version
    stamp is a vertical span at x≈11 that would otherwise outrank the real
    title). Works when both LLM extraction and PDF metadata title come up
    empty — e.g. older arXiv papers like 1706.03762."""
    try:
        doc = fitz.open(file_path)
        if len(doc) == 0:
            doc.close()
            return ""
        page = doc[0]
        blocks = page.get_text("dict").get("blocks", [])
        page_h, page_w = page.rect.height, page.rect.width
        doc.close()
        spans = []
        for b in blocks:
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    x0, y0, x1, y1 = span.get("bbox", [0, 0, 0, 0])
                    w, h = x1 - x0, y1 - y0
                    if y0 > page_h * 0.5:
                        continue
                    if x0 < page_w * 0.05:
                        continue
                    if h > w:
                        continue
                    txt = (span.get("text") or "").strip()
                    sz = span.get("size", 0)
                    if not txt or sz < 1:
                        continue
                    spans.append((sz, txt, y0, x0))
        if not spans:
            return ""
        max_size = max(s[0] for s in spans)
        title_spans = [s for s in spans if abs(s[0] - max_size) < 0.5]
        title_spans.sort(key=lambda s: (s[2], s[3]))
        title = re.sub(r"\s+", " ", " ".join(s[1] for s in title_spans)).strip()
        return title if _looks_like_real_title(title) else ""
    except Exception:
        return ""


# Paragraph and sentence boundaries — used by the chunker to avoid mid-sentence cuts.
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")
# Split on .!? followed by whitespace + uppercase / quote / paren start.
# Naive but adequate for English research-paper prose.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARA_SPLIT_RE.split(text or "") if p.strip()]


def _split_sentences(para: str) -> list[str]:
    sents = _SENT_SPLIT_RE.split(para or "")
    return [s.strip() for s in sents if s.strip()]


def _word_count(s: str) -> int:
    return len((s or "").split())


def _pack_units(units: list[str], target_words: int, overlap_words: int) -> list[str]:
    """Greedy pack of text units (paragraphs or sentences) into target-sized chunks
    with word-level overlap from the prior chunk's tail. If a single unit exceeds
    `target_words`, it is recursively split into sentences."""
    chunks: list[str] = []
    cur = ""
    cur_wc = 0

    def flush_with_overlap(next_unit: str) -> tuple[str, int]:
        words = cur.split()
        tail = " ".join(words[-overlap_words:]) if len(words) > overlap_words else cur
        new = (tail + "\n\n" + next_unit) if tail else next_unit
        return new, _word_count(new)

    for u in units:
        uwc = _word_count(u)
        if uwc > target_words:
            # Single unit too big — flush current, then chunk this unit by sentence.
            if cur:
                chunks.append(cur)
                cur, cur_wc = "", 0
            sentences = _split_sentences(u)
            if len(sentences) <= 1:
                # Sentence splitter couldn't break it (e.g. one giant sentence /
                # garbled OCR). Fall back to a hard word window so the chunk
                # still fits the embedding budget.
                words = u.split()
                stride = max(1, target_words - overlap_words)
                for i in range(0, len(words), stride):
                    piece = " ".join(words[i : i + target_words])
                    if piece:
                        chunks.append(piece)
            else:
                chunks.extend(_pack_units(sentences, target_words, overlap_words))
            continue

        if not cur or cur_wc + uwc <= target_words:
            cur = (cur + "\n\n" + u) if cur else u
            cur_wc += uwc
        else:
            chunks.append(cur)
            cur, cur_wc = flush_with_overlap(u)
    if cur:
        chunks.append(cur)
    return chunks


def chunk_text(text: str, target_words: int = 400, overlap_words: int = 80) -> list[str]:
    """Paragraph- then sentence-aware chunking. Used for notes (no page metadata)."""
    paras = _split_paragraphs(text)
    if not paras:
        return []
    return [c for c in _pack_units(paras, target_words, overlap_words) if c.strip()]


def chunk_pages(pages: list[dict], target_words: int = 400, overlap_words: int = 80) -> list[dict]:
    """Chunk per-page so each chunk carries a single page number for provenance."""
    out: list[dict] = []
    for p in pages:
        paras = _split_paragraphs(p["text"])
        if not paras:
            continue
        for chunk in _pack_units(paras, target_words, overlap_words):
            if chunk.strip():
                out.append({"text": chunk, "page": p["page"]})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ENTITY EXTRACTION (Structured Outputs)
# ══════════════════════════════════════════════════════════════════════════════


_EXTRACT_SYS_PROMPT = (
    "You are an expert academic entity extractor. "
    "Given text from a research paper, extract all structured entities and relationships. "
    "Be thorough — extract every author, method, dataset, metric, and concept you can find.\n\n"
    "EXTRACT THE FORM AS IT APPEARS IN THE TEXT. Do NOT canonicalize, expand, or paraphrase. "
    "If the paper writes 'RNN', emit 'RNN'; if it writes 'recurrent neural networks', emit that. "
    "If the paper writes 'BLEU score', emit 'BLEU score'. The system has a separate canonicalization "
    "layer that links variants like 'RNN' ↔ 'recurrent neural network' across papers — your job is "
    "to faithfully report what THIS section says.\n\n"
    "DO NOT INVENT entities. If you are not certain a method/dataset/metric is explicitly mentioned in THIS section, "
    "leave it out — a downstream validator will drop entities that do not appear in the source. "
    "For relationships, only extract those grounded in this section's text "
    "(e.g., a method 'uses' a dataset, a paper 'proposes' an algorithm, a model 'outperforms' a baseline).\n\n"
    "IGNORE the References / Bibliography / Citations section. Do NOT extract cited paper titles "
    "(e.g. 'Adam: A method for stochastic optimization', 'Long short-term memory') as methods or concepts, "
    "and do NOT extract author lists from citation entries (e.g. 'Mitchell P. Marcus, Mary Ann Marcinkiewicz, ...') "
    "as authors of THIS paper — they are authors of cited works. Only extract entities that the paper itself "
    "uses, proposes, evaluates, or describes as its own.\n\n"
    "Also IGNORE inline citations in the body — strings like '(Howard and Ruder, 2018)', 'Fedus et al. (2018)', "
    "'Dai and Le, 2015', 'Howard and Ruder', 'Dai and Le', or 'Radford et al., 2018' are references to OTHER "
    "papers, not authors of THIS one. Authors of THIS paper appear ONLY on the title page (typically right "
    "under the title). Anything matching 'Surname et al.', 'Surname and Surname', or anything containing a "
    "4-digit year is a citation and MUST NOT appear in the authors list."
)


_REFERENCES_HEADING_RE = re.compile(
    r"^\s*(?:\d+\s*[.)]?\s+)?(?:References?|REFERENCES|Bibliography|BIBLIOGRAPHY|Works\s+Cited)\s*:?\s*$",
    flags=re.MULTILINE,
)


def _strip_references_section(text: str) -> str:
    """Truncate the paper at the start of its References / Bibliography section.

    Citations leak into entity extraction otherwise: reference paper titles get
    tagged as 'methods' and citation author lists get tagged as 'authors'. The
    8B model is especially prone to this. Stripping references before slicing
    keeps the extraction focused on the paper's own contributions.

    The references heading is searched only in the latter half of the document
    so an incidental in-prose mention of the word 'references' earlier doesn't
    accidentally truncate the body."""
    if not text or len(text) < 2000:
        return text
    half = len(text) // 2
    last_match = None
    for m in _REFERENCES_HEADING_RE.finditer(text, pos=half):
        last_match = m
    if last_match:
        return text[: last_match.start()].rstrip()
    return text


# Tunables (configurable via env so we can dial extraction cost/coverage on HF Spaces
# without redeploying code).
_EXTRACT_MAX_CHUNKS = int(os.getenv("EXTRACTION_MAX_CHUNKS", "3"))
_EXTRACT_CHUNK_CHARS = int(os.getenv("EXTRACTION_CHUNK_CHARS", "8000"))
_EXTRACT_OVERLAP_CHARS = int(os.getenv("EXTRACTION_OVERLAP_CHARS", "600"))
_SKIP_ENTITY_VALIDATION = os.getenv("SKIP_ENTITY_VALIDATION", "").lower() in ("1", "true", "yes", "on")


def _slice_for_extraction(
    text: str,
    max_chunks: int = None,
    target_chars: int = None,
    overlap_chars: int = None,
) -> list[str]:
    """Slice the paper into windows for multi-pass extraction. Always keeps the
    head (title/abstract/intro) and the tail (results/conclusion); fills the
    middle up to `max_chunks` total. Defaults come from env vars so the
    extraction budget can be tuned without a redeploy."""
    text = text or ""
    max_chunks = max_chunks if max_chunks is not None else _EXTRACT_MAX_CHUNKS
    target_chars = target_chars if target_chars is not None else _EXTRACT_CHUNK_CHARS
    overlap_chars = overlap_chars if overlap_chars is not None else _EXTRACT_OVERLAP_CHARS
    if len(text) <= target_chars:
        return [text] if text.strip() else []
    stride = max(1, target_chars - overlap_chars)
    windows = []
    i = 0
    while i < len(text) and len(windows) < max_chunks - 1:  # reserve last slot for tail
        windows.append(text[i : i + target_chars])
        i += stride
    # Always include the tail explicitly so we don't miss late sections.
    tail = text[-target_chars:]
    if not windows or windows[-1] != tail:
        windows.append(tail)
    return windows[:max_chunks]


def _merge_extractions(extractions: list[dict]) -> dict:
    """Merge per-chunk PaperEntities dicts. Dedupe by canonical form."""
    merged = _empty_extraction()
    # Title: prefer the first non-empty (typically from the head chunk).
    for e in extractions:
        if e.get("title"):
            merged["title"] = e["title"]
            break

    def _dedupe_strs(field: str):
        seen, keep = set(), []
        for e in extractions:
            for v in e.get(field, []) or []:
                c = _canonicalize_entity(v)
                if c and c not in seen:
                    seen.add(c)
                    keep.append(v)
        return keep

    merged["authors"] = _dedupe_strs("authors")
    merged["methods"] = _dedupe_strs("methods")
    merged["datasets"] = _dedupe_strs("datasets")
    merged["key_concepts"] = _dedupe_strs("key_concepts")

    # Metrics: keyed on (canonical name, value).
    seen_metrics = set()
    for e in extractions:
        for m in e.get("metrics", []) or []:
            if isinstance(m, dict):
                name, val = m.get("name", ""), m.get("value", "")
            else:
                name, val = str(m), ""
            key = (_canonicalize_entity(name), str(val))
            if key[0] and key not in seen_metrics:
                seen_metrics.add(key)
                merged["metrics"].append(
                    m if isinstance(m, dict) else {"name": str(m), "value": ""}
                )

    # Relationships: keyed on (canon src, relation, canon tgt).
    seen_rels = set()
    for e in extractions:
        for r in e.get("relationships", []) or []:
            src = _canonicalize_entity(r.get("source", ""))
            tgt = _canonicalize_entity(r.get("target", ""))
            rel = r.get("relation", "")
            key = (src, rel, tgt)
            if src and tgt and key not in seen_rels:
                seen_rels.add(key)
                merged["relationships"].append(r)
    return merged


def _normalize_for_appearance(s: str) -> str:
    """Lowercase, strip parens/brackets, collapse whitespace.
    Used for fuzzy substring tests when checking entity appearance in source."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[\(\)\[\]]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _entity_appears(name: str, text_lower: str) -> bool:
    """Lenient check: True iff some recognizable form of this entity appears in
    the source text. Used to drop hallucinated entities WITHOUT discarding valid
    ones whose surface form differs from the LLM-emitted form.

    Match strategies, in order:
      1. Direct substring (name in text).
      2. Parens/whitespace-normalized substring.
      3. Canonical form (via alias map) substring.
      4. Any KNOWN ALIAS that maps to the same canonical form (e.g., LLM emitted
         'recurrent neural network' but text says 'RNN' — accept).
      5. Token overlap >=70% on tokens of length >=3 (catches minor wording
         differences without admitting unrelated entities).
    """
    n = (name or "").lower().strip()
    if not n or len(n) < 2:
        return False
    if n in text_lower:
        return True
    # Normalize both sides (strip parens, collapse whitespace).
    nn = _normalize_for_appearance(n)
    nt = _normalize_for_appearance(text_lower)
    if nn and nn in nt:
        return True
    # Canonical and all known aliases that share the canonical.
    canon = _canonicalize_entity(name)
    if canon and (canon in text_lower or canon in nt):
        return True
    if canon:
        for surface, mapped in _ENTITY_ALIASES.items():
            if mapped == canon and (surface in text_lower or surface in nt):
                return True
    # Token-overlap fallback for multi-word entities.
    tokens = [t for t in _TOKEN_RE.findall(n) if len(t) >= 3]
    if len(tokens) >= 2:
        hit = sum(1 for t in tokens if t in text_lower)
        if hit / len(tokens) >= 0.7:
            return True
    return False


# Inline citations like 'Fedus et al. (2018)', 'Howard and Ruder, 2018', or the
# bare two-author form 'Dai and Le' are embedded throughout paper bodies, so the
# references-section stripper can't remove them. Drop anything matching one of
# three citation signatures — real author entries never contain these patterns:
#   - 'et al' (with or without trailing period)
#   - a 4-digit year (1900–2099)
#   - the standalone word 'and' (two-author citation: 'Howard and Ruder')
# Word boundaries ensure 'and' inside names like 'Andrea' or 'Fernandez' is
# safe — those have no \b adjacent to the letters and-n-d.
_CITATION_PATTERN_RE = re.compile(
    r"\bet\s+al\b|\b(?:19|20)\d{2}\b|\band\b",
    flags=re.IGNORECASE,
)


def _looks_like_citation(name: str) -> bool:
    return bool(_CITATION_PATTERN_RE.search(name or ""))


def _author_appears(name: str, text_lower: str) -> bool:
    """Author names vary wildly across papers ('John Smith', 'J. Smith',
    'Smith, J.', 'Smith et al.'). Validate by last-name presence with a
    one-letter first-initial sanity check when ambiguous."""
    if not name:
        return False
    if _looks_like_citation(name):
        return False
    parts = [p for p in re.split(r"[\s,]+", name.lower().strip()) if p]
    if not parts:
        return False
    # Real author names always have at least two whitespace/comma-separated
    # tokens (first+last, initial+last, or 'last, first'). Single-token
    # 'authors' like 'OpenAI', 'Google', 'DeepMind' are organizations the LLM
    # picked up from comparison/related-work mentions, not authors of THIS
    # paper.
    if len(parts) < 2:
        return False
    # Heuristic: longest token is likely the surname (initials are short).
    surname = max((p for p in parts if len(p) >= 2), key=len, default="")
    if len(surname) < 2:
        return False
    return surname in text_lower


def _validate_extraction(entities: dict, full_text: str) -> tuple[dict, dict]:
    """Drop entities whose surface form doesn't appear in the source.
    Returns (kept, dropped_counts).

    Honors `SKIP_ENTITY_VALIDATION=1` for emergency bypass — useful if the
    validator is being too strict on a particular paper format and we want to
    fall back to the raw LLM extraction."""
    dropped = {"authors": 0, "methods": 0, "datasets": 0, "key_concepts": 0, "metrics": 0, "relationships": 0}
    if _SKIP_ENTITY_VALIDATION:
        return dict(entities), dropped

    text_lower = (full_text or "").lower()
    out = dict(entities)

    def filt(field: str, check=_entity_appears) -> list:
        kept = []
        for v in entities.get(field, []) or []:
            if check(v, text_lower):
                kept.append(v)
            else:
                dropped[field] += 1
        return kept

    out["authors"] = filt("authors", check=_author_appears)
    out["methods"] = filt("methods")
    out["datasets"] = filt("datasets")
    out["key_concepts"] = filt("key_concepts")

    metrics = []
    for m in entities.get("metrics", []) or []:
        name = m.get("name", "") if isinstance(m, dict) else str(m)
        if _entity_appears(name, text_lower):
            metrics.append(m)
        else:
            dropped["metrics"] += 1
    out["metrics"] = metrics

    rels = []
    for r in entities.get("relationships", []) or []:
        if _entity_appears(r.get("source", ""), text_lower) and _entity_appears(r.get("target", ""), text_lower):
            rels.append(r)
        else:
            dropped["relationships"] += 1
    out["relationships"] = rels

    return out, dropped


def extract_entities(text: str) -> dict:
    """Multi-pass entity extraction over the full paper, with validation.

    Pipeline:
      1. Slice the paper into ~6k-char windows (overlapping). Cap at 5 windows.
      2. Run structured extraction on each slice — IN PARALLEL via a thread pool.
         Each call is an independent LLM request, so wall time collapses to roughly
         the slowest single call instead of the sum.
      3. Merge — dedupe entities by canonical form.
      4. Validate — drop entities that don't literally appear in the source.
    """
    if not client:
        logger.warning("No LLM key — returning empty extraction")
        return _empty_extraction()

    # Strip the references/bibliography section before slicing AND before the
    # downstream validator runs — otherwise reference paper titles and citation
    # author lists pollute the extraction (they appear in source text, so the
    # validator can't drop them).
    original_len = len(text or "")
    text = _strip_references_section(text)
    if original_len and len(text) < original_len:
        logger.info("Stripped references section: %d -> %d chars", original_len, len(text))

    slices = _slice_for_extraction(text)
    if not slices:
        return _empty_extraction()

    logger.info("Extracting entities over %d slice(s) of length up to ~%d chars (parallel)...",
                len(slices), max((len(s) for s in slices), default=0))

    def _run_slice(idx: int, sl: str) -> Optional[dict]:
        try:
            result = _parse_structured(
                messages=[
                    {"role": "system", "content": _EXTRACT_SYS_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Extract entities and relationships from this section "
                            f"({idx+1}/{len(slices)}) of a research paper:\n\n{sl}"
                        ),
                    },
                ],
                response_model=PaperEntities,
                model=MODEL_FAST,
            )
            return result.model_dump()
        except Exception as e:
            logger.warning(f"Extraction pass {idx+1}/{len(slices)} failed: {e}")
            return None

    extractions: list[dict] = []
    if len(slices) == 1:
        # Single slice (e.g. a short note) — no need for the thread pool overhead.
        r = _run_slice(0, slices[0])
        if r:
            extractions.append(r)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = min(len(slices), int(os.getenv("EXTRACTION_PARALLELISM", "4")))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_run_slice, idx, sl) for idx, sl in enumerate(slices)]
            for f in as_completed(futures):
                r = f.result()
                if r:
                    extractions.append(r)

    if not extractions:
        return _empty_extraction()

    merged = _merge_extractions(extractions)
    validated, dropped_counts = _validate_extraction(merged, text)

    logger.info(
        "Extraction complete (%d/%d passes) — %d authors, %d methods, %d datasets, "
        "%d concepts, %d metrics, %d relationships (dropped by validation: %s)",
        len(extractions), len(slices),
        len(validated["authors"]), len(validated["methods"]), len(validated["datasets"]),
        len(validated["key_concepts"]), len(validated["metrics"]), len(validated["relationships"]),
        dropped_counts,
    )
    return validated


def _empty_extraction() -> dict:
    return {
        "title": None,
        "authors": [],
        "methods": [],
        "datasets": [],
        "metrics": [],
        "key_concepts": [],
        "relationships": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

# Canonical aliases — collapses surface variants to a single graph node so two
# papers that talk about the same thing actually link in the graph.
_ENTITY_ALIASES: dict = {
    # Architectures
    "rnn": "recurrent neural network",
    "rnns": "recurrent neural network",
    "recurrent neural networks": "recurrent neural network",
    "lstm": "long short-term memory",
    "lstms": "long short-term memory",
    "gru": "gated recurrent unit",
    "grus": "gated recurrent unit",
    "cnn": "convolutional neural network",
    "cnns": "convolutional neural network",
    "convolutional neural networks": "convolutional neural network",
    "mlp": "multi-layer perceptron",
    "mlps": "multi-layer perceptron",
    "ffn": "feed-forward network",
    "ffnn": "feed-forward network",
    # Attention family
    "attention": "attention",
    "attention mechanism": "attention",
    "attention mechanisms": "attention",
    "attentional mechanism": "attention",
    "attentional mechanisms": "attention",
    "soft attention": "attention",
    "additive attention": "additive attention",
    "bahdanau attention": "additive attention",
    "luong attention": "multiplicative attention",
    "multiplicative attention": "multiplicative attention",
    "dot-product attention": "scaled dot-product attention",
    "scaled dot product attention": "scaled dot-product attention",
    "multi head attention": "multi-head attention",
    "self attention": "self-attention",
    # Generic phrases
    "encoder decoder": "encoder-decoder",
    "encoder-decoder model": "encoder-decoder",
    "encoder-decoder models": "encoder-decoder",
    "encoder-decoder architecture": "encoder-decoder",
    "encoder-decoder approach": "encoder-decoder",
    "encoder-decoder framework": "encoder-decoder",
    "neural machine translation": "neural machine translation",
    "nmt": "neural machine translation",
    "machine translation": "machine translation",
    "sequence to sequence": "sequence-to-sequence",
    "sequence-to-sequence": "sequence-to-sequence",
    "seq2seq": "sequence-to-sequence",
    "seq-to-seq": "sequence-to-sequence",
    "transformer": "transformer",
    "transformer model": "transformer",
    "transformer architecture": "transformer",
    "vanilla transformer": "transformer",
    # Datasets
    "wmt'14": "wmt 2014",
    "wmt 14": "wmt 2014",
    "wmt14": "wmt 2014",
    "wmt 2014 english-french": "wmt 2014 en-fr",
    "wmt'14 english-french": "wmt 2014 en-fr",
    "wmt'14 english to french": "wmt 2014 en-fr",
    "wmt 2014 english-german": "wmt 2014 en-de",
    "wmt'14 english-german": "wmt 2014 en-de",
    "wmt'14 english to german": "wmt 2014 en-de",
    "english-french translation": "wmt 2014 en-fr",
    "english-to-french translation": "wmt 2014 en-fr",
    "english-german translation": "wmt 2014 en-de",
    "english-to-german translation": "wmt 2014 en-de",
    "wmt'15": "wmt 2015",
    "wmt 15": "wmt 2015",
    "wmt15": "wmt 2015",
    "iwslt'14": "iwslt 2014",
    "iwslt 14": "iwslt 2014",
    # Metrics
    "bleu score": "bleu",
    "bleu scores": "bleu",
    "bleu metric": "bleu",
}

_TRAILING_NOISE = (
    " models", " model", " mechanisms", " mechanism", " approach", " approaches",
    " architecture", " architectures", " framework", " frameworks",
    " method", " methods", " task", " tasks", " dataset", " datasets",
    " score", " scores", " metric", " metrics",
)


def _canonicalize_entity(name: str) -> str:
    """Lowercase, strip punctuation, expand aliases, drop trailing noise."""
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[''‘’“”`,;:.\(\)\[\]]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in _ENTITY_ALIASES:
        return _ENTITY_ALIASES[s]
    for suf in _TRAILING_NOISE:
        if s.endswith(suf):
            stripped = s[: -len(suf)].strip()
            if stripped in _ENTITY_ALIASES:
                return _ENTITY_ALIASES[stripped]
            if stripped:
                s = stripped
            break
    return s


def _add_entity_node(node_id: str, ntype: str, surface: str, **extra):
    """Add or merge an entity node, keeping the shortest surface form as the label."""
    if kg.has_node(node_id):
        existing_label = kg.nodes[node_id].get("label", "")
        # Prefer the shorter, cleaner surface form for the visible label
        if len(surface) < len(existing_label) and len(surface) > 1:
            kg.nodes[node_id]["label"] = surface
            kg.nodes[node_id]["name"] = surface
    else:
        kg.add_node(node_id, type=ntype, name=surface, label=surface, **extra)


def add_to_knowledge_graph(paper_id: str, entities: dict):
    paper_title = entities.get("title") or paper_id
    kg.add_node(paper_id, type="paper", title=paper_title, label=paper_title)

    for author in entities.get("authors", []):
        canon = _canonicalize_entity(author)
        if not canon:
            continue
        aid = f"author:{canon}"
        _add_entity_node(aid, "author", author)
        kg.add_edge(aid, paper_id, relation="authored")

    for method in entities.get("methods", []):
        canon = _canonicalize_entity(method)
        if not canon:
            continue
        mid = f"method:{canon}"
        _add_entity_node(mid, "method", method)
        kg.add_edge(paper_id, mid, relation="proposes")

    for dataset in entities.get("datasets", []):
        canon = _canonicalize_entity(dataset)
        if not canon:
            continue
        did = f"dataset:{canon}"
        _add_entity_node(did, "dataset", dataset)
        kg.add_edge(paper_id, did, relation="evaluates_on")

    for metric in entities.get("metrics", []):
        if isinstance(metric, dict):
            m_name = metric.get("name", "unknown")
            m_val = metric.get("value", "")
        else:
            m_name, m_val = str(metric), ""
        canon = _canonicalize_entity(m_name)
        if not canon:
            continue
        m_id = f"metric:{canon}"
        _add_entity_node(m_id, "metric", m_name, value=m_val)
        kg.add_edge(paper_id, m_id, relation="reports")

    for concept in entities.get("key_concepts", []):
        canon = _canonicalize_entity(concept)
        if not canon:
            continue
        cid = f"concept:{canon}"
        _add_entity_node(cid, "concept", concept)
        kg.add_edge(paper_id, cid, relation="discusses")

    for rel in entities.get("relationships", []):
        raw_src = rel.get("source", "")
        raw_tgt = rel.get("target", "")
        src = _canonicalize_entity(raw_src)
        tgt = _canonicalize_entity(raw_tgt)
        relation = rel.get("relation", "related_to")
        if not (src and tgt):
            continue
        # Reuse an existing node of the SAME canonical name (any type) before falling back to entity:
        src_id = next((nid for nid in kg.nodes if nid.split(":", 1)[-1] == src), f"entity:{src}")
        tgt_id = next((nid for nid in kg.nodes if nid.split(":", 1)[-1] == tgt), f"entity:{tgt}")
        if not kg.has_node(src_id):
            _add_entity_node(src_id, "entity", raw_src)
        if not kg.has_node(tgt_id):
            _add_entity_node(tgt_id, "entity", raw_tgt)
        kg.add_edge(src_id, tgt_id, relation=relation)

    logger.info(f"Graph updated — now {len(kg.nodes)} nodes, {len(kg.edges)} edges")


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
    collection.add(documents=docs, ids=ids, metadatas=metas)
    global _bm25_dirty
    _bm25_dirty = True
    logger.info(f"Added {len(docs)} chunks to vector store")


# ══════════════════════════════════════════════════════════════════════════════
# GraphRAG QUERY (Function Calling + Structured Output)
# ══════════════════════════════════════════════════════════════════════════════


def _papers_in_subgraph(subgraph_json: str) -> list[str]:
    """Extract paper_ids from a traverse_knowledge_graph JSON result."""
    try:
        data = json.loads(subgraph_json)
    except Exception:
        return []
    paper_ids = set()
    for entry in data.get("results", []):
        nid = entry.get("id", "")
        if nid.startswith("paper:"):
            paper_ids.add(nid)
        for c in entry.get("connections", []):
            # Connections aren't tagged with id; we only know the name. Skip.
            pass
    # Also: any paper whose title matches an entity name
    return list(paper_ids)


# ── Quote grounding ────────────────────────────────────────────────────────────
# Folding tables for characters PyMuPDF and the LLM disagree on:
#   smart quotes, various dashes, asterisk lookalikes, nonbreaking spaces, etc.
_QUOTE_FOLDS = [
    (re.compile(r"[‘’`´]"), "'"),
    (re.compile(r"[“”]"), '"'),
    (re.compile(r"[–—−‐-]"), "-"),
    (re.compile(r"[∗⋆★⁎*]"), " "),     # asterisks signal author footnotes — squash to space
    (re.compile(r"[†‡§¶]"), " "),       # footnote daggers
    (re.compile(r"[   ]"), " "),  # nonbreaking spaces
]
_QUOTE_NORMALIZE_RE = re.compile(r"\s+")


def _normalize_for_quote_match(s: str) -> str:
    if not s:
        return ""
    for pat, repl in _QUOTE_FOLDS:
        s = pat.sub(repl, s)
    s = s.lower()
    s = _QUOTE_NORMALIZE_RE.sub(" ", s).strip()
    return s


def _verify_quote(quote: str, chunk_text: str, min_words: int = 3, fuzzy_threshold: float = 0.85) -> bool:
    """Return True if `quote` is plausibly verbatim within `chunk_text`.

    Strategy:
      1. Normalize whitespace + smart-quotes + case on both sides.
      2. Exact substring? Accept.
      3. Otherwise, slide a same-length window across the chunk and accept if
         any window has SequenceMatcher ratio >= fuzzy_threshold (handles minor
         OCR/whitespace artifacts without letting paraphrases through).
      4. Reject quotes shorter than `min_words` words — they are too weak to
         constitute grounding.
    """
    if not quote or not chunk_text:
        return False
    nq = _normalize_for_quote_match(quote)
    nc = _normalize_for_quote_match(chunk_text)
    if len(nq.split()) < min_words:
        return False
    if nq in nc:
        return True
    # Fuzzy fallback: scan windows of same length as the quote.
    import difflib
    L = len(nq)
    if L > len(nc):
        return False
    # Coarse stride to keep this O(n) — granularity ~quote length / 4.
    stride = max(1, L // 4)
    best = 0.0
    for i in range(0, len(nc) - L + 1, stride):
        ratio = difflib.SequenceMatcher(None, nq, nc[i : i + L]).ratio()
        if ratio > best:
            best = ratio
            if best >= fuzzy_threshold:
                return True
    return False


def _check_faithfulness(question: str, answer: str, passages_block: str, graph_block: str) -> Optional[FaithfulnessReport]:
    """Strict fact-check: flag substantive claims in the answer that no passage
    supports. Returns None on failure (we don't want a verifier hiccup to block
    the whole query — we just lose the faithfulness signal)."""
    if not client:
        return None
    if not answer or not answer.strip():
        return None
    # Empty retrieval → we can't verify anything. Mark as zero support.
    if not passages_block or "(none retrieved)" in passages_block:
        return FaithfulnessReport(
            unsupported_claims=[],
            support_score=0.0 if answer.strip() else 1.0,
            notes="No passages retrieved; no grounding available.",
        )

    sys_prompt = (
        "You are a strict fact-checker. You will be given a QUESTION, the PASSAGES that were retrieved "
        "from the user's library, optionally a KNOWLEDGE GRAPH SUBGRAPH, and an ANSWER produced by another model. "
        "Your job is to identify any SUBSTANTIVE FACTUAL CLAIM in the ANSWER that is NOT supported by either the "
        "PASSAGES or the SUBGRAPH.\n\n"
        "What counts as a substantive factual claim:\n"
        "  • specific results / numbers / metrics\n"
        "  • who proposed / authored / introduced what\n"
        "  • method X uses / outperforms / extends Y\n"
        "  • dataset / benchmark mentions\n"
        "  • cross-paper comparisons\n"
        "Do NOT flag: meta-statements ('the papers say...', 'based on context'), hedges, generic background "
        "definitions, or restatements of the question.\n\n"
        "A claim is supported if some passage or subgraph triple states or directly entails it. "
        "Lexical paraphrase is OK; logical leaps are NOT.\n\n"
        "Quote each unsupported claim verbatim from the ANSWER (a short phrase or sentence). "
        "Then output support_score = (supported substantive claims) / (total substantive claims), in [0,1]."
    )
    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"{graph_block}\n\n{passages_block}\n\n"
        f"ANSWER TO VERIFY:\n{answer}"
    )
    try:
        return _parse_structured(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=FaithfulnessReport,
            model=MODEL_FAST,
        )
    except Exception as e:
        logger.warning(f"Faithfulness check failed: {e}")
        return None


def graphrag_query(question: str, top_k: int = 5) -> dict:
    """
    GraphRAG query pipeline (deterministic, graph-driven, grounded):
      1. Classify query → query_type, key_entities, search_strategy.
      2. Pull a 2-hop graph subgraph around key_entities (text format).
      3. Hybrid (BM25 + vector + RRF, optionally cross-encoder reranked) retrieval.
         For comparative/relational queries, retrieve a balanced top-N PER paper
         in the subgraph so one paper doesn't crowd the others out.
      4. LLM produces a structured answer with citations as (passage_idx, quote).
      5. Verify each citation: drop any whose quote doesn't actually appear in
         the cited passage (paper_title and page are derived server-side from
         the passage, never the LLM).
      6. Faithfulness check: a second LLM pass flags substantive claims in the
         answer that no passage supports. Confidence is bounded by the result.
    """
    if not client:
        vector_results = json.loads(search_vector_store(question, top_k))
        return {
            "answer": "LLM API key not configured. Raw search results returned.",
            "sources": [],
            "passages": vector_results.get("results", []),
            "confidence": 0.0,
            "follow_up_questions": [],
        }

    # ── Step 1: Classify ────────────────────────────────────────────────
    logger.info(f"Classifying query: {question}")
    try:
        query_info = _parse_structured(
            messages=[
                {"role": "system", "content": "Classify this research question to determine the best search strategy."},
                {"role": "user", "content": question},
            ],
            response_model=QueryClassification,
            model=MODEL_FAST,
        )
        logger.info(
            f"Classified — type: {query_info.query_type}, "
            f"strategy: {query_info.search_strategy}, "
            f"entities: {query_info.key_entities}"
        )
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        query_info = None

    qtype = (query_info.query_type if query_info else "exploratory").lower()
    strategy = (query_info.search_strategy if query_info else "balanced").lower()
    key_entities = query_info.key_entities if query_info else []

    use_graph = (
        qtype in ("comparative", "relational")
        or strategy in ("graph_heavy", "balanced")
        or len(papers_db) > 1
    )

    # ── Step 2: Deterministic retrieval ─────────────────────────────────
    graph_context = ""
    paper_ids_in_subgraph = []
    if use_graph and key_entities and len(kg.nodes) > 0:
        graph_raw = traverse_knowledge_graph(key_entities, hops=2)
        paper_ids_in_subgraph = _papers_in_subgraph(graph_raw)
        # Hard-cap the subgraph text in the prompt to leave headroom for the
        # passages and the response under provider TPM limits.
        GRAPH_BUDGET = int(os.getenv("PROMPT_GRAPH_BUDGET_CHARS", "4000"))
        graph_text = graph_raw if len(graph_raw) <= GRAPH_BUDGET else graph_raw[:GRAPH_BUDGET] + " […truncated]"
        graph_context = f"KNOWLEDGE GRAPH SUBGRAPH (entities + relationships from your library):\n{graph_text}"

    # For comparative/relational queries, retrieve top chunks PER PAPER from the subgraph
    # so one paper doesn't crowd the others out of the context window.
    if qtype in ("comparative", "relational") and paper_ids_in_subgraph:
        per_paper_k = max(2, top_k // max(len(paper_ids_in_subgraph), 1))
        retrieved_chunks: list[dict] = []
        for pid in paper_ids_in_subgraph:
            retrieved_chunks.extend(_search_chunks(question, top_k=per_paper_k, paper_ids=[pid]))
        retrieval_label = (
            f"balanced — top {per_paper_k} per paper across "
            f"{len(paper_ids_in_subgraph)} papers in subgraph"
        )
    else:
        retrieved_chunks = _search_chunks(question, top_k=top_k)
        retrieval_label = f"top {top_k}"

    # Build numbered passage table the LLM cites by index.
    # Verification still uses the FULL chunk text from passage_lookup, but the
    # LLM-facing prompt truncates each passage to keep the request under the
    # provider TPM cap (Groq free tier rejects single requests > ~12K tokens).
    # The cap is a budget split across passages — when more passages are
    # retrieved, each one gets a smaller slice.
    PROMPT_CHAR_BUDGET = int(os.getenv("PROMPT_PASSAGE_BUDGET_CHARS", "16000"))
    MIN_PER_PASSAGE = 400  # never go below this — too small to find a quote in
    n = max(len(retrieved_chunks), 1)
    per_passage = max(MIN_PER_PASSAGE, PROMPT_CHAR_BUDGET // n)

    passage_lookup: dict[int, dict] = {}
    passage_blocks: list[str] = []
    for i, c in enumerate(retrieved_chunks, start=1):
        passage_lookup[i] = c  # full text retained for verification
        page = f"p.{c['page']}" if c.get("page") else "p.?"
        ptitle = (c.get("paper_title") or "Unknown").replace("\n", " ")
        text = c.get("text") or ""
        if len(text) > per_passage:
            text = text[:per_passage] + " […]"
        passage_blocks.append(f"[{i}] (paper={ptitle!r} | {page})\n{text}")
    vector_context = (
        f"VECTOR PASSAGES ({retrieval_label}):\n" + "\n\n".join(passage_blocks)
        if passage_blocks
        else "VECTOR PASSAGES: (none retrieved)"
    )

    library_titles = [p.get("title", "") for p in papers_db.values()]
    library_listing = "\n".join(f"- {t}" for t in library_titles) or "(empty)"

    # ── Step 3: Generate cited answer ───────────────────────────────────
    logger.info("Generating structured answer over %d numbered passages...", len(passage_lookup))
    sys_prompt = (
        "You are PaperTrail, a research memory assistant. "
        "Answer the user's question using ONLY the supplied KNOWLEDGE GRAPH SUBGRAPH and VECTOR PASSAGES below. "
        "If the supplied context is insufficient, say so plainly — do not fall back to your training data.\n\n"
        "GROUNDING RULES — every CitedSource you emit:\n"
        "  • passage_idx: integer matching one of the numbered passages above (e.g., 3 for [3]). "
        "Only the VECTOR PASSAGES are citable — the KG SUBGRAPH is for structural context, not direct citation.\n"
        "  • quote: a CONTIGUOUS verbatim span (5–30 words) copied character-for-character from that exact passage's text. "
        "Do NOT paraphrase. Do NOT merge text from multiple passages. The system will discard any citation whose quote "
        "does not literally appear in the cited passage.\n"
        "  • relevant_detail: one short sentence describing what the quote supports.\n"
        "Do not include paper_title or page in your output — those are filled in from passage_idx automatically. "
        "If you cannot find a passage that supports a claim, omit the citation rather than inventing one."
    )
    user_prompt = (
        f"User's library ({len(library_titles)} papers):\n{library_listing}\n\n"
        f"{graph_context}\n\n{vector_context}\n\n"
        f"Question: {question}\n\n"
        f"Provide a clear, comprehensive answer grounded strictly in the context above, "
        f"citing each claim with a passage_idx + verbatim quote."
    )

    try:
        final = _parse_structured(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=QueryAnswer,
            model=MODEL_QUALITY,
        )

        # Verify each citation against the actual passage text.
        verified: list[dict] = []
        dropped: list[dict] = []
        for s in final.sources:
            chunk = passage_lookup.get(s.passage_idx)
            if not chunk:
                dropped.append({
                    "passage_idx": s.passage_idx,
                    "quote": s.quote,
                    "reason": "no such passage_idx",
                })
                continue
            if not _verify_quote(s.quote, chunk["text"]):
                dropped.append({
                    "passage_idx": s.passage_idx,
                    "paper_title": chunk.get("paper_title"),
                    "quote": s.quote,
                    "reason": "quote not found in passage",
                })
                continue
            s.paper_title = chunk.get("paper_title")
            s.page = chunk.get("page")
            s.chunk_id = chunk.get("chunk_id")
            s.verified = True
            verified.append(s.model_dump())

        if dropped:
            logger.warning(
                "Dropped %d unverifiable citation(s): %s",
                len(dropped),
                [(d.get("passage_idx"), d.get("reason")) for d in dropped],
            )

        # ── Step 4: Faithfulness check ──────────────────────────────────────
        faith = _check_faithfulness(
            question=question,
            answer=final.answer,
            passages_block=vector_context,
            graph_block=graph_context,
        )
        unsupported = list(faith.unsupported_claims) if faith else []
        support_score = faith.support_score if faith else None

        # Adjust confidence: penalize for unverified citations AND for unsupported claims.
        adj_confidence = final.confidence
        total_emitted = len(verified) + len(dropped)
        if total_emitted > 0 and len(dropped) / total_emitted >= 0.5:
            adj_confidence = min(adj_confidence, 0.4)
        if support_score is not None:
            # Blend: trust the verifier as an upper bound on confidence.
            adj_confidence = min(adj_confidence, support_score)
        if unsupported:
            adj_confidence = min(adj_confidence, max(0.1, 1.0 - 0.15 * len(unsupported)))

        logger.info(
            "Answer generated — confidence: %.2f (raw %.2f, support %.2f), "
            "verified: %d, dropped: %d, unsupported: %d",
            adj_confidence, final.confidence,
            support_score if support_score is not None else -1.0,
            len(verified), len(dropped), len(unsupported),
        )
        return {
            "answer": final.answer,
            "sources": verified,
            "dropped_sources": dropped,
            "unsupported_claims": unsupported,
            "support_score": support_score,
            "confidence": adj_confidence,
            "raw_confidence": final.confidence,
            "follow_up_questions": final.follow_up_questions,
            "query_type": qtype,
            "search_strategy": strategy,
            "graph_used": bool(qtype in ("comparative", "relational") and paper_ids_in_subgraph),
            "papers_in_subgraph": len(paper_ids_in_subgraph),
            "passages_retrieved": len(passage_lookup),
        }
    except Exception as e:
        logger.error(f"Structured answer failed: {e}")
        err_str = str(e).lower()
        if any(k in err_str for k in ("429", "rate limit", "quota", "resource_exhausted")):
            return {
                "answer": "The AI API is currently rate limited. Please wait a moment and try again.",
                "sources": [], "confidence": 0.0, "follow_up_questions": [],
                "error": "rate_limited",
            }
        return {
            "answer": f"Query failed: {e}",
            "sources": [], "confidence": 0.0, "follow_up_questions": [],
        }


# ══════════════════════════════════════════════════════════════════════════════
# API REQUEST MODELS
# ══════════════════════════════════════════════════════════════════════════════


class QueryRequest(BaseModel):
    question: str
    top_k: int = 15


class NoteRequest(BaseModel):
    title: str
    content: str


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "PaperTrail API", "version": "2.0.0"}


def _ingest_pdf_bytes(file_bytes: bytes, source_label: str) -> dict:
    """Index a PDF from raw bytes. `source_label` is the filename or URL."""
    file_hash = hashlib.md5(file_bytes).hexdigest()[:12]
    paper_id = f"paper:{file_hash}"

    if paper_id in papers_db:
        return {"message": "Paper already indexed", "paper_id": paper_id, "title": papers_db[paper_id]["title"], "duplicate": True}

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
    entities["title"] = title

    add_to_knowledge_graph(paper_id, entities)

    page_chunks = chunk_pages(pages)
    add_to_vector_store(paper_id, page_chunks, {"title": title})

    papers_db[paper_id] = {
        "title": title,
        "filename": source_label,
        "pages": len(pages),
        "chunks": len(page_chunks),
        "entities": entities,
        "uploaded_at": datetime.now().isoformat(),
    }
    save_state()

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
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    file_bytes = await file.read()
    return _ingest_pdf_bytes(file_bytes, file.filename)


def _normalize_paper_url(url: str) -> str:
    """Turn arXiv abs URLs into PDF URLs; pass through everything else."""
    url = url.strip()
    m = re.match(r"https?://(?:www\.)?arxiv\.org/abs/([^?#\s]+)", url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
    if "arxiv.org/pdf/" in url and not url.lower().endswith(".pdf"):
        return url + ".pdf"
    return url


class UrlUploadRequest(BaseModel):
    url: str


@app.post("/upload-url")
async def upload_pdf_from_url(req: UrlUploadRequest):
    import httpx
    url = _normalize_paper_url(req.url)
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as h:
            resp = await h.get(url, headers={"User-Agent": "PaperTrail/1.0 (research memory agent)"})
            if resp.status_code != 200:
                raise HTTPException(400, f"Could not fetch URL ({resp.status_code})")
            ct = resp.headers.get("content-type", "")
            if "pdf" not in ct.lower() and not url.lower().endswith(".pdf"):
                raise HTTPException(400, f"URL did not return a PDF (content-type: {ct})")
            file_bytes = resp.content
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Fetch failed: {e}")
    return _ingest_pdf_bytes(file_bytes, url)


@app.post("/note")
async def add_note(note: NoteRequest):
    note_id = f"note:{uuid.uuid4().hex[:12]}"
    entities = extract_entities(note.content)
    add_to_knowledge_graph(note_id, {**entities, "title": note.title})

    chunks = chunk_text(note.content)
    add_to_vector_store(note_id, chunks, {"title": note.title})

    papers_db[note_id] = {
        "title": note.title,
        "type": "note",
        "chunks": len(chunks),
        "entities": entities,
        "uploaded_at": datetime.now().isoformat(),
    }
    save_state()
    return {"message": "Note added", "note_id": note_id, "title": note.title}


@app.post("/query")
async def query(req: QueryRequest):
    if collection.count() == 0 and len(kg.nodes) == 0:
        return {"answer": "Your library is empty. Upload some papers first!", "sources": []}
    return graphrag_query(req.question, req.top_k)


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
            for pid, pdata in papers_db.items()
        ],
        "total": len(papers_db),
    }


@app.get("/papers/{paper_id:path}")
def get_paper(paper_id: str):
    paper_id = paper_id.replace("%3A", ":")
    if paper_id not in papers_db:
        raise HTTPException(404, "Paper not found")
    return papers_db[paper_id]


@app.get("/graph")
def get_graph():
    nodes = [
        {"id": nid, "label": data.get("label", data.get("name", nid)), "type": data.get("type", "unknown")}
        for nid, data in kg.nodes(data=True)
    ]
    edges = [
        {"source": src, "target": tgt, "relation": data.get("relation", "related")}
        for src, tgt, data in kg.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges)}


@app.get("/stats")
def get_stats():
    node_types = {}
    for _, data in kg.nodes(data=True):
        t = data.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    return {
        "papers": len(papers_db),
        "graph_nodes": len(kg.nodes),
        "graph_edges": len(kg.edges),
        "vector_chunks": collection.count(),
        "node_types": node_types,
    }


@app.delete("/papers/{paper_id:path}")
def delete_paper(paper_id: str):
    paper_id = paper_id.replace("%3A", ":")
    if paper_id not in papers_db:
        raise HTTPException(404, "Paper not found")
    # Drop chunks
    try:
        collection.delete(where={"paper_id": paper_id})
        global _bm25_dirty
        _bm25_dirty = True
    except Exception as e:
        logger.warning(f"Chroma delete failed for {paper_id}: {e}")
    # Drop graph nodes that are exclusive to this paper (the paper node and any orphans)
    if kg.has_node(paper_id):
        kg.remove_node(paper_id)
    orphans = [n for n in list(kg.nodes) if kg.degree(n) == 0]
    for n in orphans:
        kg.remove_node(n)
    papers_db.pop(paper_id, None)
    save_state()
    return {"message": "Paper deleted", "paper_id": paper_id}


@app.delete("/reset")
def reset_system():
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
    global _bm25_index, _bm25_chunk_ids, _bm25_paper_ids, _bm25_dirty
    _bm25_index = None
    _bm25_chunk_ids = []
    _bm25_paper_ids = []
    _bm25_dirty = False
    save_state()
    return {"message": "System reset complete"}


# ── Serve built frontend (single-container deploy) ────────────────────────────
import pathlib
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

_FRONTEND_DIST = pathlib.Path(__file__).parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa_fallback(full_path: str):
        candidate = _FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
