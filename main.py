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
    paper_title: str = Field(description="Title of the source paper — MUST exactly match a paper in the supplied context")
    relevant_detail: str = Field(description="Specific detail used from this paper")
    page: Optional[int] = Field(default=None, description="Page number where the detail was found, if known")


class QueryAnswer(BaseModel):
    """Structured answer to a user's question."""

    answer: str = Field(description="Clear, comprehensive answer to the question")
    sources: list[CitedSource] = Field(
        description="Papers cited in the answer with specific details"
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


def search_vector_store(query: str, top_k: int = 5, paper_ids: list = None) -> str:
    """Search ChromaDB for relevant chunks. Optionally restrict to specific paper IDs."""
    if collection.count() == 0:
        return json.dumps({"results": [], "message": "No papers indexed yet."})

    where = None
    if paper_ids:
        where = {"paper_id": {"$in": list(paper_ids)}} if len(paper_ids) > 1 else {"paper_id": paper_ids[0]}

    kwargs = {"query_texts": [query], "n_results": min(top_k, collection.count())}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    matches = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i] or {}
        matches.append(
            {
                "text": results["documents"][0][i][:600],
                "paper_title": meta.get("title", "Unknown"),
                "page": meta.get("page"),
                "distance": round(results["distances"][0][i], 4)
                if results.get("distances")
                else None,
            }
        )
    return json.dumps({"results": matches})


def traverse_knowledge_graph(entities: list[str], hops: int = 2) -> str:
    """Traverse the knowledge graph from given entities."""
    relevant_info = []

    for entity in entities:
        entity_lower = entity.lower().strip()
        for node_id, data in kg.nodes(data=True):
            node_name = data.get("name", data.get("title", "")).lower()
            if entity_lower in node_name or node_name in entity_lower:
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


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_pages(pages: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Chunk per-page so we can keep page-level provenance on every chunk."""
    out = []
    for p in pages:
        words = p["text"].split()
        if not words:
            continue
        for i in range(0, len(words), chunk_size - overlap):
            text = " ".join(words[i : i + chunk_size])
            if text:
                out.append({"text": text, "page": p["page"]})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ENTITY EXTRACTION (Structured Outputs)
# ══════════════════════════════════════════════════════════════════════════════


def extract_entities(text: str) -> dict:
    """
    Use GPT-4o with Pydantic structured output to extract entities.
    Pattern: Structured Outputs (response_format=PaperEntities)
    """
    if not client:
        logger.warning("No OpenAI key — returning empty extraction")
        return _empty_extraction()

    try:
        logger.info("Extracting entities via structured JSON output...")
        # Two-pass extraction over the whole paper, not just the abstract.
        head = text[:6000]
        tail = text[-6000:] if len(text) > 12000 else ""
        body = head if not tail else f"{head}\n\n[…middle elided…]\n\n{tail}"
        result = _parse_structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert academic entity extractor. "
                        "Given text from a research paper, extract all structured entities and relationships. "
                        "Be thorough — extract every author, method, dataset, metric, and concept you can find. "
                        "For relationships, identify how entities relate to each other "
                        "(e.g., a method 'uses' a dataset, a paper 'proposes' an algorithm, a model 'outperforms' a baseline)."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Extract all entities and relationships from this research paper text:\n\n{body}",
                },
            ],
            response_model=PaperEntities,
            model=MODEL_FAST,
        )
        logger.info(
            f"Extraction complete — {len(result.authors)} authors, {len(result.methods)} methods, "
            f"{len(result.datasets)} datasets, {len(result.key_concepts)} concepts, "
            f"{len(result.relationships)} relationships"
        )
        return result.model_dump()

    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return _empty_extraction()


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


def add_to_knowledge_graph(paper_id: str, entities: dict):
    paper_title = entities.get("title") or paper_id
    kg.add_node(paper_id, type="paper", title=paper_title, label=paper_title)

    for author in entities.get("authors", []):
        aid = f"author:{author.lower().strip()}"
        kg.add_node(aid, type="author", name=author, label=author)
        kg.add_edge(aid, paper_id, relation="authored")

    for method in entities.get("methods", []):
        mid = f"method:{method.lower().strip()}"
        kg.add_node(mid, type="method", name=method, label=method)
        kg.add_edge(paper_id, mid, relation="proposes")

    for dataset in entities.get("datasets", []):
        did = f"dataset:{dataset.lower().strip()}"
        kg.add_node(did, type="dataset", name=dataset, label=dataset)
        kg.add_edge(paper_id, did, relation="evaluates_on")

    for metric in entities.get("metrics", []):
        if isinstance(metric, dict):
            m_name = metric.get("name", "unknown")
            m_val = metric.get("value", "")
        else:
            m_name, m_val = str(metric), ""
        m_id = f"metric:{m_name.lower().strip()}"
        kg.add_node(m_id, type="metric", name=m_name, value=m_val, label=m_name)
        kg.add_edge(paper_id, m_id, relation="reports")

    for concept in entities.get("key_concepts", []):
        cid = f"concept:{concept.lower().strip()}"
        kg.add_node(cid, type="concept", name=concept, label=concept)
        kg.add_edge(paper_id, cid, relation="discusses")

    for rel in entities.get("relationships", []):
        src = rel.get("source", "").lower().strip()
        tgt = rel.get("target", "").lower().strip()
        relation = rel.get("relation", "related_to")
        if src and tgt:
            src_id = f"entity:{src}"
            tgt_id = f"entity:{tgt}"
            if not kg.has_node(src_id):
                kg.add_node(src_id, type="entity", name=rel["source"], label=rel["source"])
            if not kg.has_node(tgt_id):
                kg.add_node(tgt_id, type="entity", name=rel["target"], label=rel["target"])
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


def _filter_hallucinated_sources(sources: list, library_titles: list[str]) -> tuple[list, list]:
    """Drop sources whose paper_title doesn't fuzzy-match any indexed paper.

    Returns (kept, dropped) where dropped is a list of {paper_title, reason}.
    """
    kept, dropped = [], []
    library_lower = [(t or "").lower() for t in library_titles]
    for s in sources:
        cited = (s.get("paper_title") or "").lower().strip()
        if not cited:
            dropped.append({"paper_title": s.get("paper_title"), "reason": "empty title"})
            continue
        match = any(
            cited == lt
            or (cited and lt and (cited in lt or lt in cited))
            or (cited and lt and len(cited) > 8 and len(lt) > 8 and (cited[:20] in lt or lt[:20] in cited))
            for lt in library_lower
        )
        if match:
            kept.append(s)
        else:
            dropped.append({"paper_title": s.get("paper_title"), "reason": "not in library"})
    return kept, dropped


def graphrag_query(question: str, top_k: int = 5) -> dict:
    """
    GraphRAG query pipeline (deterministic, graph-driven):
      1. Classify query → query_type, key_entities, search_strategy
      2. Pre-retrieve graph subgraph + vector chunks based on classification
         - For comparative/relational queries, vector search is restricted to
           papers found in the graph subgraph
      3. Single structured-output call generates an answer that MUST cite only
         the supplied context (no training-data leakage)
      4. Post-filter: drop any cited paper that isn't actually in the library
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
        graph_context = f"KNOWLEDGE GRAPH SUBGRAPH (entities + relationships from your library):\n{graph_raw}"

    # For comparative queries, restrict vector search to papers found in the subgraph.
    restrict_to = paper_ids_in_subgraph if (qtype in ("comparative", "relational") and paper_ids_in_subgraph) else None
    vector_raw = search_vector_store(question, top_k=top_k, paper_ids=restrict_to)
    vector_context = f"VECTOR PASSAGES{f' (restricted to {len(restrict_to)} papers from subgraph)' if restrict_to else ''}:\n{vector_raw}"

    library_titles = [p.get("title", "") for p in papers_db.values()]
    library_listing = "\n".join(f"- {t}" for t in library_titles) or "(empty)"

    # ── Step 3: Generate cited answer ───────────────────────────────────
    logger.info("Generating structured answer...")
    sys_prompt = (
        "You are PaperTrail, a research memory assistant. "
        "Answer the user's question using ONLY the supplied KNOWLEDGE GRAPH SUBGRAPH and VECTOR PASSAGES below. "
        "DO NOT cite any paper that is not in the user's library listing. "
        "If the supplied context is insufficient, say so plainly — do not fall back to your training data. "
        "When citing, the paper_title field MUST exactly match a title from the user's library listing. "
        "Include a page number on every CitedSource when the source passage has one."
    )
    user_prompt = (
        f"User's library ({len(library_titles)} papers):\n{library_listing}\n\n"
        f"{graph_context}\n\n{vector_context}\n\n"
        f"Question: {question}\n\n"
        f"Provide a clear, comprehensive answer grounded strictly in the context above. "
        f"Cite each claim with the exact paper_title from the library listing and the page number when available."
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

        sources = [s.model_dump() for s in final.sources]
        kept, dropped = _filter_hallucinated_sources(sources, library_titles)
        if dropped:
            logger.warning(f"Dropped {len(dropped)} hallucinated source(s): {[d['paper_title'] for d in dropped]}")

        logger.info(f"Answer generated — confidence: {final.confidence}, sources kept: {len(kept)}, dropped: {len(dropped)}")
        return {
            "answer": final.answer,
            "sources": kept,
            "dropped_sources": dropped,
            "confidence": final.confidence,
            "follow_up_questions": final.follow_up_questions,
            "query_type": qtype,
            "search_strategy": strategy,
            "graph_used": bool(restrict_to),
            "papers_in_subgraph": len(paper_ids_in_subgraph),
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

    fallback_title = source_label.rsplit("/", 1)[-1].replace(".pdf", "")
    title = entities.get("title") or fallback_title
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
