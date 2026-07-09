"""GraphRAG query pipeline: classification, hybrid retrieval, grounded answer
generation, citation verification, and faithfulness checking."""
import json
import logging
import os
import re
from typing import Callable, Optional

from . import config, state
from .kgraph import traverse_knowledge_graph, _papers_in_subgraph
from .models import FaithfulnessReport, QueryAnswer, QueryClassification
from .retrieval import _search_chunks, search_vector_store

logger = logging.getLogger("papertrail")


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


def get_paper_details(paper_title: str) -> str:
    """Get paper metadata by title."""
    paper_title_lower = paper_title.lower()
    for pid, pdata in state.papers_db.items():
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


# ── Quote grounding ────────────────────────────────────────────────────────────
# Folding tables for characters PyMuPDF and the LLM disagree on:
#   smart quotes, various dashes, asterisk lookalikes, nonbreaking spaces, etc.
_QUOTE_FOLDS = [
    (re.compile(r"[‘’`´]"), "'"),
    (re.compile(r"[“”]"), '"'),
    (re.compile(r"[–—−‐-]"), "-"),
    (re.compile(r"[∗⋆★⁎*]"), " "),     # asterisks signal author footnotes — squash to space
    (re.compile(r"[†‡§¶]"), " "),       # footnote daggers
    (re.compile(r"[   ]"), " "),  # nonbreaking spaces
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
    if not config.client:
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
        return config._parse_structured(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=FaithfulnessReport,
            model=config.MODEL_FAST,
        )
    except Exception as e:
        logger.warning(f"Faithfulness check failed: {e}")
        return None


def graphrag_query(question: str, top_k: int = 5, progress: Optional[Callable[[str, dict], None]] = None) -> dict:
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

    `progress`, if given, is called with (stage, detail) as the pipeline
    advances — used by the SSE streaming endpoint to surface live status.
    """
    def _emit(stage: str, detail: dict = None):
        if progress:
            try:
                progress(stage, detail or {})
            except Exception:
                pass  # a broken progress listener must never break the query

    if not config.client:
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
    _emit("classifying")
    try:
        query_info = config._parse_structured(
            messages=[
                {"role": "system", "content": "Classify this research question to determine the best search strategy."},
                {"role": "user", "content": question},
            ],
            response_model=QueryClassification,
            model=config.MODEL_FAST,
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
        or len(state.papers_db) > 1
    )

    # ── Step 2: Deterministic retrieval ─────────────────────────────────
    _emit("retrieving", {"query_type": qtype, "strategy": strategy})
    graph_context = ""
    paper_ids_in_subgraph = []
    if use_graph and key_entities and len(state.kg.nodes) > 0:
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

    library_titles = [p.get("title", "") for p in state.papers_db.values()]
    library_listing = "\n".join(f"- {t}" for t in library_titles) or "(empty)"

    # ── Step 3: Generate cited answer ───────────────────────────────────
    logger.info("Generating structured answer over %d numbered passages...", len(passage_lookup))
    _emit("generating", {"passages": len(passage_lookup), "papers_in_subgraph": len(paper_ids_in_subgraph)})
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
        final = config._parse_structured(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=QueryAnswer,
            model=config.MODEL_QUALITY,
        )

        # Verify each citation against the actual passage text.
        _emit("verifying", {"citations": len(final.sources)})
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
        _emit("fact_checking")
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
