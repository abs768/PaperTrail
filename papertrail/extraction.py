"""LLM entity extraction: slicing, parallel structured extraction, merging,
and source-grounded validation."""
import logging
import os
import re
from typing import Optional

from . import config
from .kgraph import _canonicalize_entity, _ENTITY_ALIASES
from .models import PaperEntities
from .textproc import _strip_references_section

logger = logging.getLogger("papertrail")


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
      1. Slice the paper into overlapping windows (env-tunable count/size).
      2. Run structured extraction on each slice — IN PARALLEL via a thread pool.
         Each call is an independent LLM request, so wall time collapses to roughly
         the slowest single call instead of the sum.
      3. Merge — dedupe entities by canonical form.
      4. Validate — drop entities that don't literally appear in the source.
    """
    if not config.client:
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
            result = config._parse_structured(
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
                model=config.MODEL_FAST,
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
