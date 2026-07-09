"""PDF text extraction, title recovery, and text chunking."""
import re

import fitz  # PyMuPDF


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
