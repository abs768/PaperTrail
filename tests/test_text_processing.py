"""Unit tests for pure text-processing helpers: chunking, references
stripping, title sanity checks, and JSON salvage."""
from papertrail import api, config, textproc


class TestChunking:
    def test_chunk_text_empty(self):
        assert textproc.chunk_text("") == []

    def test_chunk_text_short_paragraph_is_one_chunk(self):
        text = "A short paragraph about transformers."
        chunks = textproc.chunk_text(text)
        assert chunks == [text]

    def test_chunks_respect_target_size(self):
        para = "This is a sentence with exactly eight words here. "
        text = "\n\n".join([para * 20] * 5)  # paragraphs of ~180 words each
        chunks = textproc.chunk_text(text, target_words=100, overlap_words=20)
        assert len(chunks) > 1
        # Chunks can exceed target only slightly due to sentence packing.
        assert all(len(c.split()) <= 160 for c in chunks)

    def test_giant_unbreakable_unit_hard_splits(self):
        text = "word " * 1000  # one paragraph, no sentence boundaries
        chunks = textproc.chunk_text(text, target_words=100, overlap_words=10)
        assert len(chunks) >= 10
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_overlap_carries_tail_of_previous_chunk(self):
        sent = "Distinct sentence number %d with several extra words attached. "
        text = "".join(sent % i for i in range(60))
        chunks = textproc.chunk_text(text, target_words=80, overlap_words=15)
        assert len(chunks) >= 2
        tail_words = chunks[0].split()[-5:]
        assert " ".join(tail_words) in chunks[1]

    def test_chunk_pages_stamps_page_numbers(self):
        pages = [
            {"page": 1, "text": "First page text. " * 30},
            {"page": 7, "text": "Seventh page text. " * 30},
        ]
        out = textproc.chunk_pages(pages, target_words=50, overlap_words=5)
        assert {c["page"] for c in out} == {1, 7}
        assert all(c["text"].strip() for c in out)


class TestReferencesStripping:
    def test_strips_references_in_latter_half(self):
        body = "Introduction and methods. " * 200
        refs = "References\n[1] Some cited paper. " * 20
        text = body + "\nReferences\n" + refs
        stripped = textproc._strip_references_section(text)
        assert "[1] Some cited paper" not in stripped
        assert "Introduction and methods" in stripped

    def test_ignores_references_word_in_first_half(self):
        text = ("We discuss references in prose here. " * 100
                + "Later body content. " * 200)
        assert textproc._strip_references_section(text) == text

    def test_short_text_untouched(self):
        text = "Tiny doc.\nReferences\n[1] x"
        assert textproc._strip_references_section(text) == text


class TestTitleHelpers:
    def test_junk_titles_rejected(self):
        assert not textproc._looks_like_real_title("untitled")
        assert not textproc._looks_like_real_title("Microsoft Word - draft.docx")
        assert not textproc._looks_like_real_title("doc.tex")
        assert not textproc._looks_like_real_title("ab")
        assert not textproc._looks_like_real_title("x" * 300)

    def test_real_title_accepted(self):
        assert textproc._looks_like_real_title("Attention Is All You Need")

    def test_llm_title_grounding(self):
        text = "Attention Is All You Need\nAshish Vaswani et al.\nAbstract..."
        assert textproc._llm_title_is_grounded("Attention Is All You Need", text)
        assert textproc._llm_title_is_grounded("attention is all you need.", text)
        assert not textproc._llm_title_is_grounded("Transformers: A Survey", text)
        assert not textproc._llm_title_is_grounded("", text)


class TestJsonSalvage:
    def test_plain_object(self):
        assert config._extract_json_object('{"a": 1}') == '{"a": 1}'

    def test_fenced_object(self):
        raw = 'Here you go:\n```json\n{"a": 1}\n```\nDone.'
        assert config._extract_json_object(raw) == '{"a": 1}'

    def test_preamble_and_trailing_prose(self):
        raw = 'Sure! {"a": {"b": 2}} hope that helps'
        assert config._extract_json_object(raw) == '{"a": {"b": 2}}'

    def test_empty(self):
        assert config._extract_json_object("") == "{}"


class TestUrlNormalization:
    def test_arxiv_abs_to_pdf(self):
        assert (api._normalize_paper_url("https://arxiv.org/abs/1706.03762")
                == "https://arxiv.org/pdf/1706.03762.pdf")

    def test_arxiv_pdf_missing_extension(self):
        assert (api._normalize_paper_url("https://arxiv.org/pdf/1706.03762")
                == "https://arxiv.org/pdf/1706.03762.pdf")

    def test_other_urls_pass_through(self):
        url = "https://example.com/paper.pdf"
        assert api._normalize_paper_url(url) == url
