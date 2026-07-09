"""Unit tests for arXiv metadata parsing (no network)."""
from papertrail import api

ATOM_OK = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <title>Attention Is All You Need</title>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
</feed>
"""

ATOM_STUB = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id</id>
    <title>Error</title>
  </entry>
</feed>
"""


class TestArxivAtomParsing:
    def test_valid_entry(self):
        meta = api._parse_arxiv_atom(ATOM_OK)
        assert meta["title"] == "Attention Is All You Need"
        assert meta["authors"] == ["Ashish Vaswani", "Noam Shazeer"]

    def test_error_stub_returns_empty(self):
        assert api._parse_arxiv_atom(ATOM_STUB) == {}

    def test_garbage_returns_empty(self):
        assert api._parse_arxiv_atom("not xml at all") == {}
        assert api._parse_arxiv_atom("<feed/>") == {}


class TestArxivIdExtraction:
    def test_matches_abs_and_pdf_urls(self):
        assert api._ARXIV_ID_RE.search("https://arxiv.org/abs/1706.03762").group(1) == "1706.03762"
        assert api._ARXIV_ID_RE.search("https://arxiv.org/pdf/1706.03762.pdf").group(1) == "1706.03762"
        assert api._ARXIV_ID_RE.search("https://arxiv.org/pdf/2301.00001v2.pdf").group(1) == "2301.00001v2"

    def test_non_arxiv_urls_dont_match(self):
        assert api._ARXIV_ID_RE.search("https://example.com/paper.pdf") is None
