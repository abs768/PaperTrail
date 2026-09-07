"""Notes must participate in graph-balanced retrieval alongside PDFs.

Notes are ingested as "note:<uuid>" while papers are "paper:<hash>", but both
are library items that own chunks in the vector store. _papers_in_subgraph
feeds the per-paper retrieval branch for comparative/relational queries, so a
note missing from its result is a note whose chunks are never retrieved there.
"""
import json

from papertrail.kgraph import _papers_in_subgraph


def _subgraph(*ids):
    return json.dumps({"results": [{"id": i, "type": "paper"} for i in ids]})


class TestPapersInSubgraph:
    def test_includes_notes_alongside_papers(self):
        found = _papers_in_subgraph(_subgraph("paper:abc123", "note:def456"))
        assert set(found) == {"paper:abc123", "note:def456"}

    def test_note_only_subgraph_is_not_empty(self):
        assert _papers_in_subgraph(_subgraph("note:def456")) == ["note:def456"]

    def test_entity_nodes_are_excluded(self):
        found = _papers_in_subgraph(
            _subgraph("paper:abc123", "method:attention", "author:vaswani", "concept:nmt")
        )
        assert found == ["paper:abc123"]

    def test_malformed_json_returns_empty(self):
        assert _papers_in_subgraph("not json at all") == []
