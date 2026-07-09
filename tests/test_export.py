"""Tests for the GET /export library backup endpoint."""
import networkx as nx
import pytest
from fastapi.testclient import TestClient

from papertrail import api, state


@pytest.fixture()
def client():
    return TestClient(api.app)


@pytest.fixture()
def seeded(monkeypatch):
    g = nx.DiGraph()
    g.add_node("paper:x", type="paper", title="X", label="X")
    g.add_node("method:m", type="method", name="m", label="m")
    g.add_edge("paper:x", "method:m", relation="proposes")
    monkeypatch.setattr(state, "kg", g)
    monkeypatch.setattr(state, "papers_db", {"paper:x": {"title": "X", "chunks": 0, "entities": {}}})


class TestExport:
    def test_export_shape(self, client, seeded):
        body = client.get("/export").json()
        assert body["paper_count"] == 1
        assert "paper:x" in body["papers"]
        assert "exported_at" in body
        assert "chunks" not in body  # opt-in only

    def test_graph_round_trips_through_node_link(self, client, seeded):
        body = client.get("/export").json()
        g = nx.node_link_graph(body["graph"])
        assert g.is_directed()
        assert set(g.nodes) == {"paper:x", "method:m"}
        assert g.edges[("paper:x", "method:m")]["relation"] == "proposes"

    def test_include_chunks_adds_chunk_list(self, client, seeded):
        body = client.get("/export", params={"include_chunks": "true"}).json()
        assert isinstance(body.get("chunks"), list)
