"""Tests for the /graph endpoint's node-limit parameter."""
import networkx as nx
import pytest
from fastapi.testclient import TestClient

from papertrail import api, state


@pytest.fixture()
def client():
    return TestClient(api.app)


@pytest.fixture()
def seeded_graph(monkeypatch):
    """A small graph: 2 papers + 10 entities of varying degree."""
    g = nx.DiGraph()
    g.add_node("paper:a", type="paper", title="Paper A", label="Paper A")
    g.add_node("paper:b", type="paper", title="Paper B", label="Paper B")
    for i in range(10):
        nid = f"concept:c{i}"
        g.add_node(nid, type="concept", name=f"c{i}", label=f"c{i}")
        g.add_edge("paper:a", nid, relation="discusses")
        if i < 3:  # c0-c2 are shared → higher degree
            g.add_edge("paper:b", nid, relation="discusses")
    monkeypatch.setattr(state, "kg", g)
    return g


class TestGraphLimit:
    def test_no_limit_returns_everything(self, client, seeded_graph):
        body = client.get("/graph").json()
        assert body["node_count"] == 12
        assert body["truncated"] is False

    def test_limit_keeps_papers_and_top_degree_entities(self, client, seeded_graph):
        body = client.get("/graph", params={"limit": 5}).json()
        ids = {n["id"] for n in body["nodes"]}
        assert {"paper:a", "paper:b"} <= ids  # papers always survive
        # The 3 highest-degree concepts (shared between both papers) fill the budget.
        assert {"concept:c0", "concept:c1", "concept:c2"} <= ids
        assert body["node_count"] == 5
        assert body["truncated"] is True
        assert body["total_nodes"] == 12

    def test_edges_to_dropped_nodes_are_omitted(self, client, seeded_graph):
        body = client.get("/graph", params={"limit": 5}).json()
        ids = {n["id"] for n in body["nodes"]}
        for e in body["edges"]:
            assert e["source"] in ids and e["target"] in ids

    def test_limit_larger_than_graph_is_noop(self, client, seeded_graph):
        body = client.get("/graph", params={"limit": 500}).json()
        assert body["node_count"] == 12
        assert body["truncated"] is False
