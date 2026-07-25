"""Deleting a paper must only remove that paper's own graph nodes.

The orphan sweep used to scan the whole graph for degree-0 nodes, which caught
unrelated library items: a paper or note whose extraction produced no entities
sits at degree 0 from the moment it is indexed, so deleting any *other* paper
silently dropped it from the graph while papers_db kept listing it.
"""
import networkx as nx
import pytest
from fastapi.testclient import TestClient

from papertrail import api, state


@pytest.fixture()
def client():
    return TestClient(api.app)


@pytest.fixture()
def two_papers(monkeypatch):
    """Paper A owns an entity; paper B has none (degree 0), like a failed extraction."""
    kg = nx.DiGraph()
    kg.add_node("paper:aaa", type="paper", title="A")
    kg.add_node("method:attention", type="method", name="attention")
    kg.add_edge("paper:aaa", "method:attention", relation="proposes")
    kg.add_node("paper:bbb", type="paper", title="B")
    monkeypatch.setattr(state, "kg", kg)
    monkeypatch.setattr(
        state, "papers_db", {"paper:aaa": {"title": "A"}, "paper:bbb": {"title": "B"}}
    )
    monkeypatch.setattr(state, "save_state", lambda: None)
    monkeypatch.setattr(api, "_ADMIN_TOKEN", "")
    return kg


class TestDeleteOrphanSweep:
    def test_unrelated_entityless_paper_survives(self, client, two_papers):
        assert client.delete("/papers/paper:aaa").status_code == 200
        assert two_papers.has_node("paper:bbb"), "unrelated paper was swept from the graph"
        assert "paper:bbb" in state.papers_db

    def test_exclusive_entity_is_removed(self, client, two_papers):
        client.delete("/papers/paper:aaa")
        assert not two_papers.has_node("method:attention")
        assert not two_papers.has_node("paper:aaa")

    def test_shared_entity_survives(self, client, monkeypatch):
        kg = nx.DiGraph()
        for pid in ("paper:aaa", "paper:ccc"):
            kg.add_node(pid, type="paper", title=pid)
            kg.add_edge(pid, "method:attention", relation="proposes")
        kg.nodes["method:attention"]["type"] = "method"
        monkeypatch.setattr(state, "kg", kg)
        monkeypatch.setattr(
            state, "papers_db", {"paper:aaa": {"title": "A"}, "paper:ccc": {"title": "C"}}
        )
        monkeypatch.setattr(state, "save_state", lambda: None)
        monkeypatch.setattr(api, "_ADMIN_TOKEN", "")

        client.delete("/papers/paper:aaa")
        assert kg.has_node("method:attention"), "entity still cited by paper:ccc was removed"
