"""Tests for loading a library snapshot back in — the counterpart to /export."""
import json

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from papertrail import api, seed, state


@pytest.fixture()
def client():
    return TestClient(api.app)


@pytest.fixture()
def empty_library(monkeypatch):
    """An empty in-memory library, so a load has somewhere to land."""
    monkeypatch.setattr(state, "kg", nx.DiGraph())
    monkeypatch.setattr(state, "papers_db", {})
    monkeypatch.setattr(state, "save_state", lambda: None)


SNAPSHOT = {
    "exported_at": "2026-01-01T00:00:00",
    "paper_count": 1,
    "papers": {"paper:x": {"title": "X", "chunks": 2, "entities": {}}},
    "graph": nx.node_link_data(
        (lambda g: (
            g.add_node("paper:x", type="paper", title="X", label="X"),
            g.add_node("method:transformer", type="method", name="transformer",
                       label="transformer"),
            g.add_edge("paper:x", "method:transformer", relation="proposes"),
            g,
        )[-1])(nx.DiGraph())
    ),
    "chunks": [
        {"id": "paper:x_chunk_0", "text": "The transformer uses self-attention.",
         "metadata": {"paper_id": "paper:x", "title": "X", "chunk_index": 0}},
        {"id": "paper:x_chunk_1", "text": "It removes recurrence entirely.",
         "metadata": {"paper_id": "paper:x", "title": "X", "chunk_index": 1}},
    ],
}


class TestLoadSnapshot:
    def test_restores_papers_and_graph(self, empty_library):
        summary = seed.load_snapshot(dict(SNAPSHOT))
        assert summary["papers"] == 1
        assert summary["graph_nodes"] == 2
        assert summary["graph_edges"] == 1
        assert state.papers_db["paper:x"]["title"] == "X"
        assert state.kg.edges[("paper:x", "method:transformer")]["relation"] == "proposes"

    def test_reads_from_a_file(self, tmp_path, empty_library):
        p = tmp_path / "library.json"
        p.write_text(json.dumps(SNAPSHOT))
        assert seed.load_snapshot(p)["papers"] == 1

    def test_refuses_to_clobber_a_non_empty_library(self, empty_library):
        state.papers_db["paper:existing"] = {"title": "keep me"}
        with pytest.raises(RuntimeError, match="already holds"):
            seed.load_snapshot(dict(SNAPSHOT))
        assert "paper:existing" in state.papers_db

    def test_missing_file_raises(self, tmp_path, empty_library):
        with pytest.raises(FileNotFoundError):
            seed.load_snapshot(tmp_path / "nope.json")


class TestSeedIfEmpty:
    def test_skips_when_library_has_papers(self, tmp_path, empty_library):
        p = tmp_path / "library.json"
        p.write_text(json.dumps(SNAPSHOT))
        state.papers_db["paper:existing"] = {"title": "keep me"}
        assert seed.seed_if_empty(p) is None

    def test_skips_when_snapshot_absent(self, tmp_path, empty_library):
        assert seed.seed_if_empty(tmp_path / "nope.json") is None

    def test_loads_into_empty_library(self, tmp_path, empty_library):
        p = tmp_path / "library.json"
        p.write_text(json.dumps(SNAPSHOT))
        assert seed.seed_if_empty(p)["papers"] == 1


class TestDemoModeGuards:
    """In demo mode the library is fixed: mutations must be refused."""

    @pytest.fixture()
    def demo(self, monkeypatch):
        monkeypatch.setattr(api, "DEMO_MODE", True)

    def test_health_reports_capabilities(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert "demo_mode" in body and "llm_enabled" in body

    def test_note_rejected(self, client, demo):
        r = client.post("/note", json={"title": "t", "content": "c"})
        assert r.status_code == 403
        assert "read-only" in r.json()["detail"]

    def test_upload_url_rejected(self, client, demo):
        r = client.post("/upload-url", json={"url": "https://arxiv.org/abs/1706.03762"})
        assert r.status_code == 403

    def test_reset_rejected(self, client, demo):
        assert client.delete("/reset").status_code == 403

    def test_delete_paper_rejected(self, client, demo):
        assert client.delete("/papers/paper:x").status_code == 403

    def test_reads_still_allowed(self, client, demo):
        assert client.get("/papers").status_code == 200
        assert client.get("/stats").status_code == 200
