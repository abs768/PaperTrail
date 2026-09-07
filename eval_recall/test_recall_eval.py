"""
Unit tests for the Recall/MRR computation and the configuration ablation —
mocks the retrieval call so the metric math is checked in isolation (no
embedding model needed).
"""
import eval_recall.recall_eval as R


def _fake_search(ranked_ids):
    """Return a stand-in _search_chunks that yields a fixed ranking."""
    def _search(query, top_k=10):
        return [{"chunk_id": cid} for cid in ranked_ids[:top_k]]
    return _search


def test_recall_and_mrr_perfect(monkeypatch):
    from papertrail import retrieval
    monkeypatch.setattr(retrieval, "_search_chunks", _fake_search(["g", "a", "b"]))
    q = [{"query": "x", "gold_chunk_id": "g", "type": "t"}]
    m = R.evaluate(q, ks=(1, 5))
    assert m["recall_at"][1] == 1.0
    assert m["mrr"] == 1.0


def test_recall_at_rank_three(monkeypatch):
    from papertrail import retrieval
    monkeypatch.setattr(retrieval, "_search_chunks", _fake_search(["a", "b", "g", "c"]))
    q = [{"query": "x", "gold_chunk_id": "g", "type": "t"}]
    m = R.evaluate(q, ks=(1, 5))
    assert m["recall_at"][1] == 0.0     # gold not at rank 1
    assert m["recall_at"][5] == 1.0     # but within top 5
    assert abs(m["mrr"] - 1 / 3) < 1e-9


def test_miss_gives_zero(monkeypatch):
    from papertrail import retrieval
    monkeypatch.setattr(retrieval, "_search_chunks", _fake_search(["a", "b", "c"]))
    q = [{"query": "x", "gold_chunk_id": "g", "type": "t"}]
    m = R.evaluate(q, ks=(1, 5))
    assert m["recall_at"][5] == 0.0
    assert m["mrr"] == 0.0


def test_averages_over_queries(monkeypatch):
    from papertrail import retrieval
    calls = {"n": 0}

    def _search(query, top_k=10):
        calls["n"] += 1
        # first query: gold at rank 1; second: gold missing
        return ([{"chunk_id": "g"}] if calls["n"] == 1 else [{"chunk_id": "z"}])

    monkeypatch.setattr(retrieval, "_search_chunks", _search)
    q = [{"query": "a", "gold_chunk_id": "g", "type": "t"},
         {"query": "b", "gold_chunk_id": "g", "type": "t"}]
    m = R.evaluate(q, ks=(1,))
    assert m["recall_at"][1] == 0.5


# ── Ablation ──────────────────────────────────────────────────────────────────


def test_evaluate_reports_latency(monkeypatch):
    from papertrail import retrieval
    monkeypatch.setattr(retrieval, "_search_chunks", _fake_search(["g"]))
    m = R.evaluate([{"query": "x", "gold_chunk_id": "g"}], ks=(1,))
    assert m["median_latency_ms"] >= 0.0


def test_explicit_retriever_overrides_default(monkeypatch):
    """A supplied retriever must be used instead of _search_chunks."""
    from papertrail import retrieval
    monkeypatch.setattr(retrieval, "_search_chunks", _fake_search(["wrong"]))
    m = R.evaluate([{"query": "x", "gold_chunk_id": "g"}], ks=(1,),
                   retriever=lambda q, k: [{"chunk_id": "g"}])
    assert m["recall_at"][1] == 1.0


def test_dense_lane_calls_vector_search_only(monkeypatch):
    """The dense lane must bypass BM25 and RRF entirely."""
    from papertrail import retrieval
    seen = {}
    monkeypatch.setattr(retrieval, "_vector_search_raw",
                        lambda q, k, paper_ids=None: seen.setdefault("dense", True) and [])
    monkeypatch.setattr(retrieval, "_bm25_search",
                        lambda *a, **k: seen.setdefault("bm25", True) and [])
    monkeypatch.setattr(retrieval, "_search_chunks",
                        lambda *a, **k: seen.setdefault("hybrid", True) and [])
    R.make_retriever("dense")("q", 10)
    assert seen == {"dense": True}


def test_bm25_lane_preserves_bm25_ranking(monkeypatch):
    """Hydration returns a dict; the lane must restore BM25's own order."""
    from papertrail import retrieval
    monkeypatch.setattr(retrieval, "_bm25_search",
                        lambda q, k, paper_ids=None: [("c2", 9.0), ("c1", 4.0)])
    monkeypatch.setattr(retrieval, "_hydrate_chunks",
                        lambda ids: {"c1": {"chunk_id": "c1"}, "c2": {"chunk_id": "c2"}})
    out = R.make_retriever("bm25")("q", 10)
    assert [c["chunk_id"] for c in out] == ["c2", "c1"]


def test_bm25_lane_drops_unhydratable_ids(monkeypatch):
    from papertrail import retrieval
    monkeypatch.setattr(retrieval, "_bm25_search",
                        lambda q, k, paper_ids=None: [("gone", 9.0), ("c1", 4.0)])
    monkeypatch.setattr(retrieval, "_hydrate_chunks",
                        lambda ids: {"c1": {"chunk_id": "c1"}})
    assert [c["chunk_id"] for c in R.make_retriever("bm25")("q", 10)] == ["c1"]


def test_unknown_config_rejected():
    import pytest
    with pytest.raises(ValueError):
        R.make_retriever("nonsense")


def test_rerank_config_unavailable_yields_no_numbers(monkeypatch):
    """A reranker that fails to load must produce a blank row, not hybrid's numbers.

    _rerank_chunks silently falls back to the fused order when the model is
    missing, so without this guard the ablation would republish hybrid's results
    under the rerank label — a fabricated comparison.
    """
    from papertrail import retrieval
    monkeypatch.setattr(retrieval, "_get_reranker", lambda: None)
    monkeypatch.setattr(retrieval, "_search_chunks", _fake_search(["g"]))
    metrics, reason = R.run_config(
        "hybrid+rerank", [{"query": "x", "gold_chunk_id": "g"}]
    )
    assert metrics is None
    assert "sentence-transformers" in reason


def test_render_leaves_blank_cells_for_unavailable_config():
    results = {
        "hybrid": ({"recall_at": {1: 1.0, 5: 1.0, 10: 1.0}, "mrr": 1.0,
                    "recall5_by_type": {"explicit": 1.0},
                    "median_latency_ms": 5.0, "n_queries": 1}, ""),
        "hybrid+rerank": (None, "cross-encoder did not load"),
    }
    out = R.render(results, n_chunks=10, n_papers=2, n_queries=1)
    assert "cross-encoder did not load" in out
    assert "Not measured" in out
    # The unavailable row must carry no digits at all.
    rerank_row = next(l for l in out.splitlines()
                      if l.startswith("| " + R.CONFIG_LABELS["hybrid+rerank"]))
    assert not any(ch.isdigit() for ch in rerank_row)


def test_render_is_deterministic_by_default():
    """Latency must stay out of the report unless explicitly asked for."""
    results = {"hybrid": ({"recall_at": {1: 0.5}, "mrr": 0.5,
                           "recall5_by_type": {}, "median_latency_ms": 123.456,
                           "n_queries": 2}, "")}
    default = R.render(results, 10, 2, 2, ks=(1,))
    with_lat = R.render(results, 10, 2, 2, ks=(1,), deterministic=False)
    assert "123" not in default
    assert "123 ms" in with_lat
