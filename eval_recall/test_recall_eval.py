"""
Unit tests for the Recall/MRR computation — mocks the retrieval call so the
metric math is checked in isolation (no embedding model needed).
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
