"""API-level tests via FastAPI's TestClient (no LLM key, no network)."""
import pytest
from fastapi.testclient import TestClient

from papertrail import api


@pytest.fixture()
def client():
    return TestClient(api.app)


class TestHealthAndStats:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_stats_shape(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        body = r.json()
        for key in ("papers", "graph_nodes", "graph_edges", "vector_chunks"):
            assert key in body

    def test_papers_listing(self, client):
        r = client.get("/papers")
        assert r.status_code == 200
        assert "papers" in r.json()

    def test_graph_shape(self, client):
        r = client.get("/graph")
        assert r.status_code == 200
        body = r.json()
        assert "nodes" in body and "edges" in body


class TestAdminToken:
    def test_reset_blocked_without_token(self, client, monkeypatch):
        monkeypatch.setattr(api, "_ADMIN_TOKEN", "sekrit")
        assert client.delete("/reset").status_code == 403

    def test_reset_allowed_with_token(self, client, monkeypatch):
        monkeypatch.setattr(api, "_ADMIN_TOKEN", "sekrit")
        r = client.delete("/reset", headers={"X-Admin-Token": "sekrit"})
        assert r.status_code == 200

    def test_reset_open_when_unset(self, client, monkeypatch):
        monkeypatch.setattr(api, "_ADMIN_TOKEN", "")
        assert client.delete("/reset").status_code == 200

    def test_delete_paper_blocked_without_token(self, client, monkeypatch):
        monkeypatch.setattr(api, "_ADMIN_TOKEN", "sekrit")
        assert client.delete("/papers/paper:abc").status_code == 403


class TestUploadValidation:
    def test_non_pdf_rejected(self, client):
        r = client.post("/upload", files={"file": ("notes.txt", b"hi", "text/plain")})
        assert r.status_code == 400

    def test_oversized_pdf_rejected(self, client, monkeypatch):
        monkeypatch.setattr(api, "_MAX_UPLOAD_BYTES", 10)
        r = client.post(
            "/upload", files={"file": ("big.pdf", b"x" * 100, "application/pdf")}
        )
        assert r.status_code == 413


class TestSsrfGuard:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/internal.pdf",
            "http://localhost:8000/x.pdf",
            "http://169.254.169.254/latest/meta-data",
            "ftp://example.com/x.pdf",
            "file:///etc/passwd",
        ],
    )
    def test_private_or_bad_scheme_rejected(self, client, url):
        r = client.post("/upload-url", json={"url": url})
        assert r.status_code == 400

    def test_query_on_empty_library(self, client):
        r = client.post("/query", json={"question": "anything?"})
        assert r.status_code == 200
        assert "sources" in r.json()


_needs_build = pytest.mark.skipif(
    not api._FRONTEND_DIST.is_dir(),
    reason="needs a built frontend/dist: the SPA route only mounts when it exists",
)


class TestSpaFallbackTraversal:
    """The catch-all SPA route must never serve files outside the dist root,
    even when the traversal is percent-encoded to survive URL normalization.
    Regression for an arbitrary-file-read that exposed .env / source / etc.

    `_safe_dist_file` is defined unconditionally in api.py, so the escape check
    below runs everywhere — including on a checkout with no frontend build. Only
    the two tests that need real files on disk (an actual index.html, and the
    mounted route) carry `_needs_build`. CI builds the frontend for the backend
    job precisely so those two run there rather than silently skipping."""

    @_needs_build
    @pytest.mark.parametrize(
        "payload",
        [
            "/%2e%2e%2f%2e%2e%2f.env",
            "/%2e%2e%2f%2e%2e%2fpapertrail%2fconfig.py",
            "/%2e%2e%2f%2e%2e%2fmain.py",
            "/%2e%2e%2f%2e%2e%2f%2e%2e%2f%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "/..%2f..%2fmain.py",
        ],
    )
    def test_traversal_does_not_escape_dist(self, client, payload):
        r = client.get(payload)
        # It should fall back to index.html, never leak the targeted file.
        assert r.status_code == 200
        body = r.text
        assert "root:x:0:0" not in body          # /etc/passwd
        assert "PaperTrail entrypoint" not in body  # main.py docstring
        assert "GROQ" not in body and "API_KEY" not in body  # .env / config
        assert "def _spa_fallback" not in body   # api.py source

    def test_safe_dist_file_rejects_escape(self):
        # No build needed: an escaping path resolves outside the dist root and
        # is rejected on that basis alone, so this is the one security assertion
        # that holds — and runs — on any checkout.
        assert api._safe_dist_file("../../main.py") is None
        assert api._safe_dist_file("../../.env") is None
        assert api._safe_dist_file("../../papertrail/config.py") is None
        assert api._safe_dist_file("../../../../../../etc/passwd") is None

    @_needs_build
    def test_safe_dist_file_allows_index(self):
        # index.html lives in dist and must still resolve.
        assert api._safe_dist_file("index.html") is not None
