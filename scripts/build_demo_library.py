"""Build the public demo's library snapshot.

Ingests a fixed set of well-known papers through the *real* pipeline — the
same /upload-url endpoint the app uses, so the knowledge graph in the snapshot
is genuine extraction output — then exports the result to demo/library.json.

Run this once, locally, against your own server:

    export GROQ_API_KEY=...          # needed for extraction, only while building
    python main.py                   # in one terminal
    python scripts/build_demo_library.py   # in another

Then commit demo/library.json. The deployed demo loads that file and needs no
API key of its own: chunk embeddings are recomputed locally on load.

Ingestion is deliberately serial — the extractor issues several LLM calls per
paper, and a free-tier key will rate-limit if papers overlap.
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "library.json"

# Chosen so the graph has real structure: these papers genuinely cite and build
# on each other, so the extracted entities overlap instead of forming isolated
# islands. Keep the list short — every paper costs extraction calls.
PAPERS = [
    ("https://arxiv.org/abs/1706.03762", "Attention Is All You Need"),
    ("https://arxiv.org/abs/1810.04805", "BERT"),
    ("https://arxiv.org/abs/2005.14165", "GPT-3 / Few-Shot Learners"),
    ("https://arxiv.org/abs/2004.04906", "Dense Passage Retrieval"),
    ("https://arxiv.org/abs/2005.11401", "Retrieval-Augmented Generation"),
    ("https://arxiv.org/abs/2106.09685", "LoRA"),
    ("https://arxiv.org/abs/2201.11903", "Chain-of-Thought Prompting"),
]


def _post(base, path, payload, timeout):
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(base, path, timeout=120):
    with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://localhost:8000",
                    help="Running PaperTrail server")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Per-paper ingestion timeout in seconds")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    try:
        health = _get(args.base, "/api/health", timeout=10)
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach {args.base} — is the server running? ({e})")

    if not health.get("llm_enabled", True):
        sys.exit(
            "The server has no LLM key configured, so entity extraction would "
            "return nothing and the snapshot would have an empty graph. Set "
            "GROQ_API_KEY (or GEMINI_API_KEY) and restart it."
        )
    if health.get("demo_mode"):
        sys.exit("The server is in DEMO_MODE (read-only). Unset DEMO_MODE and restart it.")

    print(f"Server OK at {args.base} — ingesting {len(PAPERS)} papers\n")
    ok = 0
    for i, (url, label) in enumerate(PAPERS, 1):
        print(f"[{i}/{len(PAPERS)}] {label} … ", end="", flush=True)
        t0 = time.time()
        try:
            res = _post(args.base, "/upload-url", {"url": url}, args.timeout)
        except urllib.error.HTTPError as e:
            print(f"FAILED ({e.code}: {e.read()[:160].decode(errors='replace')})")
            continue
        except Exception as e:
            print(f"FAILED ({e})")
            continue
        dup = " [already indexed]" if res.get("duplicate") else ""
        found = res.get("entities_found") or {}
        print(f"ok in {time.time()-t0:.0f}s{dup} — "
              f"{found.get('methods', 0)} methods, {found.get('key_concepts', 0)} concepts, "
              f"{found.get('relationships', 0)} relationships")
        ok += 1

    if not ok:
        sys.exit("\nNo papers ingested — not writing a snapshot.")

    print("\nExporting …")
    payload = _get(args.base, "/export?include_chunks=true", timeout=300)

    graph = payload.get("graph") or {}
    nodes, links = graph.get("nodes") or [], graph.get("links") or []
    if not nodes:
        sys.exit("Export has an empty graph — refusing to write a useless snapshot.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(
        f"Wrote {out.relative_to(ROOT)} — {payload.get('paper_count', 0)} papers, "
        f"{len(nodes)} graph nodes, {len(links)} edges, "
        f"{len(payload.get('chunks') or [])} chunks, {size_mb:.1f} MB"
    )
    if size_mb > 45:
        print("\nWARNING: over 45 MB. GitHub warns above 50 MB and rejects at 100 MB.\n"
              "Drop a paper or two from PAPERS and rebuild.")
    print("\nNext: commit it, then deploy with DEMO_MODE=1 and no API key.")


if __name__ == "__main__":
    main()
