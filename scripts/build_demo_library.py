"""Build the public demo's library snapshot.

Ingests a fixed set of well-known papers through the *real* pipeline — the
same /upload-url endpoint the app uses, so the knowledge graph in the snapshot
is genuine extraction output — then exports the result to demo/library.json.

Run this once, locally, against your own server:

    export GROQ_API_KEY=...          # needed for extraction, only while building
    python main.py                   # in one terminal
    python scripts/build_demo_library.py --reset   # in another

Pass --reset unless the library is known to be empty: ingestion short-circuits
on duplicates, so re-running over papers that are already indexed skips
extraction and yields a graph of authors only.

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


def _delete(base, path, timeout=60):
    req = urllib.request.Request(f"{base}{path}", method="DELETE")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _graph_shape(graph: dict) -> tuple[int, int, dict, dict]:
    """Node/edge counts and type histograms, independent of the JSON edge key.

    networkx renamed node_link_data's edge key from "links" to "edges" in 3.6,
    so read whichever this export used rather than assuming one.
    """
    nodes = graph.get("nodes") or []
    edges = graph.get("edges")
    if edges is None:
        edges = graph.get("links") or []
    node_types: dict = {}
    for n in nodes:
        t = n.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    relations: dict = {}
    for e in edges:
        r = e.get("relation", "unknown")
        relations[r] = relations.get(r, 0) + 1
    return len(nodes), len(edges), node_types, relations


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://localhost:8000",
                    help="Running PaperTrail server")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Per-paper ingestion timeout in seconds")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--reset", action="store_true",
                    help="Wipe the server's library first. Required when the papers "
                         "were already ingested, because ingestion short-circuits on "
                         "duplicates and would skip extraction entirely.")
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

    if args.reset:
        print("Resetting the server's library first …")
        _delete(args.base, "/reset")

    print(f"Server OK at {args.base} — ingesting {len(PAPERS)} papers\n")
    ok = 0
    duplicates = 0
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
        if res.get("duplicate"):
            duplicates += 1
            print("already indexed — extraction SKIPPED")
        else:
            found = res.get("entities_found") or {}
            print(f"ok in {time.time()-t0:.0f}s — "
                  f"{found.get('methods', 0)} methods, {found.get('key_concepts', 0)} concepts, "
                  f"{found.get('relationships', 0)} relationships")
        ok += 1

    if not ok:
        sys.exit("\nNo papers ingested — not writing a snapshot.")

    if duplicates:
        print(f"\n{duplicates}/{len(PAPERS)} paper(s) were already in the library, so their "
              f"entities were NOT re-extracted.\nIf they were first ingested without an API "
              f"key their entities are empty, and the graph\nwill have no methods or concepts. "
              f"Re-run with --reset to rebuild from scratch.")

    print("\nExporting …")
    payload = _get(args.base, "/export?include_chunks=true", timeout=300)

    graph = payload.get("graph") or {}
    n_nodes, n_edges, node_types, relations = _graph_shape(graph)
    if not n_nodes:
        sys.exit("Export has an empty graph — refusing to write a useless snapshot.")

    print(f"  nodes by type: {dict(sorted(node_types.items(), key=lambda kv: -kv[1]))}")
    print(f"  edges by relation: {dict(sorted(relations.items(), key=lambda kv: -kv[1])[:6])}")

    # A graph of nothing but "author authored paper" is what you get when the
    # papers were ingested without a key: authors come from the arXiv API, but
    # methods and concepts come from LLM extraction. It looks like a graph and
    # is worthless for showing connections *between* papers, so refuse it.
    substantive_nodes = sum(v for k, v in node_types.items()
                            if k not in ("author", "paper", "unknown"))
    substantive_edges = sum(v for k, v in relations.items() if k != "authored")
    if substantive_nodes == 0 or substantive_edges == 0:
        sys.exit(
            "\nRefusing to write this snapshot: the graph has no method/concept "
            "nodes\nand no relations beyond 'authored', so it shows no connections "
            "between\npapers — which is the whole point of the demo.\n\n"
            "The usual cause is that the papers were already indexed (see the "
            "warning\nabove) so extraction was skipped. Re-run with --reset."
        )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(
        f"Wrote {out.relative_to(ROOT)} — {payload.get('paper_count', 0)} papers, "
        f"{n_nodes} graph nodes, {n_edges} edges, "
        f"{len(payload.get('chunks') or [])} chunks, {size_mb:.1f} MB"
    )
    if size_mb > 45:
        print("\nWARNING: over 45 MB. GitHub warns above 50 MB and rejects at 100 MB.\n"
              "Drop a paper or two from PAPERS and rebuild.")
    print("\nNext: commit it, then deploy with DEMO_MODE=1 and no API key.")


if __name__ == "__main__":
    main()
