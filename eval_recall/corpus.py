"""
A controlled retrieval corpus with ground-truth relevance defined by
construction, designed to be *realistically hard* (not a trivial exact-match
test). Each "paper" has one planted result passage; queries range from
lexically explicit to semantic-only, and datasets are shared across papers so a
dataset name alone cannot disambiguate the answer.

All text is generated here (invented methods / datasets / numbers); there is no
external or copyrighted content. Swap in a real labeled set (see load_labeled)
to evaluate PaperTrail on your own papers.
"""
import random

METHODS = [
    "ZephyrNet", "OrbitalMixer", "GraniteFormer", "PelicanGraph", "VesperRAG",
    "CobaltGate", "NimbusEncoder", "AsterKV", "QuartzRouter", "HalcyonMoE",
    "DriftAlign", "SableProbe", "LumenChain", "TidalPool", "EmberIndex",
]
# 5 datasets, each shared by 3 papers -> the dataset name alone is ambiguous.
DATASETS = ["ZephyrBench", "CartoQA", "MosaicWiki", "PelagicDocs", "TundraSet"]
METRICS = ["nDCG@10", "MRR", "exact-match", "F1", "Recall@20"]
TASKS = [
    "cross-document retrieval", "citation grounding", "entity linking",
    "long-context summarization", "table extraction", "claim verification",
    "multi-hop question answering", "passage reranking",
]


def _generic(method, task, ablation):
    return (f"{method} targets {task}. It pairs a sparse lexical stage with a "
            f"dense encoder and fuses their candidate lists, aiming for "
            f"robustness under distribution shift at a bounded per-query cost. "
            f"An ablation removes the {ablation} stage to quantify its effect.")


def _fact(method, dataset, metric, value):
    # The single planted, uniquely queryable result passage.
    return (f"On the {dataset} benchmark, {method} attains a {metric} of "
            f"{value:.3f}, the strongest result we report, ahead of every "
            f"prior baseline evaluated under the same protocol.")


def build(seed=13, n_papers=15, chunks_per_paper=8):
    rng = random.Random(seed)
    papers, queries = [], []

    for pi in range(n_papers):
        method = METHODS[pi]
        dataset = DATASETS[pi % len(DATASETS)]     # each dataset shared by 3 papers
        metric = METRICS[pi % len(METRICS)]
        value = rng.uniform(0.55, 0.95)
        task = rng.choice(TASKS)

        chunks, fact_idx = [], rng.randrange(1, chunks_per_paper - 1)
        for ci in range(chunks_per_paper):
            if ci == fact_idx:
                chunks.append(_fact(method, dataset, metric, value))
            else:
                chunks.append(_generic(method, rng.choice(TASKS),
                                       rng.choice(["lexical", "dense", "fusion", "rerank"])))
        papers.append({"paper_id": f"P{pi:02d}",
                       "title": f"{method}: A System for {task.title()}",
                       "chunks": chunks})
        gold = f"P{pi:02d}_chunk_{fact_idx}"

        # 1) explicit: all key tokens present (easiest)
        queries.append({"query": f"What {metric} does {method} achieve on {dataset}?",
                        "gold_chunk_id": gold, "type": "explicit"})
        # 2) paraphrase: method + dataset, metric dropped, wording changed
        queries.append({"query": f"How well does {method} perform on the {dataset} dataset?",
                        "gold_chunk_id": gold, "type": "paraphrase"})
        # 3) semantic: dataset + metric, NO method name -> must disambiguate
        #    among the three papers that share this dataset (hardest)
        queries.append({"query": f"Which system reports the best {metric} on {dataset}?",
                        "gold_chunk_id": gold, "type": "semantic"})

    return papers, queries


def load_labeled(path):
    """Load a real labeled set: JSONL of {query, gold_chunk_id}."""
    import json
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
