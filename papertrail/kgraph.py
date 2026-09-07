"""Knowledge graph: entity canonicalization, graph building, and traversal."""
import json
import logging
import re

from . import state

logger = logging.getLogger("papertrail")

# Canonical aliases — collapses surface variants to a single graph node so two
# papers that talk about the same thing actually link in the graph.
_ENTITY_ALIASES: dict = {
    # Architectures
    "rnn": "recurrent neural network",
    "rnns": "recurrent neural network",
    "recurrent neural networks": "recurrent neural network",
    "lstm": "long short-term memory",
    "lstms": "long short-term memory",
    "gru": "gated recurrent unit",
    "grus": "gated recurrent unit",
    "cnn": "convolutional neural network",
    "cnns": "convolutional neural network",
    "convolutional neural networks": "convolutional neural network",
    "mlp": "multi-layer perceptron",
    "mlps": "multi-layer perceptron",
    "ffn": "feed-forward network",
    "ffnn": "feed-forward network",
    # Attention family
    "attention": "attention",
    "attention mechanism": "attention",
    "attention mechanisms": "attention",
    "attentional mechanism": "attention",
    "attentional mechanisms": "attention",
    "soft attention": "attention",
    "additive attention": "additive attention",
    "bahdanau attention": "additive attention",
    "luong attention": "multiplicative attention",
    "multiplicative attention": "multiplicative attention",
    "dot-product attention": "scaled dot-product attention",
    "scaled dot product attention": "scaled dot-product attention",
    "multi head attention": "multi-head attention",
    "self attention": "self-attention",
    # Generic phrases
    "encoder decoder": "encoder-decoder",
    "encoder-decoder model": "encoder-decoder",
    "encoder-decoder models": "encoder-decoder",
    "encoder-decoder architecture": "encoder-decoder",
    "encoder-decoder approach": "encoder-decoder",
    "encoder-decoder framework": "encoder-decoder",
    "neural machine translation": "neural machine translation",
    "nmt": "neural machine translation",
    "machine translation": "machine translation",
    "sequence to sequence": "sequence-to-sequence",
    "sequence-to-sequence": "sequence-to-sequence",
    "seq2seq": "sequence-to-sequence",
    "seq-to-seq": "sequence-to-sequence",
    "transformer": "transformer",
    "transformer model": "transformer",
    "transformer architecture": "transformer",
    "vanilla transformer": "transformer",
    # Datasets
    "wmt'14": "wmt 2014",
    "wmt 14": "wmt 2014",
    "wmt14": "wmt 2014",
    "wmt 2014 english-french": "wmt 2014 en-fr",
    "wmt'14 english-french": "wmt 2014 en-fr",
    "wmt'14 english to french": "wmt 2014 en-fr",
    "wmt 2014 english-german": "wmt 2014 en-de",
    "wmt'14 english-german": "wmt 2014 en-de",
    "wmt'14 english to german": "wmt 2014 en-de",
    "english-french translation": "wmt 2014 en-fr",
    "english-to-french translation": "wmt 2014 en-fr",
    "english-german translation": "wmt 2014 en-de",
    "english-to-german translation": "wmt 2014 en-de",
    "wmt'15": "wmt 2015",
    "wmt 15": "wmt 2015",
    "wmt15": "wmt 2015",
    "iwslt'14": "iwslt 2014",
    "iwslt 14": "iwslt 2014",
    # Metrics
    "bleu score": "bleu",
    "bleu scores": "bleu",
    "bleu metric": "bleu",
}

_TRAILING_NOISE = (
    " models", " model", " mechanisms", " mechanism", " approach", " approaches",
    " architecture", " architectures", " framework", " frameworks",
    " method", " methods", " task", " tasks", " dataset", " datasets",
    " score", " scores", " metric", " metrics",
)


def _canonicalize_entity(name: str) -> str:
    """Lowercase, strip punctuation, expand aliases, drop trailing noise."""
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[''‘’“”`,;:.\(\)\[\]]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in _ENTITY_ALIASES:
        return _ENTITY_ALIASES[s]
    for suf in _TRAILING_NOISE:
        if s.endswith(suf):
            stripped = s[: -len(suf)].strip()
            if stripped in _ENTITY_ALIASES:
                return _ENTITY_ALIASES[stripped]
            if stripped:
                s = stripped
            break
    return s


def _add_entity_node(node_id: str, ntype: str, surface: str, **extra):
    """Add or merge an entity node, keeping the shortest surface form as the label."""
    kg = state.kg
    if kg.has_node(node_id):
        existing_label = kg.nodes[node_id].get("label", "")
        # Prefer the shorter, cleaner surface form for the visible label
        if len(surface) < len(existing_label) and len(surface) > 1:
            kg.nodes[node_id]["label"] = surface
            kg.nodes[node_id]["name"] = surface
    else:
        kg.add_node(node_id, type=ntype, name=surface, label=surface, **extra)


def add_to_knowledge_graph(paper_id: str, entities: dict):
    kg = state.kg
    paper_title = entities.get("title") or paper_id
    kg.add_node(paper_id, type="paper", title=paper_title, label=paper_title)

    for author in entities.get("authors", []):
        canon = _canonicalize_entity(author)
        if not canon:
            continue
        aid = f"author:{canon}"
        _add_entity_node(aid, "author", author)
        kg.add_edge(aid, paper_id, relation="authored")

    for method in entities.get("methods", []):
        canon = _canonicalize_entity(method)
        if not canon:
            continue
        mid = f"method:{canon}"
        _add_entity_node(mid, "method", method)
        kg.add_edge(paper_id, mid, relation="proposes")

    for dataset in entities.get("datasets", []):
        canon = _canonicalize_entity(dataset)
        if not canon:
            continue
        did = f"dataset:{canon}"
        _add_entity_node(did, "dataset", dataset)
        kg.add_edge(paper_id, did, relation="evaluates_on")

    for metric in entities.get("metrics", []):
        if isinstance(metric, dict):
            m_name = metric.get("name", "unknown")
            m_val = metric.get("value", "")
        else:
            m_name, m_val = str(metric), ""
        canon = _canonicalize_entity(m_name)
        if not canon:
            continue
        m_id = f"metric:{canon}"
        _add_entity_node(m_id, "metric", m_name, value=m_val)
        kg.add_edge(paper_id, m_id, relation="reports")

    for concept in entities.get("key_concepts", []):
        canon = _canonicalize_entity(concept)
        if not canon:
            continue
        cid = f"concept:{canon}"
        _add_entity_node(cid, "concept", concept)
        kg.add_edge(paper_id, cid, relation="discusses")

    for rel in entities.get("relationships", []):
        raw_src = rel.get("source", "")
        raw_tgt = rel.get("target", "")
        src = _canonicalize_entity(raw_src)
        tgt = _canonicalize_entity(raw_tgt)
        relation = rel.get("relation", "related_to")
        if not (src and tgt):
            continue
        # Reuse an existing node of the SAME canonical name (any type) before falling back to entity:
        src_id = next((nid for nid in kg.nodes if nid.split(":", 1)[-1] == src), f"entity:{src}")
        tgt_id = next((nid for nid in kg.nodes if nid.split(":", 1)[-1] == tgt), f"entity:{tgt}")
        if not kg.has_node(src_id):
            _add_entity_node(src_id, "entity", raw_src)
        if not kg.has_node(tgt_id):
            _add_entity_node(tgt_id, "entity", raw_tgt)
        kg.add_edge(src_id, tgt_id, relation=relation)

    logger.info(f"Graph updated — now {len(kg.nodes)} nodes, {len(kg.edges)} edges")


def traverse_knowledge_graph(entities: list[str], hops: int = 2) -> str:
    """Traverse the knowledge graph from given entities."""
    kg = state.kg
    relevant_info = []

    for entity in entities:
        entity_lower = entity.lower().strip()
        entity_canon = _canonicalize_entity(entity)
        for node_id, data in kg.nodes(data=True):
            node_name = data.get("name", data.get("title", "")).lower()
            node_canon = _canonicalize_entity(data.get("name", data.get("title", "")))
            if (
                entity_lower in node_name
                or node_name in entity_lower
                or (entity_canon and entity_canon == node_canon)
                or (entity_canon and (entity_canon in node_canon or node_canon in entity_canon))
            ):
                visited = {node_id}
                frontier = [node_id]
                for _ in range(hops):
                    next_frontier = []
                    for n in frontier:
                        neighbors = list(kg.successors(n)) + list(kg.predecessors(n))
                        for nb in neighbors:
                            if nb not in visited:
                                visited.add(nb)
                                next_frontier.append(nb)
                    frontier = next_frontier

                for nid in visited:
                    ndata = kg.nodes[nid]
                    node_info = {
                        "id": nid,
                        "type": ndata.get("type", "unknown"),
                        "name": ndata.get("name", ndata.get("title", nid)),
                        "connections": [],
                    }
                    for _, target, edata in kg.out_edges(nid, data=True):
                        tdata = kg.nodes.get(target, {})
                        node_info["connections"].append(
                            {
                                "relation": edata.get("relation", "related"),
                                "target": tdata.get("name", tdata.get("title", target)),
                                "target_type": tdata.get("type", "unknown"),
                            }
                        )
                    for source, _, edata in kg.in_edges(nid, data=True):
                        sdata = kg.nodes.get(source, {})
                        node_info["connections"].append(
                            {
                                "relation": f"<-{edata.get('relation', 'related')}-",
                                "target": sdata.get("name", sdata.get("title", source)),
                                "target_type": sdata.get("type", "unknown"),
                            }
                        )
                    relevant_info.append(node_info)

    if not relevant_info:
        return json.dumps({"results": [], "message": f"No matches found for: {entities}"})

    seen = set()
    unique = []
    for info in relevant_info:
        if info["id"] not in seen:
            seen.add(info["id"])
            unique.append(info)

    return json.dumps({"results": unique[:30]})


# Library items — the things that own chunks in the vector store. Notes are
# ingested as "note:<uuid>" and papers as "paper:<hash>"; both are added to the
# graph by add_to_knowledge_graph, so both must be recognised here.
_LIBRARY_ID_PREFIXES = ("paper:", "note:")


def _papers_in_subgraph(subgraph_json: str) -> list[str]:
    """Extract library item ids (papers and notes) from a traverse_knowledge_graph result."""
    try:
        data = json.loads(subgraph_json)
    except Exception:
        return []
    paper_ids = set()
    for entry in data.get("results", []):
        nid = entry.get("id", "")
        if nid.startswith(_LIBRARY_ID_PREFIXES):
            paper_ids.add(nid)
    return list(paper_ids)
