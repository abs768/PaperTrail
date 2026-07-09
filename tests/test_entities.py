"""Unit tests for entity canonicalization, validation, and extraction merging."""
from papertrail import extraction, kgraph


class TestCanonicalization:
    def test_alias_expansion(self):
        assert kgraph._canonicalize_entity("RNN") == "recurrent neural network"
        assert kgraph._canonicalize_entity("RNNs") == "recurrent neural network"
        assert kgraph._canonicalize_entity("seq2seq") == "sequence-to-sequence"
        assert kgraph._canonicalize_entity("BLEU score") == "bleu"

    def test_trailing_noise_stripped(self):
        assert kgraph._canonicalize_entity("transformer architecture") == "transformer"
        assert kgraph._canonicalize_entity("Adam method") == "adam"

    def test_punctuation_and_case(self):
        assert kgraph._canonicalize_entity("  Self-Attention, ") == "self-attention"
        assert kgraph._canonicalize_entity("") == ""

    def test_variants_collapse_to_same_node(self):
        a = kgraph._canonicalize_entity("recurrent neural networks")
        b = kgraph._canonicalize_entity("RNNs")
        assert a == b


class TestCitationDetection:
    def test_et_al_is_citation(self):
        assert extraction._looks_like_citation("Fedus et al. (2018)")
        assert extraction._looks_like_citation("Radford et al., 2018")

    def test_two_author_and_form_is_citation(self):
        assert extraction._looks_like_citation("Howard and Ruder")

    def test_year_is_citation(self):
        assert extraction._looks_like_citation("Dai and Le, 2015")

    def test_plain_names_are_not_citations(self):
        assert not extraction._looks_like_citation("Ashish Vaswani")
        # 'and' inside a name must not trigger the two-author pattern.
        assert not extraction._looks_like_citation("Alexandra Fernandez")


class TestAuthorValidation:
    TEXT = "attention is all you need ashish vaswani noam shazeer google brain"

    def test_author_with_surname_in_text(self):
        assert extraction._author_appears("Ashish Vaswani", self.TEXT)

    def test_single_token_org_rejected(self):
        assert not extraction._author_appears("Google", self.TEXT)

    def test_citation_rejected(self):
        assert not extraction._author_appears("Vaswani et al.", self.TEXT)

    def test_absent_author_rejected(self):
        assert not extraction._author_appears("Jane Nowhere", self.TEXT)


class TestEntityAppearance:
    def test_direct_substring(self):
        assert extraction._entity_appears("transformer", "the transformer model wins")

    def test_alias_bridging(self):
        # LLM emitted the long form; the text only has the acronym.
        assert extraction._entity_appears("recurrent neural network", "we use an rnn here")

    def test_token_overlap(self):
        assert extraction._entity_appears(
            "multi-head self attention", "our multi-head attention layers"
        )

    def test_hallucination_rejected(self):
        assert not extraction._entity_appears("quantum annealing", "we train a transformer")


class TestValidationAndMerge:
    def test_validate_drops_hallucinated_entities(self):
        entities = {
            **extraction._empty_extraction(),
            "methods": ["transformer", "quantum annealing"],
            "datasets": ["wmt 2014"],
        }
        text = "The Transformer evaluates on WMT 2014 English-German."
        kept, dropped = extraction._validate_extraction(entities, text)
        assert kept["methods"] == ["transformer"]
        assert dropped["methods"] == 1
        assert kept["datasets"] == ["wmt 2014"]

    def test_merge_dedupes_by_canonical_form(self):
        a = {**extraction._empty_extraction(), "title": "T", "methods": ["RNN"]}
        b = {**extraction._empty_extraction(), "methods": ["recurrent neural networks"]}
        merged = extraction._merge_extractions([a, b])
        assert merged["title"] == "T"
        assert len(merged["methods"]) == 1

    def test_merge_relationships_deduped(self):
        rel = {"source": "transformer", "relation": "outperforms", "target": "RNN"}
        rel2 = {"source": "Transformer", "relation": "outperforms",
                "target": "recurrent neural network"}
        a = {**extraction._empty_extraction(), "relationships": [rel]}
        b = {**extraction._empty_extraction(), "relationships": [rel2]}
        merged = extraction._merge_extractions([a, b])
        assert len(merged["relationships"]) == 1
