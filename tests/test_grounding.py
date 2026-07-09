"""Unit tests for quote verification (citation grounding)."""
from papertrail import query

CHUNK = (
    "The dominant sequence transduction models are based on complex recurrent "
    "or convolutional neural networks that include an encoder and a decoder. "
    "The best performing models also connect the encoder and decoder through "
    "an attention mechanism."
)


class TestVerifyQuote:
    def test_exact_quote_accepted(self):
        assert query._verify_quote(
            "the best performing models also connect the encoder", CHUNK
        )

    def test_smart_quotes_and_case_normalized(self):
        assert query._verify_quote(
            "The Best–performing models also connect the encoder".replace(
                "–", "-"
            ),
            CHUNK,
        )

    def test_paraphrase_rejected(self):
        assert not query._verify_quote(
            "top models use attention to link encoders with decoders and improve results",
            CHUNK,
        )

    def test_too_short_rejected(self):
        assert not query._verify_quote("attention mechanism", CHUNK, min_words=3)

    def test_empty_rejected(self):
        assert not query._verify_quote("", CHUNK)
        assert not query._verify_quote("some quote here", "")

    def test_minor_ocr_noise_accepted_by_fuzzy_match(self):
        quote = "complex recurrent or convolutional neural netvvorks that include an encoder"
        assert query._verify_quote(quote, CHUNK)

    def test_whitespace_differences_accepted(self):
        quote = "sequence   transduction models are based\non complex recurrent"
        assert query._verify_quote(quote, CHUNK)
