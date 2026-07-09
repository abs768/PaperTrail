"""Pydantic models: structured LLM outputs and API request bodies."""
from typing import Optional

from pydantic import BaseModel, Field


class Metric(BaseModel):
    name: str = Field(description="Name of the metric (e.g., accuracy, F1, BLEU)")
    value: str = Field(description="Reported value or qualitative result")


class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    relation: str = Field(
        description="Relation type: proposes, uses, evaluates_on, outperforms, extends, cites, applies, compares_with"
    )
    target: str = Field(description="Target entity name")


class PaperEntities(BaseModel):
    """Structured extraction of entities from a research paper."""

    title: Optional[str] = Field(description="Paper title if found, else null")
    authors: list[str] = Field(description="Author names found in the paper")
    methods: list[str] = Field(
        description="Methods, algorithms, or techniques mentioned"
    )
    datasets: list[str] = Field(description="Dataset names mentioned")
    metrics: list[Metric] = Field(description="Metrics and their reported values")
    key_concepts: list[str] = Field(
        description="Important domain concepts and technical terms"
    )
    relationships: list[Relationship] = Field(
        description="Explicit relationships between entities found in the text"
    )


class QueryClassification(BaseModel):
    """Router: classify what kind of query this is."""

    query_type: str = Field(
        description="Type: 'factual' (specific fact), 'comparative' (compare papers/methods), 'exploratory' (broad overview), 'relational' (connections between entities)"
    )
    key_entities: list[str] = Field(
        description="Key entities/concepts from the question to search for"
    )
    search_strategy: str = Field(
        description="Recommended search: 'vector_heavy' (rely on passages), 'graph_heavy' (rely on entity connections), 'balanced' (both)"
    )


class CitedSource(BaseModel):
    """A grounded citation. paper_title and page are derived server-side from
    passage_idx — do not have the LLM emit them. The LLM emits only:
      - passage_idx: the 1-indexed passage number it pulled the quote from
      - quote: a verbatim contiguous quote from that passage's text
      - relevant_detail: one short sentence describing what the quote supports
    """

    passage_idx: int = Field(
        description="1-indexed passage number from the numbered context list. MUST refer to an actual supplied passage."
    )
    quote: str = Field(
        description="Verbatim contiguous quote (5-30 words) copied character-for-character from the passage. No paraphrasing."
    )
    relevant_detail: str = Field(
        description="One short sentence describing what claim this quote supports."
    )
    # Filled in server-side from passage_idx — LLM does not emit these.
    paper_title: Optional[str] = Field(default=None)
    page: Optional[int] = Field(default=None)
    chunk_id: Optional[str] = Field(default=None)
    verified: Optional[bool] = Field(default=None)


class FaithfulnessReport(BaseModel):
    """Output of the post-generation fact-check pass."""

    unsupported_claims: list[str] = Field(
        description="Substantive factual claims from the ANSWER that are NOT supported by any PASSAGE. "
                    "Quote each claim verbatim from the answer. Skip meta-statements, hedges, and definitions."
    )
    support_score: float = Field(
        description="Fraction of substantive factual claims in the ANSWER that ARE supported by the PASSAGES (0..1)."
    )
    notes: str = Field(
        default="",
        description="One-sentence reasoning."
    )


class QueryAnswer(BaseModel):
    """Structured answer to a user's question."""

    answer: str = Field(description="Clear, comprehensive answer to the question")
    sources: list[CitedSource] = Field(
        description="Grounded citations. Each source must reference a numbered passage and quote it verbatim."
    )
    confidence: float = Field(
        description="Confidence in the answer (0-1), lower if context was sparse"
    )
    follow_up_questions: list[str] = Field(
        description="2-3 suggested follow-up questions the user might ask"
    )


# ── API request bodies ─────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str
    top_k: int = 15


class NoteRequest(BaseModel):
    title: str
    content: str


class UrlUploadRequest(BaseModel):
    url: str
