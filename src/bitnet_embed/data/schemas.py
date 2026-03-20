from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class PairExample:
    anchor: str
    positive: str
    task: str = "semantic_similarity"
    source: str = "unknown"


@dataclass(slots=True)
class TripletExample:
    anchor: str
    positive: str
    negative: str
    task: str = "retrieval"
    source: str = "unknown"


@dataclass(slots=True)
class QueryDocumentExample:
    query: str
    document: str
    label: int
    source: str = "unknown"


@dataclass(slots=True)
class EncodedEmbeddingRecord:
    record_id: str
    text: str
    embedding: list[float]
    normalized: bool
    dim: int
    model: str
    metadata: dict[str, str] = field(default_factory=dict)


TaskMode = Literal["query", "document"]
