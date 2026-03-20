from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from torch.utils.data import Dataset

from bitnet_embed.data.schemas import (
    LabeledTextExample,
    PairExample,
    QueryDocumentExample,
    ScoredPairExample,
    TripletExample,
)


@dataclass(slots=True)
class DatasetSpec:
    name: str
    subset: str | None = None
    split: str = "train"
    sample_size: int | None = None
    local_path: str | None = None
    format: str = "pair"


T = TypeVar("T")


class ExampleDataset(Dataset[T], Generic[T]):
    def __init__(self, items: list[T]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]


def load_jsonl_records(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected mapping rows in {path}")
            records.append(payload)
    return records


def load_dataset_records(spec: DatasetSpec) -> list[dict[str, Any]]:
    if spec.local_path is not None:
        rows = load_jsonl_records(spec.local_path)
        return rows[: spec.sample_size] if spec.sample_size is not None else rows

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load Hugging Face datasets") from exc

    dataset = load_dataset(spec.name, spec.subset, split=spec.split)
    if spec.sample_size is not None:
        dataset = dataset.select(range(min(spec.sample_size, len(dataset))))
    return [dict(row) for row in dataset]


def build_smoke_pairs() -> list[PairExample]:
    return [
        PairExample(anchor="a happy dog", positive="a joyful dog", source="smoke"),
        PairExample(anchor="a fast car", positive="a quick automobile", source="smoke"),
        PairExample(anchor="an apple a day", positive="daily fruit can help", source="smoke"),
        PairExample(anchor="ocean waves", positive="sea surf", source="smoke"),
    ]


def build_smoke_triplets() -> list[TripletExample]:
    return [
        TripletExample(
            anchor="find lupus symptoms",
            positive="lupus symptoms include fatigue and joint pain",
            negative="stock prices rose today",
            source="smoke",
        ),
        TripletExample(
            anchor="benefits of vitamin d",
            positive="vitamin d helps bone health",
            negative="car tires need rotation",
            source="smoke",
        ),
    ]


def build_smoke_scored_pairs() -> list[ScoredPairExample]:
    return [
        ScoredPairExample(left="a happy dog", right="a joyful dog", score=0.95, source="smoke"),
        ScoredPairExample(
            left="a fast car", right="a quick automobile", score=0.92, source="smoke"
        ),
        ScoredPairExample(left="ocean waves", right="sea surf", score=0.88, source="smoke"),
        ScoredPairExample(left="vitamin d", right="car tires", score=0.05, source="smoke"),
    ]


def build_smoke_query_documents() -> list[QueryDocumentExample]:
    return [
        QueryDocumentExample(
            query="find lupus symptoms",
            document="lupus symptoms include fatigue and joint pain",
            label=1,
            source="smoke",
        ),
        QueryDocumentExample(
            query="find lupus symptoms",
            document="stock prices rose today",
            label=0,
            source="smoke",
        ),
        QueryDocumentExample(
            query="benefits of vitamin d",
            document="vitamin d supports bone health",
            label=1,
            source="smoke",
        ),
        QueryDocumentExample(
            query="benefits of vitamin d",
            document="car tires need rotation",
            label=0,
            source="smoke",
        ),
    ]


def build_smoke_labeled_texts() -> list[LabeledTextExample]:
    return [
        LabeledTextExample(text="cat and dog", label=0, source="smoke"),
        LabeledTextExample(text="puppy and kitten", label=0, source="smoke"),
        LabeledTextExample(text="stock market rally", label=1, source="smoke"),
        LabeledTextExample(text="earnings and finance", label=1, source="smoke"),
    ]


def rows_to_pair_examples(rows: list[dict[str, Any]]) -> list[PairExample]:
    return [
        PairExample(
            anchor=str(
                row.get("anchor", row.get("query", row.get("sentence1", row.get("text", ""))))
            ),
            positive=str(
                row.get(
                    "positive",
                    row.get("document", row.get("answer", row.get("sentence2", ""))),
                )
            ),
            task=str(row.get("task", "semantic_similarity")),
            source=str(row.get("source", "unknown")),
        )
        for row in rows
    ]


def rows_to_triplet_examples(rows: list[dict[str, Any]]) -> list[TripletExample]:
    return [
        TripletExample(
            anchor=str(row.get("anchor", row.get("query", ""))),
            positive=str(row.get("positive", row.get("document", row.get("answer", "")))),
            negative=str(row.get("negative", row.get("hard_negative", ""))),
            task=str(row.get("task", "retrieval")),
            source=str(row.get("source", "unknown")),
        )
        for row in rows
    ]


def rows_to_query_document_examples(rows: list[dict[str, Any]]) -> list[QueryDocumentExample]:
    return [
        QueryDocumentExample(
            query=str(row.get("query", row.get("anchor", ""))),
            document=str(row.get("document", row.get("positive", ""))),
            label=int(row.get("label", row.get("score", 0))),
            source=str(row.get("source", "unknown")),
        )
        for row in rows
    ]


def rows_to_scored_pair_examples(rows: list[dict[str, Any]]) -> list[ScoredPairExample]:
    return [
        ScoredPairExample(
            left=str(row.get("left", row.get("sentence1", row.get("anchor", "")))),
            right=str(row.get("right", row.get("sentence2", row.get("positive", "")))),
            score=float(row.get("score", row.get("label", 0.0))),
            source=str(row.get("source", "unknown")),
        )
        for row in rows
    ]


def rows_to_labeled_text_examples(rows: list[dict[str, Any]]) -> list[LabeledTextExample]:
    return [
        LabeledTextExample(
            text=str(row.get("text", row.get("document", row.get("anchor", "")))),
            label=int(row.get("label", 0)),
            source=str(row.get("source", "unknown")),
        )
        for row in rows
    ]


def build_dataset_spec(payload: dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(
        name=str(payload.get("name", payload.get("local_path", "local"))),
        subset=str(payload["subset"]) if payload.get("subset") is not None else None,
        split=str(payload.get("split", "train")),
        sample_size=int(payload["sample_size"]) if payload.get("sample_size") is not None else None,
        local_path=str(payload["local_path"]) if payload.get("local_path") is not None else None,
        format=str(payload.get("format", "pair")),
    )


def load_examples(
    spec: DatasetSpec,
) -> Sequence[
    PairExample | TripletExample | QueryDocumentExample | ScoredPairExample | LabeledTextExample
]:
    rows = load_dataset_records(spec)
    if spec.format == "pair":
        return rows_to_pair_examples(rows)
    if spec.format == "triplet":
        return rows_to_triplet_examples(rows)
    if spec.format == "query_document":
        return rows_to_query_document_examples(rows)
    if spec.format == "scored_pair":
        return rows_to_scored_pair_examples(rows)
    if spec.format == "labeled_text":
        return rows_to_labeled_text_examples(rows)
    raise ValueError(f"Unsupported dataset format: {spec.format}")
