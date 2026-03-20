from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from torch.utils.data import Dataset

from bitnet_embed.data.schemas import PairExample, QueryDocumentExample, TripletExample


@dataclass(slots=True)
class DatasetSpec:
    name: str
    subset: str | None = None
    split: str = "train"
    sample_size: int | None = None
    local_path: str | None = None


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


def rows_to_pair_examples(rows: list[dict[str, Any]]) -> list[PairExample]:
    return [
        PairExample(
            anchor=str(row["anchor"]),
            positive=str(row["positive"]),
            task=str(row.get("task", "semantic_similarity")),
            source=str(row.get("source", "unknown")),
        )
        for row in rows
    ]


def rows_to_triplet_examples(rows: list[dict[str, Any]]) -> list[TripletExample]:
    return [
        TripletExample(
            anchor=str(row["anchor"]),
            positive=str(row["positive"]),
            negative=str(row["negative"]),
            task=str(row.get("task", "retrieval")),
            source=str(row.get("source", "unknown")),
        )
        for row in rows
    ]


def rows_to_query_document_examples(rows: list[dict[str, Any]]) -> list[QueryDocumentExample]:
    return [
        QueryDocumentExample(
            query=str(row["query"]),
            document=str(row["document"]),
            label=int(row.get("label", 0)),
            source=str(row.get("source", "unknown")),
        )
        for row in rows
    ]
