from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Any, Generic, TypeVar

from torch.utils.data import Dataset, IterableDataset

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
    materialization: str = "eager"


T = TypeVar("T")


class ExampleDataset(Dataset[T], Generic[T]):
    def __init__(self, items: list[T]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]


class IterableExampleDataset(IterableDataset[T], Generic[T]):
    def __init__(self, iterator_factory: Callable[[], Iterator[T]]) -> None:
        self._iterator_factory = iterator_factory

    def __iter__(self) -> Iterator[T]:
        return self._iterator_factory()


def load_jsonl_records(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected mapping rows in {path}")
            records.append(payload)
    return records


def iter_jsonl_records(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected mapping rows in {path}")
            yield payload


def count_jsonl_records(path: Path | str) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
    return count


def _is_lazy(spec: DatasetSpec) -> bool:
    return spec.materialization == "lazy"


def iter_dataset_records(spec: DatasetSpec) -> Iterator[dict[str, Any]]:
    if spec.local_path is not None:
        rows: Iterable[dict[str, Any]] = iter_jsonl_records(spec.local_path)
        if spec.sample_size is not None:
            rows = islice(rows, spec.sample_size)
        yield from rows
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load Hugging Face datasets") from exc

    dataset = load_dataset(
        spec.name,
        spec.subset,
        split=spec.split,
        streaming=_is_lazy(spec),
    )
    dataset_rows: Iterable[dict[str, Any]] = (dict(row) for row in dataset)
    if spec.sample_size is not None:
        dataset_rows = islice(dataset_rows, spec.sample_size)
    yield from dataset_rows


def infer_dataset_size(spec: DatasetSpec) -> int | None:
    if spec.sample_size is not None:
        return spec.sample_size
    if _is_lazy(spec):
        return count_jsonl_records(spec.local_path) if spec.local_path is not None else None
    if spec.local_path is not None:
        return count_jsonl_records(spec.local_path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load Hugging Face datasets") from exc

    dataset = load_dataset(spec.name, spec.subset, split=spec.split)
    return len(dataset)


def load_dataset_records(spec: DatasetSpec) -> list[dict[str, Any]]:
    return list(iter_dataset_records(spec))


def _row_to_example(
    row: dict[str, Any], dataset_format: str
) -> PairExample | TripletExample | QueryDocumentExample | ScoredPairExample | LabeledTextExample:
    if dataset_format == "pair":
        return PairExample(
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
    if dataset_format == "triplet":
        return TripletExample(
            anchor=str(row.get("anchor", row.get("query", ""))),
            positive=str(row.get("positive", row.get("document", row.get("answer", "")))),
            negative=str(row.get("negative", row.get("hard_negative", ""))),
            task=str(row.get("task", "retrieval")),
            source=str(row.get("source", "unknown")),
        )
    if dataset_format == "query_document":
        return QueryDocumentExample(
            query=str(row.get("query", row.get("anchor", ""))),
            document=str(row.get("document", row.get("positive", ""))),
            label=int(row.get("label", row.get("score", 0))),
            source=str(row.get("source", "unknown")),
        )
    if dataset_format == "scored_pair":
        return ScoredPairExample(
            left=str(row.get("left", row.get("sentence1", row.get("anchor", "")))),
            right=str(row.get("right", row.get("sentence2", row.get("positive", "")))),
            score=float(row.get("score", row.get("label", 0.0))),
            source=str(row.get("source", "unknown")),
        )
    if dataset_format == "labeled_text":
        return LabeledTextExample(
            text=str(row.get("text", row.get("document", row.get("anchor", "")))),
            label=int(row.get("label", 0)),
            source=str(row.get("source", "unknown")),
        )
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def iter_examples(
    spec: DatasetSpec,
) -> Iterator[
    PairExample | TripletExample | QueryDocumentExample | ScoredPairExample | LabeledTextExample
]:
    for row in iter_dataset_records(spec):
        yield _row_to_example(row, spec.format)


def iter_examples_from_specs(
    specs: Sequence[DatasetSpec],
) -> Iterator[
    PairExample | TripletExample | QueryDocumentExample | ScoredPairExample | LabeledTextExample
]:
    iterables = [iter_examples(spec) for spec in specs]
    return chain.from_iterable(iterables)


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
        item
        for item in (_row_to_example(row, "pair") for row in rows)
        if isinstance(item, PairExample)
    ]


def rows_to_triplet_examples(rows: list[dict[str, Any]]) -> list[TripletExample]:
    return [
        item
        for item in (_row_to_example(row, "triplet") for row in rows)
        if isinstance(item, TripletExample)
    ]


def rows_to_query_document_examples(rows: list[dict[str, Any]]) -> list[QueryDocumentExample]:
    return [
        item
        for item in (_row_to_example(row, "query_document") for row in rows)
        if isinstance(item, QueryDocumentExample)
    ]


def rows_to_scored_pair_examples(rows: list[dict[str, Any]]) -> list[ScoredPairExample]:
    return [
        item
        for item in (_row_to_example(row, "scored_pair") for row in rows)
        if isinstance(item, ScoredPairExample)
    ]


def rows_to_labeled_text_examples(rows: list[dict[str, Any]]) -> list[LabeledTextExample]:
    return [
        item
        for item in (_row_to_example(row, "labeled_text") for row in rows)
        if isinstance(item, LabeledTextExample)
    ]


def build_dataset_spec(payload: dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(
        name=str(payload.get("name", payload.get("local_path", "local"))),
        subset=str(payload["subset"]) if payload.get("subset") is not None else None,
        split=str(payload.get("split", "train")),
        sample_size=int(payload["sample_size"]) if payload.get("sample_size") is not None else None,
        local_path=str(payload["local_path"]) if payload.get("local_path") is not None else None,
        format=str(payload.get("format", "pair")),
        materialization=str(payload.get("materialization", "eager")),
    )


def load_examples(
    spec: DatasetSpec,
) -> Sequence[
    PairExample | TripletExample | QueryDocumentExample | ScoredPairExample | LabeledTextExample
]:
    return list(iter_examples(spec))
