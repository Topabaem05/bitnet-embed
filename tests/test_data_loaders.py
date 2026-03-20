from __future__ import annotations

from bitnet_embed.data.loaders import (
    build_dataset_spec,
    infer_dataset_size,
    iter_examples,
    load_examples,
    load_jsonl_records,
)
from bitnet_embed.data.schemas import (
    PairExample,
    QueryDocumentExample,
    ScoredPairExample,
    TripletExample,
)


def test_load_jsonl_records_reads_local_smoke_files() -> None:
    rows = load_jsonl_records("data/smoke/pairs.jsonl")
    assert len(rows) == 4
    assert rows[0]["anchor"] == "a happy dog"


def test_load_examples_supports_all_local_smoke_formats() -> None:
    pair_examples = load_examples(
        build_dataset_spec({"local_path": "data/smoke/pairs.jsonl", "format": "pair"})
    )
    triplet_examples = load_examples(
        build_dataset_spec({"local_path": "data/smoke/triplets.jsonl", "format": "triplet"})
    )
    scored_pair_examples = load_examples(
        build_dataset_spec({"local_path": "data/smoke/sts.jsonl", "format": "scored_pair"})
    )
    query_document_examples = load_examples(
        build_dataset_spec(
            {"local_path": "data/smoke/query_documents.jsonl", "format": "query_document"}
        )
    )
    assert isinstance(pair_examples[0], PairExample)
    assert isinstance(triplet_examples[0], TripletExample)
    assert isinstance(scored_pair_examples[0], ScoredPairExample)
    assert isinstance(query_document_examples[0], QueryDocumentExample)


def test_iter_examples_supports_lazy_local_materialization() -> None:
    spec = build_dataset_spec(
        {
            "local_path": "data/smoke/pairs.jsonl",
            "format": "pair",
            "materialization": "lazy",
            "sample_size": 2,
        }
    )
    examples = list(iter_examples(spec))
    assert len(examples) == 2
    assert isinstance(examples[0], PairExample)


def test_infer_dataset_size_for_lazy_local_dataset() -> None:
    spec = build_dataset_spec(
        {"local_path": "data/smoke/pairs.jsonl", "format": "pair", "materialization": "lazy"}
    )
    assert infer_dataset_size(spec) == 4
