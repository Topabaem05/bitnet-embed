from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch

from bitnet_embed.data.schemas import QueryDocumentExample, ScoredPairExample
from bitnet_embed.eval.retrieval import evaluate_retrieval
from bitnet_embed.eval.sts import spearman_correlation
from bitnet_embed.modeling.model import EncodeConfig


def evaluate_scored_pairs(model: Any, examples: Sequence[ScoredPairExample]) -> dict[str, float]:
    if not examples:
        return {}
    left_embeddings = model.encode(
        [example.left for example in examples],
        EncodeConfig(task="query", batch_size=max(1, len(examples))),
    )
    right_embeddings = model.encode(
        [example.right for example in examples],
        EncodeConfig(task="document", batch_size=max(1, len(examples))),
    )
    labels = torch.tensor([example.score for example in examples], dtype=torch.float32)
    return {"sts_spearman": spearman_correlation(left_embeddings, right_embeddings, labels)}


def evaluate_query_documents(
    model: Any,
    examples: Sequence[QueryDocumentExample],
) -> dict[str, float]:
    if not examples:
        return {}

    grouped_documents: dict[str, list[str]] = defaultdict(list)
    relevant_indices: dict[int, set[int]] = defaultdict(set)
    ordered_queries: list[str] = []
    corpus: list[str] = []
    corpus_index: dict[tuple[str, str], int] = {}

    for example in examples:
        if example.query not in grouped_documents:
            grouped_documents[example.query] = []
            ordered_queries.append(example.query)
        key = (example.query, example.document)
        if key not in corpus_index:
            corpus_index[key] = len(corpus)
            corpus.append(example.document)
            grouped_documents[example.query].append(example.document)
        if example.label > 0:
            query_idx = ordered_queries.index(example.query)
            relevant_indices[query_idx].add(corpus_index[key])

    query_embeddings = model.encode(
        ordered_queries,
        EncodeConfig(task="query", batch_size=max(1, len(ordered_queries))),
    )
    doc_embeddings = model.encode(
        corpus,
        EncodeConfig(task="document", batch_size=max(1, len(corpus))),
    )
    return evaluate_retrieval(query_embeddings, doc_embeddings, dict(relevant_indices))
