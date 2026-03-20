from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from bitnet_embed.data.schemas import QueryDocumentExample
from bitnet_embed.eval.retrieval import ndcg_at_k, recall_at_k, reciprocal_rank
from bitnet_embed.modeling.model import EncodeConfig


@dataclass(slots=True)
class SearchHit:
    index: int
    score: float


class InMemoryAnnIndex:
    def __init__(self, embeddings: torch.Tensor) -> None:
        self.embeddings = functional.normalize(embeddings, p=2, dim=-1)

    def search(self, query_embeddings: torch.Tensor, top_k: int) -> list[list[SearchHit]]:
        normalized_queries = functional.normalize(query_embeddings, p=2, dim=-1)
        scores = normalized_queries @ self.embeddings.T
        top_scores, top_indices = scores.topk(k=min(top_k, self.embeddings.size(0)), dim=1)
        results: list[list[SearchHit]] = []
        for row_indices, row_scores in zip(top_indices.tolist(), top_scores.tolist(), strict=True):
            results.append(
                [
                    SearchHit(index=index, score=score)
                    for index, score in zip(row_indices, row_scores, strict=True)
                ]
            )
        return results


def build_query_document_corpus(
    examples: Sequence[QueryDocumentExample],
) -> tuple[list[str], list[str], dict[int, set[int]]]:
    ordered_queries: list[str] = []
    query_index_map: dict[str, int] = {}
    documents: list[str] = []
    document_index_map: dict[tuple[str, str], int] = {}
    relevant_indices: dict[int, set[int]] = defaultdict(set)

    for example in examples:
        if example.query not in query_index_map:
            query_index_map[example.query] = len(ordered_queries)
            ordered_queries.append(example.query)
        key = (example.query, example.document)
        if key not in document_index_map:
            document_index_map[key] = len(documents)
            documents.append(example.document)
        if example.label > 0:
            relevant_indices[query_index_map[example.query]].add(document_index_map[key])
    return ordered_queries, documents, dict(relevant_indices)


def evaluate_ann_search(
    rankings: Sequence[Sequence[SearchHit]],
    relevant_doc_indices: dict[int, set[int]],
    ks: Sequence[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    ranked_indices = [[hit.index for hit in hits] for hits in rankings]
    for k in ks:
        recalls = []
        reciprocal_ranks = []
        ndcgs = []
        for query_index, ranked in enumerate(ranked_indices):
            relevant = relevant_doc_indices.get(query_index, set())
            recalls.append(recall_at_k(ranked, relevant, k))
            reciprocal_ranks.append(reciprocal_rank(ranked, relevant, k))
            ndcgs.append(ndcg_at_k(ranked, relevant, k))
        metrics[f"ann_recall@{k}"] = sum(recalls) / max(1, len(recalls))
        metrics[f"ann_mrr@{k}"] = sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))
        metrics[f"ann_ndcg@{k}"] = sum(ndcgs) / max(1, len(ndcgs))
    return metrics


def validate_ann(
    model: Any,
    examples: Sequence[QueryDocumentExample],
    *,
    top_k: int = 10,
) -> dict[str, float]:
    if not examples:
        return {}
    queries, documents, relevant_doc_indices = build_query_document_corpus(examples)
    query_embeddings = model.encode(
        queries,
        EncodeConfig(task="query", batch_size=max(1, len(queries))),
    )
    document_embeddings = model.encode(
        documents,
        EncodeConfig(task="document", batch_size=max(1, len(documents))),
    )
    index = InMemoryAnnIndex(document_embeddings)
    rankings = index.search(query_embeddings, top_k=top_k)
    return evaluate_ann_search(rankings, relevant_doc_indices, ks=(1, min(5, top_k), top_k))
