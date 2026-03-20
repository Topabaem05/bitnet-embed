from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as functional


def cosine_similarity_matrix(
    query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor
) -> torch.Tensor:
    query_embeddings = functional.normalize(query_embeddings, p=2, dim=-1)
    doc_embeddings = functional.normalize(doc_embeddings, p=2, dim=-1)
    return query_embeddings @ doc_embeddings.T


def recall_at_k(ranked_indices: Sequence[int], relevant_indices: set[int], k: int) -> float:
    if not relevant_indices:
        return 0.0
    hits = sum(1 for index in ranked_indices[:k] if index in relevant_indices)
    return hits / len(relevant_indices)


def reciprocal_rank(ranked_indices: Sequence[int], relevant_indices: set[int], k: int) -> float:
    for rank, index in enumerate(ranked_indices[:k], start=1):
        if index in relevant_indices:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_indices: Sequence[int], relevant_indices: set[int], k: int) -> float:
    dcg = 0.0
    for rank, index in enumerate(ranked_indices[:k], start=1):
        if index in relevant_indices:
            dcg += 1.0 / torch.log2(torch.tensor(rank + 1.0)).item()
    ideal_hits = min(k, len(relevant_indices))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(
        1.0 / torch.log2(torch.tensor(rank + 1.0)).item() for rank in range(1, ideal_hits + 1)
    )
    return dcg / idcg


def evaluate_retrieval(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    relevant_doc_indices: dict[int, set[int]],
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    similarity = cosine_similarity_matrix(query_embeddings, doc_embeddings)
    rankings = similarity.argsort(dim=1, descending=True)
    metrics: dict[str, float] = {}
    for k in ks:
        recalls = []
        reciprocal_ranks = []
        ndcgs = []
        for query_index, ranked in enumerate(rankings.tolist()):
            relevant = relevant_doc_indices.get(query_index, set())
            recalls.append(recall_at_k(ranked, relevant, k))
            reciprocal_ranks.append(reciprocal_rank(ranked, relevant, k))
            ndcgs.append(ndcg_at_k(ranked, relevant, k))
        metrics[f"recall@{k}"] = sum(recalls) / max(1, len(recalls))
        metrics[f"mrr@{k}"] = sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))
        metrics[f"ndcg@{k}"] = sum(ndcgs) / max(1, len(ndcgs))
    return metrics
