from __future__ import annotations

from bitnet_embed.data.loaders import build_smoke_query_documents, build_smoke_scored_pairs
from bitnet_embed.eval.harness import evaluate_query_documents, evaluate_scored_pairs
from bitnet_embed.modeling.smoke import build_toy_embedding_model


def test_evaluate_scored_pairs_returns_spearman_metric() -> None:
    model = build_toy_embedding_model()
    metrics = evaluate_scored_pairs(model, build_smoke_scored_pairs())
    assert "sts_spearman" in metrics


def test_evaluate_query_documents_returns_retrieval_metrics() -> None:
    model = build_toy_embedding_model()
    metrics = evaluate_query_documents(model, build_smoke_query_documents())
    assert "recall@1" in metrics
    assert "mrr@5" in metrics
