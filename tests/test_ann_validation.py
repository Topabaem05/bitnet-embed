from __future__ import annotations

from pathlib import Path

import torch

from bitnet_embed.data.schemas import QueryDocumentExample
from bitnet_embed.eval.ann import InMemoryAnnIndex, validate_ann
from bitnet_embed.eval.ann_report import run_ann_validation
from bitnet_embed.modeling.smoke import build_toy_embedding_model


def test_in_memory_ann_index_returns_ranked_hits() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    queries = torch.tensor([[0.9, 0.1]], dtype=torch.float32)
    hits = InMemoryAnnIndex(embeddings).search(queries, top_k=2)
    assert hits[0][0].index == 0
    assert hits[0][0].score > hits[0][1].score


def test_validate_ann_returns_ann_metrics() -> None:
    model = build_toy_embedding_model()
    examples = [
        QueryDocumentExample(query="fast car", document="quick automobile", label=1),
        QueryDocumentExample(query="fast car", document="stock report", label=0),
    ]
    metrics = validate_ann(model, examples, top_k=2)
    assert "ann_recall@1" in metrics


def test_run_ann_validation_writes_report(tmp_path: Path) -> None:
    config_path = tmp_path / "ann.yaml"
    output_path = tmp_path / "ann.json"
    config_path.write_text(
        "\n".join(
            [
                "service_config: configs/service/api.yaml",
                "data_config: configs/data/smoke_retrieval.yaml",
                "top_k: 5",
                f"output_path: {output_path}",
            ]
        ),
        encoding="utf-8",
    )
    metrics = run_ann_validation(str(config_path))
    assert output_path.exists()
    assert "ann_recall@1" in metrics
