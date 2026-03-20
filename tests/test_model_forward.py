from __future__ import annotations

import torch

from bitnet_embed.modeling.model import EncodeConfig
from bitnet_embed.modeling.smoke import build_toy_embedding_model


def test_model_forward_returns_expected_projection_dim() -> None:
    model = build_toy_embedding_model(projection_dim=16)
    assert model.tokenizer is not None
    batch = model.tokenizer(["hello", "world"], padding=True, truncation=True, max_length=16)
    embeddings = model(batch["input_ids"], batch["attention_mask"])
    assert embeddings.shape == (2, 16)


def test_model_encode_truncates_dimension() -> None:
    model = build_toy_embedding_model(projection_dim=16)
    embeddings = model.encode(["alpha", "beta"], EncodeConfig(truncate_dim=8, batch_size=2))
    assert embeddings.shape == (2, 8)
    norms = torch.linalg.norm(embeddings, dim=-1)
    assert torch.all(norms > 0)
