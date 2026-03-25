from __future__ import annotations

import torch
import torch.nn as nn

from bitnet_embed.modeling.backbone import BackboneFeatures
from bitnet_embed.modeling.model import BitNetEmbeddingModel, EncodeConfig
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


class _DummyTokenizer:
    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        batch_size = len(texts)
        return {
            "input_ids": torch.ones((batch_size, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 2), dtype=torch.long),
        }


class _DummyBackbone(nn.Module):
    hidden_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = _DummyTokenizer()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BackboneFeatures:
        batch_size, sequence_length = input_ids.shape
        features = torch.ones((batch_size, sequence_length, self.hidden_size), dtype=torch.bfloat16)
        return BackboneFeatures(token_embeddings=features, attention_mask=attention_mask)


def test_model_encode_casts_embeddings_to_projection_dtype() -> None:
    model = BitNetEmbeddingModel(_DummyBackbone(), projection_dim=4)
    embeddings = model.encode(["alpha"], EncodeConfig(batch_size=1, normalize=False))
    assert embeddings.shape == (1, 4)
    assert embeddings.dtype == torch.float32
