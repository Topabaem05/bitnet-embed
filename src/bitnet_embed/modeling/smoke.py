from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from bitnet_embed.modeling.model import BitNetEmbeddingModel


class ToyTokenizer:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def _encode_text(self, text: str, max_length: int) -> list[int]:
        payload = [2 + (byte % 253) for byte in text.encode("utf-8")[: max(1, max_length - 1)]]
        payload.append(self.eos_token_id)
        return payload[:max_length]

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        max_length_value = kwargs.get("max_length", 64)
        max_length = max_length_value if isinstance(max_length_value, int) else 64
        encoded = [self._encode_text(text, max_length) for text in texts]
        width = max(len(row) for row in encoded)
        padded = [row + [self.pad_token_id] * (width - len(row)) for row in encoded]
        attention_mask = [[1] * len(row) + [0] * (width - len(row)) for row in encoded]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class ToyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 64, vocab_size: int = 256) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, use_cache=False)
        self.tokenizer = ToyTokenizer()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        hidden = self.projection(self.embedding(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)


def build_toy_embedding_model(
    projection_dim: int = 32,
    pooling: str = "masked_mean",
    normalize: bool = True,
) -> BitNetEmbeddingModel:
    backbone = ToyBackbone()
    return BitNetEmbeddingModel(
        backbone,
        projection_dim=projection_dim,
        pooling=pooling,
        normalize=normalize,
    )
