from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(slots=True)
class ProjectionConfig:
    hidden_size: int
    embedding_dim: int = 768
    dropout: float = 0.0
    use_layer_norm: bool = False
    intermediate_dim: int | None = None


class ProjectionHead(nn.Module):
    def __init__(self, config: ProjectionConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = config.hidden_size
        if config.intermediate_dim is not None:
            layers.extend(
                [
                    nn.Linear(input_dim, config.intermediate_dim, bias=False),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            input_dim = config.intermediate_dim
        layers.append(nn.Linear(input_dim, config.embedding_dim, bias=False))
        if config.use_layer_norm:
            layers.append(nn.LayerNorm(config.embedding_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.network(embeddings))


def truncate_embeddings(embeddings: torch.Tensor, truncate_dim: int | None) -> torch.Tensor:
    if truncate_dim is None:
        return embeddings
    if truncate_dim <= 0 or truncate_dim > embeddings.size(-1):
        raise ValueError(f"truncate_dim must be in [1, {embeddings.size(-1)}], got {truncate_dim}")
    return embeddings[..., :truncate_dim]
