from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class MatryoshkaLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, dims: Sequence[int]) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.dims = list(dims)

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        losses = []
        for dim in self.dims:
            losses.append(self.base_loss(query_embeddings[:, :dim], doc_embeddings[:, :dim]))
        return torch.stack(losses).mean()
