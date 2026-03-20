from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class SymmetricInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        scores = query_embeddings @ doc_embeddings.T / self.temperature
        labels = torch.arange(scores.size(0), device=scores.device)
        return 0.5 * (
            functional.cross_entropy(scores, labels) + functional.cross_entropy(scores.T, labels)
        )
