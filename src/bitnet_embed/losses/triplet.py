from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class CosineTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        positive_distance = 1 - functional.cosine_similarity(anchor_embeddings, positive_embeddings)
        negative_distance = 1 - functional.cosine_similarity(anchor_embeddings, negative_embeddings)
        return torch.relu(positive_distance - negative_distance + self.margin).mean()
