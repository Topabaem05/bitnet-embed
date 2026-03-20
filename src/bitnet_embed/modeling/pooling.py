from __future__ import annotations

import torch


def masked_mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def eos_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_indices = attention_mask.sum(dim=1).clamp(min=1) - 1
    batch_indices = torch.arange(token_embeddings.size(0), device=token_embeddings.device)
    return token_embeddings[batch_indices, last_indices]


def last_token_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return eos_pool(token_embeddings, attention_mask)


def pool_hidden_states(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str = "masked_mean",
) -> torch.Tensor:
    if mode == "masked_mean":
        return masked_mean_pool(token_embeddings, attention_mask)
    if mode == "eos":
        return eos_pool(token_embeddings, attention_mask)
    if mode == "last_token":
        return last_token_pool(token_embeddings, attention_mask)
    raise ValueError(f"Unsupported pooling mode: {mode}")
