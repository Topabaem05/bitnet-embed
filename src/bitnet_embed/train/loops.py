from __future__ import annotations

from typing import Any

import torch


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {
                nested_key: nested_value.to(device) if hasattr(nested_value, "to") else nested_value
                for nested_key, nested_value in value.items()
            }
        elif hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def encode_pair_batch(
    model: torch.nn.Module, batch: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_batch = batch["anchor"]
    positive_batch = batch["positive"]
    anchor_embeddings = model(anchor_batch["input_ids"], anchor_batch["attention_mask"])
    positive_embeddings = model(positive_batch["input_ids"], positive_batch["attention_mask"])
    return anchor_embeddings, positive_embeddings


def encode_triplet_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor_batch = batch["anchor"]
    positive_batch = batch["positive"]
    negative_batch = batch["negative"]
    anchor_embeddings = model(anchor_batch["input_ids"], anchor_batch["attention_mask"])
    positive_embeddings = model(positive_batch["input_ids"], positive_batch["attention_mask"])
    negative_embeddings = model(negative_batch["input_ids"], negative_batch["attention_mask"])
    return anchor_embeddings, positive_embeddings, negative_embeddings
