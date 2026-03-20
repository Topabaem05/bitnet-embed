from __future__ import annotations

import torch

from bitnet_embed.modeling.pooling import eos_pool, masked_mean_pool, pool_hidden_states


def test_masked_mean_pool_respects_attention_mask() -> None:
    embeddings = torch.tensor(
        [[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    pooled = masked_mean_pool(embeddings, attention_mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_eos_pool_selects_last_visible_token() -> None:
    embeddings = torch.tensor([[[1.0], [2.0], [9.0]]], dtype=torch.float32)
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    pooled = eos_pool(embeddings, attention_mask)
    assert torch.allclose(pooled, torch.tensor([[2.0]]))


def test_pool_hidden_states_raises_for_unknown_mode() -> None:
    embeddings = torch.ones((1, 2, 3))
    attention_mask = torch.ones((1, 2), dtype=torch.long)
    try:
        pool_hidden_states(embeddings, attention_mask, mode="unknown")
    except ValueError as exc:
        assert "Unsupported pooling mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown pooling mode")
