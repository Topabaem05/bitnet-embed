from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional


def cosine_scores(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return functional.cosine_similarity(left, right)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman_correlation(left: torch.Tensor, right: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = cosine_scores(left, right).detach().cpu().numpy()
    gold = labels.detach().cpu().numpy()
    pred_rank = _rank(predictions)
    gold_rank = _rank(gold)
    pred_centered = pred_rank - pred_rank.mean()
    gold_centered = gold_rank - gold_rank.mean()
    denominator = np.linalg.norm(pred_centered) * np.linalg.norm(gold_centered)
    if denominator == 0:
        return 0.0
    return float(np.dot(pred_centered, gold_centered) / denominator)
