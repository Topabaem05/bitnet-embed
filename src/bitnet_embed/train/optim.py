from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class OptimizerConfig:
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    total_steps: int = 100


def build_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: OptimizerConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(config.total_steps * config.warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining_steps = max(1, config.total_steps - warmup_steps)
        progress = (step - warmup_steps) / remaining_steps
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
