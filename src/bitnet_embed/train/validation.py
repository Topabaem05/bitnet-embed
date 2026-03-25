from __future__ import annotations

import torch


def is_finite_tensor(tensor: torch.Tensor) -> bool:
    if not torch.is_tensor(tensor):
        return True
    if tensor.numel() == 0:
        return True
    return bool(torch.isfinite(tensor).all().item())


def assert_finite_loss(loss: torch.Tensor, *, step: int) -> None:
    if not is_finite_tensor(loss):
        raise RuntimeError(f"Training failed: loss is not finite at step {step} ({loss.item()})")


def assert_finite_gradients(model: torch.nn.Module, *, step: int) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if not is_finite_tensor(param.grad):
                raise RuntimeError(
                    f"Training failed: non-finite gradient detected in parameter '{name}' at step {step}"
                )


def assert_finite_parameters(model: torch.nn.Module, *, step: int) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not is_finite_tensor(param):
                raise RuntimeError(
                    f"Training failed: non-finite parameter detected in '{name}' at step {step}"
                )


def global_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1.0 / 2)
