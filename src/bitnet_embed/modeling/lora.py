from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BiasMode = Literal["none", "all", "lora_only"]

DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(slots=True)
class LoraConfigSpec:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: BiasMode = "none"
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES))


def module_name_index(module_names: list[str]) -> set[str]:
    return {name.rsplit(".", maxsplit=1)[-1] for name in module_names}


def resolve_lora_target_modules(module_names: list[str]) -> list[str]:
    available = module_name_index(module_names)
    return [name for name in DEFAULT_LORA_TARGET_MODULES if name in available]


def create_peft_lora_config(spec: LoraConfigSpec) -> Any:
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise RuntimeError("PEFT is required to create a LoRA config") from exc

    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        bias=spec.bias,
        target_modules=spec.target_modules,
    )
