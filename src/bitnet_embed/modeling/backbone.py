from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class ConfigWithHiddenSize(Protocol):
    hidden_size: int


@runtime_checkable
class BackboneWithConfig(Protocol):
    config: ConfigWithHiddenSize


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


@dataclass(slots=True)
class BackboneConfig:
    model_name: str = "microsoft/bitnet-b1.58-2B-4T-bf16"
    dtype: str | torch.dtype | None = torch.bfloat16
    use_last_k_layers: int = 1
    trust_remote_code: bool = False
    gradient_checkpointing: bool = False
    load_device: str | None = None


@dataclass(slots=True)
class BackboneFeatures:
    token_embeddings: torch.Tensor
    attention_mask: torch.Tensor


def select_hidden_states(
    last_hidden_state: torch.Tensor,
    hidden_states: tuple[torch.Tensor, ...] | None,
    use_last_k_layers: int,
) -> torch.Tensor:
    if use_last_k_layers <= 1 or not hidden_states:
        return last_hidden_state
    layers = hidden_states[-use_last_k_layers:]
    return torch.stack(layers, dim=0).mean(dim=0)


class BitNetBackbone(nn.Module):
    def __init__(
        self,
        config: BackboneConfig,
        *,
        backbone: nn.Module | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        if backbone is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("transformers is required to load a BitNet backbone") from exc

            dtype = resolve_dtype(config.dtype)
            load_device = config.load_device

            import torch

            if load_device == "cpu" or not torch.cuda.is_available():
                import os
                import torch._dynamo

                torch._dynamo.config.disable = True
                os.environ["TORCH_COMPILE_DISABLE"] = "1"

            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name, trust_remote_code=config.trust_remote_code
            )
            model_kwargs: dict[str, Any] = {
                "dtype": dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": config.trust_remote_code,
                "use_safetensors": True,
            }
            if load_device is not None:
                model_kwargs["device_map"] = load_device
            backbone = AutoModel.from_pretrained(
                config.model_name,
                **model_kwargs,
            )
            if dtype is not None:
                backbone = cast(nn.Module, backbone).to(dtype=dtype)
            if load_device is not None and not hasattr(backbone, "hf_device_map"):
                backbone = cast(nn.Module, backbone).to(load_device)

        self.backbone = cast(nn.Module, backbone)
        if self.tokenizer is not None:
            pad_token = getattr(self.tokenizer, "pad_token", None)
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if pad_token is None and eos_token is not None:
                self.tokenizer.pad_token = eos_token
        if not isinstance(self.backbone, BackboneWithConfig):
            raise RuntimeError("Backbone must expose config.hidden_size")
        self._hidden_size = int(self.backbone.config.hidden_size)

        backbone_config = getattr(self.backbone, "config", None)
        if backbone_config is not None and hasattr(backbone_config, "use_cache"):
            cast(Any, backbone_config).use_cache = False
        gradient_checkpoint = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if config.gradient_checkpointing and callable(gradient_checkpoint):
            gradient_checkpoint()

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def tokenize(
        self,
        texts: list[str],
        *,
        max_length: int,
        padding: bool = True,
        truncation: bool = True,
    ) -> dict[str, torch.Tensor]:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise RuntimeError("Tokenizer is not available on the backbone")
        encoded = tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BackboneFeatures:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.config.use_last_k_layers > 1,
            use_cache=False,
            return_dict=True,
        )
        token_embeddings = select_hidden_states(
            outputs.last_hidden_state,
            getattr(outputs, "hidden_states", None),
            self.config.use_last_k_layers,
        )
        return BackboneFeatures(token_embeddings=token_embeddings, attention_mask=attention_mask)
