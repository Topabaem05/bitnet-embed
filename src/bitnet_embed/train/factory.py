from __future__ import annotations

from typing import Any

from bitnet_embed.modeling.backbone import BackboneConfig, BitNetBackbone
from bitnet_embed.modeling.lora import LoraConfigSpec, create_peft_lora_config
from bitnet_embed.modeling.model import BitNetEmbeddingModel
from bitnet_embed.modeling.smoke import build_toy_embedding_model


def build_model(
    model_config: dict[str, Any], lora_config: dict[str, Any] | None = None
) -> BitNetEmbeddingModel:
    backend = str(model_config.get("backend", "huggingface"))
    if backend == "toy":
        if lora_config and bool(lora_config.get("enabled", False)):
            raise RuntimeError("LoRA is not supported for the toy backend")
        return build_toy_embedding_model(
            projection_dim=int(model_config.get("projection_dim", 32)),
            pooling=str(model_config.get("pooling", "masked_mean")),
            normalize=bool(model_config.get("normalize", True)),
        )

    backbone = BitNetBackbone(
        BackboneConfig(
            model_name=str(model_config.get("backbone_name", "microsoft/bitnet-b1.58-2B-4T-bf16")),
            dtype=model_config.get("dtype", "bfloat16"),
            use_last_k_layers=int(model_config.get("use_last_k_layers", 1)),
            gradient_checkpointing=bool(model_config.get("gradient_checkpointing", False)),
        )
    )
    model = BitNetEmbeddingModel(
        backbone,
        projection_dim=int(model_config.get("projection_dim", 768)),
        pooling=str(model_config.get("pooling", "masked_mean")),
        normalize=bool(model_config.get("normalize", True)),
    )

    if lora_config and bool(lora_config.get("enabled", False)):
        try:
            from peft import get_peft_model
            from transformers import PreTrainedModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required for LoRA mode") from exc
        target_modules = list(lora_config.get("target_modules", []))
        spec = LoraConfigSpec(
            enabled=True,
            r=int(lora_config.get("r", 16)),
            alpha=int(lora_config.get("alpha", 32)),
            dropout=float(lora_config.get("dropout", 0.05)),
            bias=lora_config.get("bias", "none"),
            target_modules=target_modules,
        )
        if not isinstance(backbone.backbone, PreTrainedModel):
            raise RuntimeError("LoRA requires a Hugging Face PreTrainedModel backbone")
        backbone.backbone = get_peft_model(backbone.backbone, create_peft_lora_config(spec))
    return model


def freeze_backbone(model: BitNetEmbeddingModel) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.projection.parameters():
        parameter.requires_grad = True


def unfreeze_all(model: BitNetEmbeddingModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
