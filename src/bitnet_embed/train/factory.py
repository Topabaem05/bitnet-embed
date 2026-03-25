from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from bitnet_embed.modeling.backbone import BackboneConfig, BitNetBackbone
from bitnet_embed.modeling.lora import LoraConfigSpec, create_peft_lora_config
from bitnet_embed.modeling.model import BitNetEmbeddingModel
from bitnet_embed.modeling.smoke import build_toy_embedding_model
from bitnet_embed.utils.io import load_json


@dataclass(slots=True)
class TrainingCheckpointState:
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    global_step: int
    mode: str | None = None


_ADAPTED_WEIGHT_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _infer_global_step_from_checkpoint_dir(checkpoint_path: Path) -> int:
    match = re.search(r"step-(\d+)$", checkpoint_path.name)
    if match is None:
        return 0
    return int(match.group(1))


def load_training_checkpoint_state(checkpoint_dir: str | Path) -> TrainingCheckpointState:
    checkpoint_path = Path(checkpoint_dir)
    training_state_path = checkpoint_path / "training_state.json"
    metadata_path = checkpoint_path / "metadata.json"
    if training_state_path.exists():
        training_state = load_json(training_state_path)
        global_step_raw = training_state.get("global_step", 0)
    else:
        if metadata_path.exists():
            metadata = load_json(metadata_path)
            global_step_raw = metadata.get("global_step", 0)
        else:
            global_step_raw = 0
    global_step = int(global_step_raw) if global_step_raw is not None else 0
    if global_step == 0:
        global_step = _infer_global_step_from_checkpoint_dir(checkpoint_path)
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    return TrainingCheckpointState(
        model_state=torch.load(checkpoint_path / "model.pt", map_location="cpu", weights_only=True),
        optimizer_state=torch.load(checkpoint_path / "optimizer.pt", map_location="cpu"),
        scheduler_state=torch.load(checkpoint_path / "scheduler.pt", map_location="cpu"),
        global_step=global_step,
        mode=str(metadata.get("mode")) if metadata.get("mode") is not None else None,
    )


def load_checkpoint_weights(
    model: torch.nn.Module, checkpoint_state: TrainingCheckpointState
) -> tuple[list[str], list[str]]:
    target_state = model.state_dict()
    remapped_state: dict[str, Any] = {}
    unmatched: list[str] = []

    for source_key, value in checkpoint_state.model_state.items():
        candidates = [source_key]
        if ".base_model.model." in source_key:
            plain_key = source_key.replace(".base_model.model.", ".")
            candidates.append(plain_key)
            candidates.append(plain_key.replace(".base_layer.", "."))
        else:
            peft_key = source_key.replace(
                "backbone.backbone.", "backbone.backbone.base_model.model."
            )
            candidates.append(peft_key)
            for module_name in _ADAPTED_WEIGHT_MODULES:
                candidates.append(
                    peft_key.replace(f".{module_name}.weight", f".{module_name}.base_layer.weight")
                )

        matched_key: str | None = None
        for candidate in candidates:
            if candidate in target_state and target_state[candidate].shape == value.shape:
                matched_key = candidate
                break
        if matched_key is None:
            unmatched.append(source_key)
            continue
        remapped_state[matched_key] = value

    incompatibility = model.load_state_dict(remapped_state, strict=False)
    missing = list(incompatibility.missing_keys)
    return missing, unmatched


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


def freeze_for_lora(model: BitNetEmbeddingModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = True
    for parameter in model.projection.parameters():
        parameter.requires_grad = True


def load_model_checkpoint(checkpoint_dir: str | Path) -> BitNetEmbeddingModel:
    checkpoint_path = Path(checkpoint_dir)
    config_snapshot = load_json(checkpoint_path / "config.json")
    model = build_model(config_snapshot.get("model", {}), config_snapshot.get("lora"))
    cast(Any, model).default_max_length = int(
        config_snapshot.get("tokenization", {}).get("max_length", 256)
    )
    state_dict = torch.load(checkpoint_path / "model.pt", map_location="cpu", weights_only=True)

    for name, tensor in state_dict.items():
        if (
            torch.is_tensor(tensor)
            and tensor.numel() > 0
            and not bool(torch.isfinite(tensor).all().item())
        ):
            raise ValueError(
                f"Checkpoint contains non-finite weights in parameter '{name}'; retraining is required."
            )

    model.load_state_dict(state_dict)
    model.eval()
    return model
