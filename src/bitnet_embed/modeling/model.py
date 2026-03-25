from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as functional

from bitnet_embed.modeling.backbone import BackboneFeatures, BitNetBackbone
from bitnet_embed.modeling.pooling import pool_hidden_states
from bitnet_embed.modeling.projection import ProjectionConfig, ProjectionHead, truncate_embeddings
from bitnet_embed.modeling.prompts import PromptConfig, TaskType, format_batch


class TokenizerProtocol(Protocol):
    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]: ...


@runtime_checkable
class HiddenSizeModule(Protocol):
    hidden_size: int


@dataclass(slots=True)
class EncodeConfig:
    batch_size: int = 32
    normalize: bool = True
    task: TaskType = "document"
    truncate_dim: int | None = None
    max_length: int = 256
    instruction: str | None = None


def _infer_tokenizer(module: nn.Module) -> TokenizerProtocol | None:
    tokenizer = getattr(module, "tokenizer", None)
    if tokenizer is None:
        return None
    return cast(TokenizerProtocol, tokenizer)


class BitNetEmbeddingModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        projection_dim: int = 768,
        pooling: str = "masked_mean",
        normalize: bool = True,
        prompt_config: PromptConfig | None = None,
        projection_head: ProjectionHead | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.normalize = normalize
        self.prompt_config = prompt_config or PromptConfig()
        if not isinstance(backbone, HiddenSizeModule):
            raise RuntimeError("Backbone must expose a hidden_size attribute")
        hidden_size = int(backbone.hidden_size)
        self.projection = projection_head or ProjectionHead(
            ProjectionConfig(hidden_size=hidden_size, embedding_dim=projection_dim)
        )
        self.tokenizer = _infer_tokenizer(backbone)

    def _project_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        projection_param = next(self.projection.parameters(), None)
        if projection_param is not None:
            embeddings = embeddings.to(
                device=projection_param.device,
                dtype=projection_param.dtype,
            )
        projected = self.projection(embeddings)
        return torch.as_tensor(projected)

    def forward_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> BackboneFeatures:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, BackboneFeatures):
            return outputs
        if hasattr(outputs, "token_embeddings") and hasattr(outputs, "attention_mask"):
            return BackboneFeatures(
                token_embeddings=outputs.token_embeddings,
                attention_mask=outputs.attention_mask,
            )
        if hasattr(outputs, "last_hidden_state"):
            return BackboneFeatures(
                token_embeddings=outputs.last_hidden_state,
                attention_mask=attention_mask,
            )
        raise TypeError(f"Unsupported backbone output type: {type(outputs)!r}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = pool_hidden_states(
            features.token_embeddings,
            features.attention_mask,
            mode=self.pooling,
        )
        sentence_embeddings = self._project_embeddings(sentence_embeddings)
        if self.normalize:
            sentence_embeddings = functional.normalize(sentence_embeddings, p=2, dim=-1)
        return sentence_embeddings

    @torch.inference_mode()
    def encode(self, texts: list[str], config: EncodeConfig | None = None) -> torch.Tensor:
        encode_config = config or EncodeConfig(normalize=self.normalize)
        if self.tokenizer is None and not isinstance(self.backbone, BitNetBackbone):
            raise RuntimeError("Tokenizer is required for encode()")

        tokenizer = self.tokenizer
        if tokenizer is None and isinstance(self.backbone, BitNetBackbone):
            tokenizer = cast(TokenizerProtocol, self.backbone.tokenizer)
        if tokenizer is None:
            raise RuntimeError("Tokenizer is required for encode()")

        formatted = format_batch(
            texts,
            task=encode_config.task,
            prompt_config=self.prompt_config,
            instruction=encode_config.instruction,
        )
        device = next(self.parameters()).device
        outputs: list[torch.Tensor] = []
        for start in range(0, len(formatted), encode_config.batch_size):
            batch_texts = formatted[start : start + encode_config.batch_size]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=encode_config.max_length,
                return_tensors="pt",
            )
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            features = self.forward_features(batch["input_ids"], batch["attention_mask"])
            embeddings = pool_hidden_states(
                features.token_embeddings,
                features.attention_mask,
                mode=self.pooling,
            )
            embeddings = self._project_embeddings(embeddings)
            if encode_config.normalize:
                embeddings = functional.normalize(embeddings, p=2, dim=-1)
            embeddings = truncate_embeddings(embeddings, encode_config.truncate_dim)
            outputs.append(embeddings.cpu())
        return torch.cat(outputs, dim=0)
