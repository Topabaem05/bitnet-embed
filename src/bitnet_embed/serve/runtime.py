from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from bitnet_embed.modeling.model import EncodeConfig
from bitnet_embed.serve.config import ServiceConfig
from bitnet_embed.serve.schemas import EmbeddingData, EmbeddingRequest, EmbeddingResponse, UsageInfo
from bitnet_embed.train.factory import load_model_checkpoint


class EncodableModel(Protocol):
    def encode(self, texts: list[str], config: EncodeConfig) -> torch.Tensor: ...


class DeterministicEmbeddingBackend:
    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str], config: EncodeConfig) -> torch.Tensor:
        rows = []
        for text in texts:
            vector = torch.zeros(self.dimension, dtype=torch.float32)
            encoded = text.encode("utf-8")
            for index, value in enumerate(encoded[: self.dimension]):
                vector[index] = float(value) / 255.0
            if config.normalize and torch.linalg.norm(vector) > 0:
                vector = torch.nn.functional.normalize(vector, dim=0)
            rows.append(vector)
        return torch.stack(rows, dim=0)


@dataclass(slots=True)
class EmbeddingRuntime:
    model: EncodableModel
    model_name: str = "bitnet-embed-smoke"
    normalize_default: bool = True
    truncate_dim_default: int = 768

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        inputs = request.to_inputs()
        encode_config = EncodeConfig(
            task=request.task,
            normalize=request.normalize
            if request.normalize is not None
            else self.normalize_default,
            truncate_dim=(
                request.truncate_dim
                if request.truncate_dim is not None
                else self.truncate_dim_default
            ),
        )
        embeddings = self.model.encode(inputs, encode_config)
        data = [
            EmbeddingData(index=index, embedding=row.tolist())
            for index, row in enumerate(embeddings)
        ]
        tokens = sum(len(text.split()) for text in inputs)
        return EmbeddingResponse(
            model=self.model_name,
            data=data,
            usage=UsageInfo(input_texts=len(inputs), tokens=tokens),
        )


def build_default_runtime(config: ServiceConfig | None = None) -> EmbeddingRuntime:
    service_config = config or ServiceConfig()
    if service_config.backend == "checkpoint":
        if service_config.checkpoint_dir is None:
            raise RuntimeError("checkpoint_dir is required when backend=checkpoint")
        model = load_model_checkpoint(service_config.checkpoint_dir)
        return EmbeddingRuntime(
            model=model,
            model_name=service_config.model_name,
            normalize_default=service_config.normalize_default,
            truncate_dim_default=service_config.truncate_dim_default,
        )
    return EmbeddingRuntime(
        model=DeterministicEmbeddingBackend(dimension=service_config.truncate_dim_default),
        model_name=service_config.model_name,
        normalize_default=service_config.normalize_default,
        truncate_dim_default=service_config.truncate_dim_default,
    )
