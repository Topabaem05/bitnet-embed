from __future__ import annotations

from typing import Any

import numpy as np
import torch
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from numpy.typing import NDArray

from bitnet_embed.modeling.model import EncodeConfig
from bitnet_embed.modeling.prompts import TaskType


class BitNetMtebWrapper(AbsEncoder):
    def __init__(
        self,
        model: Any,
        *,
        model_name: str = "bitnet-embed-local",
        revision: str = "local",
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        projection = getattr(model, "projection", None)
        projection_network = getattr(projection, "network", None)
        embed_dim = None
        if projection_network is not None and len(projection_network) > 0:
            output_layer = projection_network[-1]
            embed_dim = getattr(output_layer, "out_features", None)
        resolved_name = model_name if "/" in model_name else f"custom/{model_name}"
        self.mteb_model_meta = ModelMeta.create_empty(
            {
                "name": resolved_name,
                "revision": revision,
                "framework": ["PyTorch"],
                "embed_dim": embed_dim,
                "similarity_fn_name": ScoringFunction.COSINE,
            }
        )

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        del task_metadata, hf_split, hf_subset
        texts = [text for batch in inputs for text in batch["text"]]
        resolved_batch_size = int(kwargs.get("batch_size", self.batch_size))
        resolved_max_length = int(
            getattr(self.model, "default_max_length", EncodeConfig().max_length)
        )
        prompt_value = getattr(prompt_type, "value", prompt_type)
        normalized_prompt = str(prompt_value).lower() if prompt_value is not None else "document"
        task: TaskType = "query" if "query" in normalized_prompt else "document"
        embeddings = self.model.encode(
            texts,
            EncodeConfig(
                batch_size=resolved_batch_size,
                task=task,
                normalize=bool(getattr(self.model, "normalize", True)),
                max_length=resolved_max_length,
            ),
        )
        return torch.as_tensor(embeddings).detach().cpu().float().numpy()
