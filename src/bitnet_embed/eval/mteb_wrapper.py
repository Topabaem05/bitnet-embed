from __future__ import annotations

import importlib
from typing import Any

import torch

from bitnet_embed.modeling.model import EncodeConfig
from bitnet_embed.modeling.prompts import TaskType


class BitNetMtebWrapper:
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
        mteb_module = importlib.import_module("mteb.models.model_meta")
        model_meta_class = mteb_module.ModelMeta
        self.mteb_model_meta = model_meta_class.create_empty(
            {
                "name": model_name,
                "revision": revision,
                "framework": ["PyTorch"],
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
    ) -> torch.Tensor:
        del task_metadata, hf_split, hf_subset, kwargs
        texts = [text for batch in inputs for text in batch["text"]]
        prompt_value = str(prompt_type).lower() if prompt_type is not None else "document"
        task: TaskType = "query" if "query" in prompt_value else "document"
        return torch.as_tensor(
            self.model.encode(texts, EncodeConfig(batch_size=self.batch_size, task=task))
        )
