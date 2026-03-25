from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from bitnet_embed.eval.mteb_wrapper import BitNetMtebWrapper
from bitnet_embed.modeling.smoke import build_toy_embedding_model


def test_mteb_wrapper_encodes_query_batches() -> None:
    model = build_toy_embedding_model(projection_dim=8)
    cast(Any, model).default_max_length = 128
    wrapper = BitNetMtebWrapper(model, model_name="toy-mteb", revision="local")
    inputs = [{"text": ["alpha", "beta"]}]
    task_metadata = SimpleNamespace(name="ToyTask")
    embeddings = wrapper.encode(
        inputs,
        task_metadata=task_metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type="query",
    )
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 8)
    assert wrapper.mteb_model_meta is not None
    assert wrapper.mteb_model_meta.name == "custom/toy-mteb"


def test_mteb_wrapper_accepts_string_prompt_type() -> None:
    model = build_toy_embedding_model(projection_dim=8)
    cast(Any, model).default_max_length = 64
    wrapper = BitNetMtebWrapper(model, model_name="toy-mteb", revision="local", batch_size=1)

    embeddings = wrapper.encode(
        [{"text": ["alpha"]}],
        task_metadata=SimpleNamespace(name="ToyTask"),
        hf_split="test",
        hf_subset="default",
        prompt_type="query",
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 8)
