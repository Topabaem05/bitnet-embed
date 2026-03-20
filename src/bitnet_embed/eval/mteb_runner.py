from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from bitnet_embed.eval.mteb_wrapper import BitNetMtebWrapper


def run_mteb(
    model: Any,
    task_names: Sequence[str],
    *,
    output_folder: str | None = None,
    model_name: str = "bitnet-embed-local",
    revision: str = "local",
) -> Any:
    try:
        mteb_module = importlib.import_module("mteb")
    except ImportError as exc:
        raise RuntimeError("mteb is required to run the MTEB benchmark") from exc
    evaluation_class = mteb_module.MTEB
    evaluation = evaluation_class(tasks=list(task_names))
    wrapped_model = (
        model
        if hasattr(model, "mteb_model_meta")
        else BitNetMtebWrapper(
            model,
            model_name=model_name,
            revision=revision,
        )
    )
    if output_folder is None:
        return evaluation.run(wrapped_model)
    return evaluation.run(wrapped_model, output_folder=output_folder)
