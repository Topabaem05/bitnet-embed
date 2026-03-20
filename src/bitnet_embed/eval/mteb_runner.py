from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any


def run_mteb(model: Any, task_names: Sequence[str]) -> Any:
    try:
        mteb_module = importlib.import_module("mteb")
    except ImportError as exc:
        raise RuntimeError("mteb is required to run the MTEB benchmark") from exc
    evaluation_class = mteb_module.MTEB
    evaluation = evaluation_class(tasks=list(task_names))
    return evaluation.run(model)
