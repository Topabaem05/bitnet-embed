from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from bitnet_embed.eval.benchmark import measure_latency, measure_startup
from bitnet_embed.modeling.model import EncodeConfig
from bitnet_embed.modeling.prompts import TaskType
from bitnet_embed.serve.config import load_service_config
from bitnet_embed.serve.runtime import build_default_runtime
from bitnet_embed.utils.io import ensure_dir, load_yaml


def run_benchmark(config_path: str) -> dict[str, float]:
    config = load_yaml(config_path)
    service_config_path = str(config.get("service_config", "configs/service/api.yaml"))
    service_config = load_service_config(service_config_path)
    task = cast(TaskType, str(config.get("task", "document")))
    normalize = config.get("normalize")
    truncate_dim = config.get("truncate_dim")
    batch_size = int(config.get("batch_size", 8))
    repetitions = int(config.get("repetitions", 3))
    startup_metrics = measure_startup(lambda: build_default_runtime(service_config), repetitions)
    runtime = build_default_runtime(service_config)
    encode_config = EncodeConfig(
        batch_size=batch_size,
        task=task,
        normalize=service_config.normalize_default if normalize is None else bool(normalize),
        truncate_dim=(
            service_config.truncate_dim_default if truncate_dim is None else int(truncate_dim)
        ),
    )
    batches = [[str(item) for item in batch] for batch in config.get("batches", [])]
    metrics = measure_latency(
        lambda texts: runtime.model.encode(texts, encode_config), batches, repetitions
    )
    result = startup_metrics | metrics
    output_path = config.get("output_path")
    if output_path is not None:
        destination = Path(str(output_path))
        ensure_dir(destination.parent)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
