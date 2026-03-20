from __future__ import annotations

import json
from pathlib import Path

import psutil
import torch

from bitnet_embed.modeling.model import EncodeConfig
from bitnet_embed.serve.config import load_service_config
from bitnet_embed.serve.runtime import build_default_runtime
from bitnet_embed.utils.io import ensure_dir, load_yaml


def bytes_to_mb(value: int) -> float:
    return float(value) / (1024.0 * 1024.0)


def run_memory_benchmark(config_path: str) -> dict[str, float]:
    config = load_yaml(config_path)
    service_config = load_service_config(
        str(config.get("service_config", "configs/service/api.yaml"))
    )
    repetitions = int(config.get("repetitions", 3))
    batch_size = int(config.get("batch_size", 8))
    task = str(config.get("task", "document"))
    process = psutil.Process()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        startup_cuda_mb = bytes_to_mb(torch.cuda.memory_allocated())
    else:
        startup_cuda_mb = 0.0
    startup_rss_mb = bytes_to_mb(process.memory_info().rss)
    runtime = build_default_runtime(service_config)
    peak_rss_mb = startup_rss_mb
    encode_config = EncodeConfig(
        batch_size=batch_size,
        task="query" if task == "query" else "document",
        normalize=service_config.normalize_default,
        truncate_dim=service_config.truncate_dim_default,
    )
    batches = [[str(item) for item in batch] for batch in config.get("batches", [])]
    for _ in range(repetitions):
        for batch in batches:
            runtime.model.encode(batch, encode_config)
            peak_rss_mb = max(peak_rss_mb, bytes_to_mb(process.memory_info().rss))
    peak_cuda_mb = (
        bytes_to_mb(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
    )
    result = {
        "startup_rss_mb": startup_rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "startup_cuda_mb": startup_cuda_mb,
        "peak_cuda_mb": peak_cuda_mb,
    }
    output_path = config.get("output_path")
    if output_path is not None:
        destination = Path(str(output_path))
        ensure_dir(destination.parent)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
