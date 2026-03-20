from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bitnet_embed.data.loaders import build_dataset_spec, load_examples
from bitnet_embed.data.schemas import LabeledTextExample
from bitnet_embed.eval.clustering import evaluate_kmeans
from bitnet_embed.modeling.model import EncodeConfig
from bitnet_embed.serve.config import load_service_config
from bitnet_embed.serve.runtime import build_default_runtime
from bitnet_embed.utils.io import ensure_dir, load_yaml


def run_clustering_report(config_path: str) -> dict[str, float]:
    config = load_yaml(config_path)
    service_config = load_service_config(
        str(config.get("service_config", "configs/service/api.yaml"))
    )
    data_spec = build_dataset_spec(
        {
            "local_path": str(config.get("data_path", "data/smoke/clustering.jsonl")),
            "format": "labeled_text",
        }
    )
    examples = [item for item in load_examples(data_spec) if isinstance(item, LabeledTextExample)]
    runtime = build_default_runtime(service_config)
    embeddings = runtime.model.encode(
        [example.text for example in examples],
        EncodeConfig(task="document", batch_size=max(1, len(examples))),
    )
    metrics = evaluate_kmeans(np.asarray(embeddings), [example.label for example in examples])
    output_path = config.get("output_path")
    if output_path is not None:
        destination = Path(str(output_path))
        ensure_dir(destination.parent)
        destination.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics
