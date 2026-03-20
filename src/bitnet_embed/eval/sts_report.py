from __future__ import annotations

import json
from pathlib import Path

from bitnet_embed.data.loaders import build_dataset_spec, load_examples
from bitnet_embed.data.schemas import ScoredPairExample
from bitnet_embed.eval.harness import evaluate_scored_pairs
from bitnet_embed.serve.config import load_service_config
from bitnet_embed.serve.runtime import build_default_runtime
from bitnet_embed.utils.io import ensure_dir, load_yaml


def run_sts_report(config_path: str) -> dict[str, float]:
    config = load_yaml(config_path)
    service_config = load_service_config(
        str(config.get("service_config", "configs/service/api.yaml"))
    )
    data_config = load_yaml(str(config.get("data_config", "configs/data/smoke_semantic.yaml")))
    eval_payload = data_config.get("eval_sets", [])[0]
    examples = [
        item
        for item in load_examples(build_dataset_spec(eval_payload))
        if isinstance(item, ScoredPairExample)
    ]
    runtime = build_default_runtime(service_config)
    metrics = evaluate_scored_pairs(runtime.model, examples)
    output_path = config.get("output_path")
    if output_path is not None:
        destination = Path(str(output_path))
        ensure_dir(destination.parent)
        destination.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics
