from __future__ import annotations

import argparse
import json

from bitnet_embed.eval.mteb_runner import run_mteb
from bitnet_embed.export.hf_package import load_hf_package
from bitnet_embed.modeling.smoke import build_toy_embedding_model
from bitnet_embed.train.factory import load_model_checkpoint
from bitnet_embed.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/mteb.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if config.get("package_dir") is not None:
        model = load_hf_package(str(config["package_dir"]))
    elif config.get("checkpoint_dir") is not None:
        model = load_model_checkpoint(str(config["checkpoint_dir"]))
    else:
        model = build_toy_embedding_model()
    tasks = [str(task) for task in config.get("tasks", ["STSBenchmark"])]
    result = run_mteb(
        model,
        tasks,
        output_folder=(str(config["output_folder"]) if config.get("output_folder") else None),
        model_name=str(config.get("model_name", "bitnet-embed-local")),
        revision=str(config.get("revision", "local")),
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
