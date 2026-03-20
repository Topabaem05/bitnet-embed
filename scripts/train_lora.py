from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from bitnet_embed.train.workflow import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/lora_retrieval.yaml")
    args = parser.parse_args()
    summary = run_training(args.config, mode_override="lora")
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
