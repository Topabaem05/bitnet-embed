from __future__ import annotations

import argparse
import json

from bitnet_embed.eval.memory_report import run_memory_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/memory.yaml")
    args = parser.parse_args()
    metrics = run_memory_benchmark(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
