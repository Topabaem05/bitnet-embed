from __future__ import annotations

import argparse
import json

from bitnet_embed.eval.sts_report import run_sts_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/sts.yaml")
    args = parser.parse_args()
    metrics = run_sts_report(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
