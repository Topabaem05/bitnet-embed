from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from common import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/smoke.yaml")
    args = parser.parse_args()
    summary = run_training(args.config)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
