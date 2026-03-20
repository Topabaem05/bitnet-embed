from __future__ import annotations

import argparse
import json

from bitnet_embed.train.plan import run_stage_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/plan/smoke_stages.yaml")
    args = parser.parse_args()
    summary = run_stage_plan(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
