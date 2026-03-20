from __future__ import annotations

import argparse
import json

from bitnet_embed.train.search import run_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/search/smoke_budget.yaml")
    args = parser.parse_args()
    summary = run_search(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
