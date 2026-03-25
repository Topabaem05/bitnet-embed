from __future__ import annotations

import argparse
import json

from bitnet_embed.train.autoresearch import run_autoresearch_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/search/autoresearch_msmarco_retrieval.yaml",
    )
    args = parser.parse_args()
    result = run_autoresearch_search(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
