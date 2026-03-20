from __future__ import annotations

import argparse
import json

from bitnet_embed.eval.mteb_runner import run_mteb
from bitnet_embed.modeling.smoke import build_toy_embedding_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", default=["STSBenchmark"])
    args = parser.parse_args()
    model = build_toy_embedding_model()
    result = run_mteb(model, args.task)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
