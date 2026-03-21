from __future__ import annotations

import argparse
import json

from bitnet_embed.eval.finalist_confirmation import run_finalist_confirmation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/finalist_confirmation.yaml")
    args = parser.parse_args()
    summary = run_finalist_confirmation(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
