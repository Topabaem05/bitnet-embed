from __future__ import annotations

import argparse
import json

from bitnet_embed.eval.report_bundle import run_report_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/report_bundle.yaml")
    args = parser.parse_args()
    bundle = run_report_bundle(args.config)
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
