from __future__ import annotations

import argparse
import json

from bitnet_embed.eval.compare_reports import run_report_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/compare_reports.yaml")
    args = parser.parse_args()
    comparison = run_report_comparison(args.config)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
