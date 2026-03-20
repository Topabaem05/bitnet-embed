from __future__ import annotations

import argparse
import json

from bitnet_embed.bitnet_cpp_bridge import run_bitnet_cpp_feasibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/runtime/bitnet_cpp.yaml")
    args = parser.parse_args()
    report = run_bitnet_cpp_feasibility(args.config)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
