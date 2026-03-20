from __future__ import annotations

import argparse
import json

from bitnet_embed.export.hf_package import export_hf_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--package-name", default=None)
    args = parser.parse_args()
    manifest = export_hf_package(
        args.checkpoint,
        args.output,
        package_name=args.package_name,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
