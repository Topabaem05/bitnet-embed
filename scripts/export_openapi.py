from __future__ import annotations

import argparse
import json
from pathlib import Path

from bitnet_embed.serve.api import create_app
from bitnet_embed.serve.config import load_service_config
from bitnet_embed.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/service/api.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    service_config = load_service_config(args.config)
    output_path = Path(args.output or service_config.openapi_path)
    ensure_dir(output_path.parent)
    app = create_app(service_config=service_config)
    output_path.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
