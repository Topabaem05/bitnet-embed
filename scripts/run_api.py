from __future__ import annotations

import argparse

import uvicorn

from bitnet_embed.serve.api import create_app
from bitnet_embed.serve.config import load_service_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/service/api.yaml")
    args = parser.parse_args()

    service_config = load_service_config(args.config)
    app = create_app(service_config=service_config)
    uvicorn.run(app, host=service_config.host, port=service_config.port)


if __name__ == "__main__":
    main()
