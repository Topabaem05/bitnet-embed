from __future__ import annotations

from pathlib import Path

from bitnet_embed.serve.config import load_service_config


def test_load_service_config_reads_yaml_defaults() -> None:
    config = load_service_config(Path("configs/service/api.yaml"))
    assert config.model_name == "bitnet-embed-smoke"
    assert config.port == 8000
    assert config.openapi_path == "docs/openapi.json"
