from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bitnet_embed.utils.io import load_yaml


@dataclass(slots=True)
class ServiceConfig:
    model_name: str = "bitnet-embed-smoke"
    host: str = "0.0.0.0"
    port: int = 8000
    normalize_default: bool = True
    truncate_dim_default: int = 768
    backend: str = "deterministic"
    openapi_path: str = "docs/openapi.json"


def load_service_config(path: str | Path) -> ServiceConfig:
    payload = load_yaml(path)
    return ServiceConfig(
        model_name=str(payload.get("model_name", "bitnet-embed-smoke")),
        host=str(payload.get("host", "0.0.0.0")),
        port=int(payload.get("port", 8000)),
        normalize_default=bool(payload.get("normalize_default", True)),
        truncate_dim_default=int(payload.get("truncate_dim_default", 768)),
        backend=str(payload.get("backend", "deterministic")),
        openapi_path=str(payload.get("openapi_path", "docs/openapi.json")),
    )
