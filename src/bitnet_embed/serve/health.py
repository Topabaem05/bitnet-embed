from __future__ import annotations


def build_health_payload(model_name: str, ready: bool = True) -> dict[str, str | bool]:
    return {"status": "ok" if ready else "degraded", "ready": ready, "model": model_name}
