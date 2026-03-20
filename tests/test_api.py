from __future__ import annotations

from fastapi.testclient import TestClient

from bitnet_embed.serve.api import create_app


def test_embeddings_api_returns_openai_style_shape() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/embeddings",
        json={"input": ["hello world", "fast car"], "task": "document", "normalize": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "bitnet-embed-smoke"
    assert len(payload["data"]) == 2
    assert payload["usage"]["input_texts"] == 2


def test_metrics_and_health_endpoints_exist() -> None:
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200
