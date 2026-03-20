from __future__ import annotations

from bitnet_embed.serve.api import create_app


def test_openapi_schema_contains_embeddings_route() -> None:
    schema = create_app().openapi()
    assert "/v1/embeddings" in schema["paths"]
    assert "EmbeddingResponse" in schema["components"]["schemas"]
