from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from bitnet_embed.serve.config import ServiceConfig
from bitnet_embed.serve.health import build_health_payload
from bitnet_embed.serve.runtime import EmbeddingRuntime, build_default_runtime
from bitnet_embed.serve.schemas import EmbeddingRequest, EmbeddingResponse

REQUEST_COUNTER = Counter("bitnet_embed_requests_total", "Total embedding requests")
REQUEST_LATENCY = Histogram("bitnet_embed_request_latency_seconds", "Embedding request latency")


def create_app(
    runtime: EmbeddingRuntime | None = None,
    service_config: ServiceConfig | None = None,
) -> FastAPI:
    resolved_config = service_config or ServiceConfig()
    service_runtime = runtime or build_default_runtime(resolved_config)
    app = FastAPI(title="bitnet-embed", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str | bool]:
        return build_health_payload(service_runtime.model_name)

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
        REQUEST_COUNTER.inc()
        with REQUEST_LATENCY.time():
            return service_runtime.embed(request)

    return app


app = create_app
