# bitnet-embed

BitNet-based embedding research stack built from `bitnet_embedding_sdd_full.md`.

## Scope

- BitNet BF16 backbone integration for embedding research
- Masked mean pooling + projection + normalization baseline
- Head-only smoke training loop with Accelerate
- FastAPI `/v1/embeddings` service with health and metrics endpoints
- Smoke evaluation harnesses for retrieval, STS, and latency

## Quickstart

```bash
uv sync --all-groups
uv run pytest
uv run python scripts/train_smoke.py --config configs/train/smoke.yaml
uv run python scripts/export_openapi.py --config configs/service/api.yaml
uv run python scripts/run_api.py --config configs/service/api.yaml
```

## Layout

- `docs/sdd.md`: in-repo pointer to the source SDD
- `configs/`: model, data, train, and eval YAML configs
- `data/smoke/`: local staged smoke datasets for pair, triplet, STS, and retrieval flows
- `src/bitnet_embed/`: package source
- `scripts/`: runnable entrypoints for smoke, training, evaluation, and benchmark flows
- `tests/`: unit and integration smoke tests

## Stage Plans

- `configs/plan/smoke_stages.yaml`: local multi-stage progression matching the SDD's early-stage flow
- `scripts/run_stage_plan.py`: sequential stage runner with JSON and Markdown plan summaries

## Service Artifacts

- `configs/service/api.yaml`: service runtime defaults
- `docs/openapi.json`: generated OpenAPI schema
- `.github/workflows/ci.yml`: lint, typecheck, test, smoke-train, and OpenAPI export checks

## Notes

- The first milestone keeps BitNet quality validation separate from any future `bitnet.cpp` efficiency track.
- The training/test defaults use tiny synthetic or local data paths so the repository can smoke-test without downloading large public datasets.
