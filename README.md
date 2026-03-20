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
uv run uvicorn bitnet_embed.serve.api:app --factory
```

## Layout

- `docs/sdd.md`: in-repo pointer to the source SDD
- `configs/`: model, data, train, and eval YAML configs
- `src/bitnet_embed/`: package source
- `scripts/`: runnable entrypoints for smoke, training, evaluation, and benchmark flows
- `tests/`: unit and integration smoke tests

## Notes

- The first milestone keeps BitNet quality validation separate from any future `bitnet.cpp` efficiency track.
- The training/test defaults use tiny synthetic or local data paths so the repository can smoke-test without downloading large public datasets.
