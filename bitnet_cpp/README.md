# bitnet.cpp Feasibility Track

This directory holds the Phase 2 low-bit runtime scaffold described in `bitnet_embedding_sdd_full.md`.

Current state:

- `configs/runtime/bitnet_cpp.yaml` defines the expected binary and exported package paths.
- `scripts/phase2_feasibility.py` emits a readiness report for the specialized runtime path.
- `bitnet_cpp/feasibility.json` is generated output, not source-of-truth configuration.

Next steps:

1. Provide a compiled `bitnet.cpp` binary or Python bridge.
2. Validate embedding extraction parity against the Python runtime.
3. Run low-bit latency, memory, and throughput benchmarks.
