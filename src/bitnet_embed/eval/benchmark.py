from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import torch


def measure_latency(
    encode_fn: Callable[[list[str]], torch.Tensor],
    batches: Sequence[list[str]],
    repetitions: int = 3,
) -> dict[str, float]:
    timings: list[float] = []
    items = 0
    for _ in range(repetitions):
        for batch in batches:
            start = time.perf_counter()
            encode_fn(list(batch))
            timings.append(time.perf_counter() - start)
            items += len(batch)
    if not timings:
        return {"p50_latency": 0.0, "p95_latency": 0.0, "throughput": 0.0}
    ordered = sorted(timings)
    p50_index = min(len(ordered) - 1, int(0.50 * (len(ordered) - 1)))
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    total_time = sum(timings)
    return {
        "p50_latency": ordered[p50_index],
        "p95_latency": ordered[p95_index],
        "throughput": items / total_time if total_time > 0 else 0.0,
    }
