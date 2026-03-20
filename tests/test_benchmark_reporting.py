from __future__ import annotations

from pathlib import Path

from bitnet_embed.eval.latency_report import run_benchmark


def test_run_benchmark_writes_report(tmp_path: Path) -> None:
    config_path = tmp_path / "latency.yaml"
    output_path = tmp_path / "latency.json"
    config_path.write_text(
        "\n".join(
            [
                "service_config: configs/service/api.yaml",
                "task: document",
                "batch_size: 4",
                "normalize: true",
                "truncate_dim: 32",
                "batches:",
                '  - ["hello world"]',
                '  - ["quick fox", "bright ocean"]',
                "repetitions: 2",
                f"output_path: {output_path}",
            ]
        ),
        encoding="utf-8",
    )
    metrics = run_benchmark(str(config_path))
    assert output_path.exists()
    assert "startup_p50" in metrics
    assert "p95_latency" in metrics
