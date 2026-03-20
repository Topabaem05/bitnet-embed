from __future__ import annotations

from pathlib import Path

from bitnet_embed.eval.memory_report import run_memory_benchmark


def test_run_memory_benchmark_writes_report(tmp_path: Path) -> None:
    config_path = tmp_path / "memory.yaml"
    output_path = tmp_path / "memory.json"
    config_path.write_text(
        "\n".join(
            [
                "service_config: configs/service/api.yaml",
                "task: document",
                "batch_size: 4",
                "batches:",
                '  - ["hello world"]',
                '  - ["quick fox", "bright ocean"]',
                "repetitions: 2",
                f"output_path: {output_path}",
            ]
        ),
        encoding="utf-8",
    )
    metrics = run_memory_benchmark(str(config_path))
    assert output_path.exists()
    assert "startup_rss_mb" in metrics
    assert "peak_rss_mb" in metrics
