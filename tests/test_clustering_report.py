from __future__ import annotations

from pathlib import Path

from bitnet_embed.eval.clustering_report import run_clustering_report


def test_run_clustering_report_writes_report(tmp_path: Path) -> None:
    config_path = tmp_path / "clustering.yaml"
    output_path = tmp_path / "clustering.json"
    config_path.write_text(
        "\n".join(
            [
                "service_config: configs/service/api.yaml",
                "data_path: data/smoke/clustering.jsonl",
                f"output_path: {output_path}",
            ]
        ),
        encoding="utf-8",
    )
    metrics = run_clustering_report(str(config_path))
    assert output_path.exists()
    assert "nmi" in metrics
    assert "ari" in metrics
