from __future__ import annotations

from pathlib import Path

from bitnet_embed.eval.sts_report import run_sts_report


def test_run_sts_report_writes_report(tmp_path: Path) -> None:
    config_path = tmp_path / "sts.yaml"
    output_path = tmp_path / "sts.json"
    config_path.write_text(
        "\n".join(
            [
                "service_config: configs/service/api.yaml",
                "data_config: configs/data/smoke_semantic.yaml",
                f"output_path: {output_path}",
            ]
        ),
        encoding="utf-8",
    )
    metrics = run_sts_report(str(config_path))
    assert output_path.exists()
    assert "sts_spearman" in metrics
