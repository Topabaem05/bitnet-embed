from __future__ import annotations

from pathlib import Path

from bitnet_embed.eval.compare_reports import run_report_comparison
from bitnet_embed.utils.io import dump_json


def test_run_report_comparison_writes_outputs(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    dump_json(
        report_path,
        {
            "stage_summary": {"best_stage_by_loss": "stage-a", "best_stage_loss": 0.2},
            "latency": {"p50_latency": 0.01},
            "ann": {"ann_recall@5": 1.0},
            "package": {"package_name": "demo", "metrics": {"sts_spearman": 0.9}},
        },
    )
    config_path = tmp_path / "compare.yaml"
    config_path.write_text(
        "\n".join(
            [
                "comparison_name: demo-compare",
                f"output_dir: {tmp_path / 'out'}",
                "reports:",
                "  - name: smoke",
                f"    path: {report_path}",
            ]
        ),
        encoding="utf-8",
    )
    comparison = run_report_comparison(str(config_path))
    assert comparison["reports"][0]["name"] == "smoke"
    assert (tmp_path / "out" / "comparison.json").exists()
    assert (tmp_path / "out" / "comparison.md").exists()
