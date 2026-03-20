from __future__ import annotations

from pathlib import Path

from bitnet_embed.eval.report_bundle import run_report_bundle
from bitnet_embed.utils.io import dump_json


def test_run_report_bundle_writes_json_and_markdown(tmp_path: Path) -> None:
    stage_plan = tmp_path / "plan.json"
    latency = tmp_path / "latency.json"
    ann = tmp_path / "ann.json"
    package = tmp_path / "package.json"
    dump_json(
        stage_plan,
        {
            "plan_name": "demo-plan",
            "stage_count": 2,
            "stages": [
                {"name": "stage-a", "avg_loss": 0.4},
                {"name": "stage-b", "avg_loss": 0.2},
            ],
        },
    )
    dump_json(latency, {"p50_latency": 0.01, "throughput": 100.0})
    dump_json(ann, {"ann_recall@5": 1.0})
    dump_json(
        package,
        {
            "package_name": "demo-package",
            "format": "bitnet-embed-hf-style",
            "metrics": {"sts_spearman": 0.9},
        },
    )
    config_path = tmp_path / "report.yaml"
    config_path.write_text(
        "\n".join(
            [
                "report_name: demo-report",
                f"output_dir: {tmp_path / 'out'}",
                f"stage_plan: {stage_plan}",
                f"latency_report: {latency}",
                f"ann_report: {ann}",
                f"package_manifest: {package}",
            ]
        ),
        encoding="utf-8",
    )
    bundle = run_report_bundle(str(config_path))
    assert bundle["stage_summary"]["best_stage_by_loss"] == "stage-b"
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "summary.md").exists()
