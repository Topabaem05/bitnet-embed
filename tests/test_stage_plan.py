from __future__ import annotations

from pathlib import Path

from bitnet_embed.train.plan import run_stage_plan


def test_run_stage_plan_writes_json_and_markdown(tmp_path: Path) -> None:
    output_root = tmp_path / "stage-plan"
    config_path = tmp_path / "plan.yaml"
    config_path.write_text(
        "\n".join(
            [
                "plan_name: tmp_stage_plan",
                f"output_root: {output_root}",
                "stages:",
                "  - name: semantic",
                "    train_config: configs/train/smoke.yaml",
                "  - name: retrieval",
                "    train_config: configs/train/smoke_retrieval.yaml",
                "    mode_override: full_ft",
            ]
        ),
        encoding="utf-8",
    )
    summary = run_stage_plan(str(config_path))
    assert summary["stage_count"] == 2
    assert isinstance(summary["plan_run_id"], str)
    assert summary["plan_run_id"]
    assert (output_root / "plan_summary.json").exists()
    assert (output_root / "plan_summary.md").exists()
