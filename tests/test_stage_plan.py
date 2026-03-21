from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bitnet_embed.train.plan import run_stage_plan
from bitnet_embed.train.trainer import TrainingSummary

_UNSET = object()


def test_run_stage_plan_defaults_to_no_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    calls: list[dict[str, Any]] = []

    def fake_run_training(
        config_path: str,
        *,
        mode_override: str | None = None,
        plan_name: str | None = None,
        parent_run_id: str | None = None,
        resume_from_checkpoint: str | None | object = _UNSET,
    ) -> TrainingSummary:
        calls.append(
            {
                "config_path": config_path,
                "mode_override": mode_override,
                "plan_name": plan_name,
                "parent_run_id": parent_run_id,
                "resume_from_checkpoint": resume_from_checkpoint,
            }
        )
        return TrainingSummary(
            run_id=f"run-{len(calls)}",
            global_step=1,
            avg_loss=0.5,
            throughput=1.0,
            checkpoint_dir=f"/checkpoints/stage-{len(calls)}",
            metrics={},
        )

    monkeypatch.setattr("bitnet_embed.train.plan.run_training", fake_run_training)

    summary = run_stage_plan(str(config_path))
    assert summary["stage_count"] == 2
    assert isinstance(summary["plan_run_id"], str)
    assert summary["plan_run_id"]
    assert len(calls) == 2
    assert calls[0]["resume_from_checkpoint"] is _UNSET
    assert calls[1]["resume_from_checkpoint"] is _UNSET
    stages = summary["stages"]
    assert isinstance(stages, list)
    assert stages[0]["resume_policy"] == "none"
    assert stages[0]["resume_handoff"] == "none"
    assert stages[0]["resume_from_checkpoint"] is None
    assert stages[1]["resume_policy"] == "none"
    assert stages[1]["resume_handoff"] == "none"
    assert stages[1]["resume_from_checkpoint"] is None
    assert (output_root / "plan_summary.json").exists()
    assert (output_root / "plan_summary.md").exists()


def test_run_stage_plan_hands_off_previous_stage_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "stage-plan"
    config_path = tmp_path / "plan.yaml"
    config_path.write_text(
        "\n".join(
            [
                "plan_name: tmp_stage_plan",
                f"output_root: {output_root}",
                "stages:",
                "  - name: stage1",
                "    train_config: configs/train/smoke.yaml",
                "  - name: stage2",
                "    train_config: configs/train/smoke_retrieval.yaml",
                "    resume_policy: previous_stage_checkpoint",
                "  - name: stage3",
                "    train_config: configs/train/smoke.yaml",
                "    resume_policy: previous_stage_checkpoint",
            ]
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, Any]] = []
    checkpoints = ["/checkpoints/stage-1", None, "/checkpoints/stage-3"]

    def fake_run_training(
        config_path: str,
        *,
        mode_override: str | None = None,
        plan_name: str | None = None,
        parent_run_id: str | None = None,
        resume_from_checkpoint: str | None | object = _UNSET,
    ) -> TrainingSummary:
        _ = (mode_override, plan_name, parent_run_id)
        stage_index = len(calls)
        calls.append(
            {
                "config_path": config_path,
                "resume_from_checkpoint": resume_from_checkpoint,
            }
        )
        return TrainingSummary(
            run_id=f"run-{stage_index + 1}",
            global_step=1,
            avg_loss=0.5,
            throughput=1.0,
            checkpoint_dir=checkpoints[stage_index],
            metrics={},
        )

    monkeypatch.setattr("bitnet_embed.train.plan.run_training", fake_run_training)

    summary = run_stage_plan(str(config_path))
    assert summary["stage_count"] == 3
    assert len(calls) == 3
    assert calls[0]["resume_from_checkpoint"] is _UNSET
    assert calls[1]["resume_from_checkpoint"] == "/checkpoints/stage-1"
    assert calls[2]["resume_from_checkpoint"] is None

    stages = summary["stages"]
    assert isinstance(stages, list)
    assert stages[1]["resume_policy"] == "previous_stage_checkpoint"
    assert stages[1]["resume_handoff"] == "previous_stage_checkpoint"
    assert stages[1]["resume_from_checkpoint"] == "/checkpoints/stage-1"
    assert stages[2]["resume_policy"] == "previous_stage_checkpoint"
    assert stages[2]["resume_handoff"] == "previous_stage_checkpoint_missing"
    assert stages[2]["resume_from_checkpoint"] is None
