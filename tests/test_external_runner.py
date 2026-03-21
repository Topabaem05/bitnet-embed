from __future__ import annotations

import json
import subprocess
import sys

import pytest

from bitnet_embed.train.external_runner import (
    _UNSET_RESUME_FROM_CHECKPOINT,
    build_external_training_command,
    run_training_external,
)


def test_build_external_training_command_includes_expected_arguments() -> None:
    command = build_external_training_command(
        "configs/train/smoke.yaml",
        mode_override="head_only",
        plan_name="plan-a",
        parent_run_id="parent-1",
        resume_from_checkpoint="runs/checkpoints/step-00001",
    )
    assert command[0] == sys.executable
    assert command[1].endswith("scripts/run_training.py")
    assert command[2:] == [
        "--config",
        "configs/train/smoke.yaml",
        "--mode-override",
        "head_only",
        "--plan-name",
        "plan-a",
        "--parent-run-id",
        "parent-1",
        "--resume-from-checkpoint",
        "runs/checkpoints/step-00001",
    ]


def test_build_external_training_command_can_clear_resume_checkpoint() -> None:
    command = build_external_training_command(
        "configs/train/smoke.yaml",
        resume_from_checkpoint=None,
    )
    assert command[-2:] == ["--resume-from-checkpoint", ""]

    command_without_resume = build_external_training_command(
        "configs/train/smoke.yaml",
        resume_from_checkpoint=_UNSET_RESUME_FROM_CHECKPOINT,
    )
    assert "--resume-from-checkpoint" not in command_without_resume


def test_run_training_external_parses_training_summary_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_subprocess_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = args
        _ = kwargs
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "run_id": "run-1",
                    "global_step": 3,
                    "avg_loss": 0.4,
                    "throughput": 2.5,
                    "checkpoint_dir": "/tmp/checkpoint",
                    "metrics": {"score": 0.9},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("bitnet_embed.train.external_runner.subprocess.run", fake_subprocess_run)
    summary = run_training_external("configs/train/smoke.yaml")
    assert summary.run_id == "run-1"
    assert summary.global_step == 3
    assert summary.metrics["score"] == 0.9


def test_run_training_external_raises_for_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_subprocess_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = args
        _ = kwargs
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="not-json", stderr="")

    monkeypatch.setattr("bitnet_embed.train.external_runner.subprocess.run", fake_subprocess_run)
    with pytest.raises(RuntimeError, match="parse external training summary"):
        run_training_external("configs/train/smoke.yaml")
