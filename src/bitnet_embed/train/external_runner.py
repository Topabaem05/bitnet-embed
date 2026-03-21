from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from bitnet_embed.train.trainer import TrainingSummary

_UNSET_RESUME_FROM_CHECKPOINT = object()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_external_training_command(
    config_path: str,
    *,
    mode_override: str | None = None,
    plan_name: str | None = None,
    parent_run_id: str | None = None,
    resume_from_checkpoint: str | None | object = _UNSET_RESUME_FROM_CHECKPOINT,
) -> list[str]:
    command = [
        sys.executable,
        str(_project_root() / "scripts" / "run_training.py"),
        "--config",
        config_path,
    ]
    if mode_override is not None:
        command.extend(["--mode-override", mode_override])
    if plan_name is not None:
        command.extend(["--plan-name", plan_name])
    if parent_run_id is not None:
        command.extend(["--parent-run-id", parent_run_id])
    if resume_from_checkpoint is not _UNSET_RESUME_FROM_CHECKPOINT:
        resume_value = "" if resume_from_checkpoint is None else str(resume_from_checkpoint)
        command.extend(["--resume-from-checkpoint", resume_value])
    return command


def _parse_training_summary(stdout: str) -> TrainingSummary:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to parse external training summary JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("External training summary must be a JSON object")
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise RuntimeError("External training summary metrics must be a mapping")
    return TrainingSummary(
        run_id=str(payload["run_id"]),
        global_step=int(payload["global_step"]),
        avg_loss=float(payload["avg_loss"]),
        throughput=float(payload["throughput"]),
        checkpoint_dir=(str(payload["checkpoint_dir"]) if payload.get("checkpoint_dir") else None),
        metrics={str(key): float(value) for key, value in metrics.items()},
    )


def run_training_external(
    config_path: str,
    *,
    mode_override: str | None = None,
    plan_name: str | None = None,
    parent_run_id: str | None = None,
    resume_from_checkpoint: str | None | object = _UNSET_RESUME_FROM_CHECKPOINT,
) -> TrainingSummary:
    command = build_external_training_command(
        config_path,
        mode_override=mode_override,
        plan_name=plan_name,
        parent_run_id=parent_run_id,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    completed = subprocess.run(
        command,
        cwd=_project_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return _parse_training_summary(completed.stdout)
