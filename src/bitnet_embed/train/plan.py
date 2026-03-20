from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from bitnet_embed.eval.reporting import build_stage_plan_markdown
from bitnet_embed.train.workflow import run_training
from bitnet_embed.utils.io import dump_json, ensure_dir, load_yaml


@dataclass(slots=True)
class StageSpec:
    name: str
    train_config: str
    description: str = ""
    mode_override: str | None = None


def load_stage_specs(payload: dict[str, Any]) -> tuple[str, list[StageSpec], str]:
    plan_name = str(payload.get("plan_name", "bitnet_stage_plan"))
    output_root = str(payload.get("output_root", f"runs/plans/{plan_name}"))
    stages = [
        StageSpec(
            name=str(stage["name"]),
            train_config=str(stage["train_config"]),
            description=str(stage.get("description", "")),
            mode_override=(str(stage["mode_override"]) if stage.get("mode_override") else None),
        )
        for stage in payload.get("stages", [])
    ]
    if not stages:
        raise RuntimeError("At least one stage must be configured")
    return plan_name, stages, output_root


def run_stage_plan(config_path: str) -> dict[str, object]:
    payload = load_yaml(config_path)
    plan_name, stages, output_root = load_stage_specs(payload)
    plan_run_id = f"{plan_name}-{uuid4().hex[:12]}"
    output_dir = ensure_dir(output_root)
    stage_summaries: list[dict[str, object]] = []
    for stage in stages:
        summary = run_training(
            stage.train_config,
            mode_override=stage.mode_override,
            plan_name=plan_name,
            parent_run_id=plan_run_id,
        )
        stage_summaries.append(
            {
                "name": stage.name,
                "description": stage.description,
                "train_config": stage.train_config,
                "mode_override": stage.mode_override,
                **asdict(summary),
            }
        )
    plan_summary: dict[str, object] = {
        "plan_name": plan_name,
        "plan_run_id": plan_run_id,
        "config_path": config_path,
        "stage_count": len(stage_summaries),
        "stages": stage_summaries,
    }
    dump_json(Path(output_dir) / "plan_summary.json", plan_summary)
    markdown = build_stage_plan_markdown(plan_name, stage_summaries)
    Path(output_dir, "plan_summary.md").write_text(markdown, encoding="utf-8")
    return plan_summary
