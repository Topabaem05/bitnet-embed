from __future__ import annotations

from pathlib import Path
from typing import Any

from bitnet_embed.utils.io import dump_json, ensure_dir, load_json, load_yaml


def load_optional_json(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    payload = load_json(target)
    return payload if isinstance(payload, dict) else {}


def summarize_stages(stage_plan: dict[str, Any]) -> dict[str, Any]:
    stages = stage_plan.get("stages", [])
    if not isinstance(stages, list) or not stages:
        return {}
    best_stage = min(
        stages,
        key=lambda stage: float(stage.get("avg_loss", float("inf"))),
    )
    return {
        "plan_name": stage_plan.get("plan_name"),
        "stage_count": stage_plan.get("stage_count", len(stages)),
        "stage_names": [stage.get("name") for stage in stages],
        "best_stage_by_loss": best_stage.get("name"),
        "best_stage_loss": best_stage.get("avg_loss"),
    }


def build_report_markdown(bundle: dict[str, Any]) -> str:
    lines = [f"# {bundle['report_name']}", "", "## Overview", ""]
    stage_summary = bundle.get("stage_summary", {})
    if stage_summary:
        lines.append(f"- plan: `{stage_summary.get('plan_name')}`")
        lines.append(f"- stage_count: `{stage_summary.get('stage_count')}`")
        lines.append(f"- best_stage_by_loss: `{stage_summary.get('best_stage_by_loss')}`")
        lines.append("")
    latency = bundle.get("latency", {})
    if latency:
        lines.append("## Latency")
        lines.append("")
        for key, value in latency.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    sts = bundle.get("sts", {})
    if sts:
        lines.append("## STS")
        lines.append("")
        for key, value in sts.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    memory = bundle.get("memory", {})
    if memory:
        lines.append("## Memory")
        lines.append("")
        for key, value in memory.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    ann = bundle.get("ann", {})
    if ann:
        lines.append("## ANN")
        lines.append("")
        for key, value in ann.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    package = bundle.get("package", {})
    if package:
        lines.append("## Package")
        lines.append("")
        lines.append(f"- `package_name`: `{package.get('package_name')}`")
        lines.append(f"- `format`: `{package.get('format')}`")
        metrics = package.get("metrics", {})
        if isinstance(metrics, dict) and metrics:
            lines.append("- metrics:")
            for key, value in metrics.items():
                lines.append(f"  - `{key}`: `{value}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_report_bundle(config_path: str) -> dict[str, Any]:
    config = load_yaml(config_path)
    report_name = str(config.get("report_name", "bitnet-embed-report"))
    output_dir = Path(str(config.get("output_dir", "reports/latest")))
    ensure_dir(output_dir)
    stage_plan = load_optional_json(str(config["stage_plan"]) if config.get("stage_plan") else None)
    sts = load_optional_json(str(config["sts_report"]) if config.get("sts_report") else None)
    latency = load_optional_json(
        str(config["latency_report"]) if config.get("latency_report") else None
    )
    memory = load_optional_json(
        str(config["memory_report"]) if config.get("memory_report") else None
    )
    ann = load_optional_json(str(config["ann_report"]) if config.get("ann_report") else None)
    package = load_optional_json(
        str(config["package_manifest"]) if config.get("package_manifest") else None
    )
    bundle = {
        "report_name": report_name,
        "stage_summary": summarize_stages(stage_plan),
        "sts": sts,
        "latency": latency,
        "memory": memory,
        "ann": ann,
        "package": package,
    }
    dump_json(output_dir / "summary.json", bundle)
    (output_dir / "summary.md").write_text(build_report_markdown(bundle), encoding="utf-8")
    return bundle
