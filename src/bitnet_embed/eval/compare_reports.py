from __future__ import annotations

from pathlib import Path
from typing import Any

from bitnet_embed.utils.io import dump_json, ensure_dir, load_json, load_yaml


def flatten_report(report: dict[str, Any]) -> dict[str, float | str]:
    flat: dict[str, float | str] = {}
    stage_summary = report.get("stage_summary", {})
    if isinstance(stage_summary, dict):
        for key, value in stage_summary.items():
            if isinstance(value, (int, float, str)):
                flat[f"stage_{key}"] = value
    for section in ("latency", "memory", "ann"):
        values = report.get(section, {})
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    flat[f"{section}_{key}"] = value
    package = report.get("package", {})
    if isinstance(package, dict):
        flat["package_name"] = str(package.get("package_name", ""))
        metrics = package.get("metrics", {})
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    flat[f"package_{key}"] = value
    return flat


def run_report_comparison(config_path: str) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_dir = Path(str(config.get("output_dir", "reports/comparisons")))
    ensure_dir(output_dir)
    reports = []
    for entry in config.get("reports", []):
        report = load_json(str(entry["path"]))
        reports.append({"name": str(entry["name"]), "metrics": flatten_report(report)})
    comparison = {
        "comparison_name": str(config.get("comparison_name", "report-comparison")),
        "reports": reports,
    }
    dump_json(output_dir / "comparison.json", comparison)
    lines = [f"# {comparison['comparison_name']}", "", "## Reports", ""]
    for report in reports:
        lines.append(f"### {report['name']}")
        metrics = report.get("metrics", {})
        if isinstance(metrics, dict):
            for key, value in sorted(metrics.items()):
                lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    (output_dir / "comparison.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return comparison
