from __future__ import annotations

from collections.abc import Sequence


def collect_metric_keys(stage_summaries: Sequence[dict[str, object]]) -> list[str]:
    keys: set[str] = set()
    for summary in stage_summaries:
        metrics = summary.get("metrics", {})
        if isinstance(metrics, dict):
            keys.update(str(key) for key in metrics)
    return sorted(keys)


def build_stage_plan_markdown(plan_name: str, stage_summaries: Sequence[dict[str, object]]) -> str:
    metric_keys = collect_metric_keys(stage_summaries)
    lines = [f"# {plan_name}", "", "## Stage Summary", ""]
    for summary in stage_summaries:
        lines.append(f"### {summary['name']}")
        description = summary.get("description")
        if isinstance(description, str) and description:
            lines.extend([description, ""])
        lines.append(f"- config: `{summary['train_config']}`")
        lines.append(f"- steps: `{summary['global_step']}`")
        lines.append(f"- avg_loss: `{summary['avg_loss']}`")
        lines.append(f"- throughput: `{summary['throughput']}`")
        checkpoint_dir = summary.get("checkpoint_dir")
        if isinstance(checkpoint_dir, str) and checkpoint_dir:
            lines.append(f"- checkpoint: `{checkpoint_dir}`")
        metrics = summary.get("metrics", {})
        if isinstance(metrics, dict) and metrics:
            lines.append("- metrics:")
            for key in metric_keys:
                if key in metrics:
                    lines.append(f"  - `{key}`: `{metrics[key]}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
