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
        lines.append(f"- resume_policy: `{summary.get('resume_policy', 'none')}`")
        lines.append(f"- resume_handoff: `{summary.get('resume_handoff', 'none')}`")
        resume_from_checkpoint = summary.get("resume_from_checkpoint")
        lines.append(
            f"- resume_from_checkpoint: `{resume_from_checkpoint}`"
            if resume_from_checkpoint
            else "- resume_from_checkpoint: `none`"
        )
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
