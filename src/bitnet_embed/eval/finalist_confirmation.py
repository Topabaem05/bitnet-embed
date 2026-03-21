from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bitnet_embed.export.hf_package import export_hf_package
from bitnet_embed.utils.io import dump_json, ensure_dir, load_json, load_yaml


def _sanitize_slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "-" for character in value
    )


def _to_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trial_name": str(raw.get("trial_name", "")),
        "run_id": str(raw.get("run_id", "")),
        "checkpoint_dir": str(raw.get("checkpoint_dir", "")),
        "metric_value": raw.get("metric_value"),
        "rank": raw.get("rank"),
        "config_path": str(raw.get("config_path", "")),
        "resume_from_checkpoint": raw.get("resume_from_checkpoint"),
        "package_manifest_path": None,
    }


def _final_rung(summary: Mapping[str, Any]) -> dict[str, Any]:
    rungs = summary.get("rungs")
    if not isinstance(rungs, list) or not rungs:
        return {}
    final_rung = rungs[-1]
    return final_rung if isinstance(final_rung, dict) else {}


def _rank_key(record: Mapping[str, Any]) -> tuple[float, str]:
    rank_value = record.get("rank")
    rank = float(rank_value) if isinstance(rank_value, (int, float)) else float("inf")
    return rank, str(record.get("trial_name", ""))


def resolve_finalists(
    summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_rung = _final_rung(summary)
    ranked_trials: list[dict[str, Any]] = []
    raw_ranked = final_rung.get("ranked_trials")
    if isinstance(raw_ranked, list):
        ranked_trials = [item for item in raw_ranked if isinstance(item, dict)]
        ranked_trials = sorted(ranked_trials, key=_rank_key)

    candidates = [_to_record(trial) for trial in ranked_trials]

    finalist_names = final_rung.get("finalists")
    if isinstance(finalist_names, list) and finalist_names:
        selected = {str(name) for name in finalist_names if isinstance(name, str) and name}
        finalists = [
            _to_record(trial) for trial in ranked_trials if str(trial.get("trial_name")) in selected
        ]
        if finalists:
            return candidates, finalists

    best_trial = summary.get("best_trial")
    if isinstance(best_trial, dict):
        best_record = _to_record(best_trial)
        if not candidates:
            candidates = [best_record]
        return candidates, [best_record]

    return candidates, []


def build_confirmation_markdown(summary: Mapping[str, Any]) -> str:
    lines = [f"# {summary['confirmation_name']}", "", "## Overview", ""]
    lines.append(f"- `search_name`: `{summary['search_name']}`")
    lines.append(f"- `search_run_id`: `{summary['search_run_id']}`")
    lines.append(f"- `primary_metric`: `{summary['primary_metric']}`")
    lines.append(f"- `candidate_count`: `{summary['candidate_count']}`")
    lines.append(f"- `finalist_count`: `{summary['finalist_count']}`")
    lines.append("")

    lines.append("## Finalists")
    lines.append("")
    for finalist in summary.get("finalists", []):
        if not isinstance(finalist, dict):
            continue
        lines.append(f"- trial `{finalist.get('trial_name')}`")
        lines.append(f"  - rank: `{finalist.get('rank')}`")
        lines.append(f"  - metric: `{finalist.get('metric_value')}`")
        lines.append(f"  - run_id: `{finalist.get('run_id')}`")
        lines.append(f"  - checkpoint_dir: `{finalist.get('checkpoint_dir')}`")
        lines.append(f"  - config_path: `{finalist.get('config_path')}`")
        lines.append(f"  - resume_from_checkpoint: `{finalist.get('resume_from_checkpoint')}`")
        lines.append(f"  - package_manifest_path: `{finalist.get('package_manifest_path')}`")
    if not summary.get("finalists"):
        lines.append("- none")
    lines.append("")

    lines.append("## Candidates")
    lines.append("")
    for candidate in summary.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        lines.append(
            "- "
            f"rank `{candidate.get('rank')}`: `{candidate.get('trial_name')}` "
            f"metric=`{candidate.get('metric_value')}` run_id=`{candidate.get('run_id')}`"
        )
    if not summary.get("candidates"):
        lines.append("- none")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_finalist_confirmation(config_path: str) -> dict[str, Any]:
    config = load_yaml(config_path)
    search_summary_path = config.get("search_summary")
    if not isinstance(search_summary_path, str) or not search_summary_path:
        raise RuntimeError("Finalist confirmation config requires search_summary")

    search_summary_payload = load_json(search_summary_path)
    if not isinstance(search_summary_payload, dict):
        raise RuntimeError("search_summary JSON must be a mapping")

    candidates, finalists = resolve_finalists(search_summary_payload)
    search_name = str(search_summary_payload.get("search_name", "search"))
    search_run_id = str(search_summary_payload.get("search_run_id", ""))
    output_dir = ensure_dir(Path(str(config.get("output_dir", "reports/finalists/latest"))))
    should_export_packages = bool(config.get("export_finalist_packages", False))
    package_output_root = ensure_dir(
        Path(str(config.get("package_output_root", output_dir / "packages")))
    )

    if should_export_packages:
        for index, finalist in enumerate(finalists, start=1):
            checkpoint_dir = finalist.get("checkpoint_dir")
            if not isinstance(checkpoint_dir, str) or not checkpoint_dir:
                trial_name = finalist.get("trial_name")
                raise RuntimeError(
                    f"Finalist '{trial_name}' is missing checkpoint_dir for packaging"
                )
            trial_name = str(finalist.get("trial_name", f"finalist-{index}"))
            rank_value = finalist.get("rank")
            rank = int(rank_value) if isinstance(rank_value, (int, float)) else index
            package_dir = package_output_root / f"{rank:02d}_{_sanitize_slug(trial_name)}"
            package_name = f"{_sanitize_slug(search_name)}-{_sanitize_slug(trial_name)}"
            export_hf_package(checkpoint_dir, package_dir, package_name=package_name)
            finalist["package_manifest_path"] = str(package_dir / "config.json")

    confirmation_name = str(config.get("confirmation_name", f"{search_name}-finalists"))
    payload = {
        "confirmation_name": confirmation_name,
        "config_path": config_path,
        "search_summary_path": search_summary_path,
        "search_name": search_name,
        "search_run_id": search_run_id,
        "primary_metric": str(search_summary_payload.get("primary_metric", "")),
        "maximize": bool(search_summary_payload.get("maximize", False)),
        "candidate_count": len(candidates),
        "finalist_count": len(finalists),
        "candidates": candidates,
        "finalists": finalists,
    }
    dump_json(output_dir / "finalist_confirmation.json", payload)
    (output_dir / "finalist_confirmation.md").write_text(
        build_confirmation_markdown(payload),
        encoding="utf-8",
    )
    return payload
